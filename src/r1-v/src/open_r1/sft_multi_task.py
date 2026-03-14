import os
os.environ.setdefault("WANDB_MODE", "offline")

import os
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
TREEVGR_ROOT = os.path.join(ROOT, "treevgr")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from src.open_r1.vision_process import process_vision_info

from datasets import Dataset, DatasetDict

import wandb
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import torch.nn.functional as F


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended ScriptArguments with auxiliary loss options."""
    aux_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for auxiliary loss (bbox + time regression). 0.0 to disable."}
    )
    bbox_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for bbox regression loss within aux loss."}
    )
    time_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for time regression loss within aux loss."}
    )
    aux_loss_type: str = field(
        default="l1",
        metadata={"help": "Type of regression loss: 'l1' or 'mse'"}
    )


def extract_bbox_and_time_from_text(text: str):
    """Extract bbox coordinates and time values from text.
    
    Returns:
        bboxes: List of [x1, y1, x2, y2] coordinates
        times: List of time values in seconds
    """
    import re
    
    # Extract bboxes: <box>[x1,y1,x2,y2]</box>
    bbox_pattern = r'<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>'
    bbox_matches = re.findall(bbox_pattern, text)
    bboxes = [[int(x) for x in match] for match in bbox_matches]
    
    # Extract times: <t>time</t> or at<t>time</t>s
    time_pattern = r'<t>(\d+(?:\.\d+)?)</t>'
    time_matches = re.findall(time_pattern, text)
    times = [float(t) for t in time_matches]
    
    return bboxes, times


def compute_aux_regression_loss(
    pred_ids: torch.Tensor,
    label_ids: torch.Tensor,
    tokenizer,
    bbox_weight: float = 1.0,
    time_weight: float = 1.0,
    loss_type: str = "l1"
) -> torch.Tensor:
    """Compute auxiliary regression loss for bbox and time predictions.
    
    Args:
        pred_ids: Predicted token ids (argmax of logits)
        label_ids: Ground truth token ids
        tokenizer: Tokenizer for decoding
        bbox_weight: Weight for bbox loss
        time_weight: Weight for time loss
        loss_type: 'l1' or 'mse'
    
    Returns:
        Auxiliary loss tensor
    """
    device = pred_ids.device
    batch_size = pred_ids.shape[0]
    
    total_bbox_loss = torch.tensor(0.0, device=device)
    total_time_loss = torch.tensor(0.0, device=device)
    valid_bbox_count = 0
    valid_time_count = 0
    
    for b in range(batch_size):
        # Decode predictions and labels
        pred_text = tokenizer.decode(pred_ids[b], skip_special_tokens=True)
        
        # For labels, replace -100 with pad token for decoding
        valid_labels = label_ids[b].clone()
        valid_labels[valid_labels == -100] = tokenizer.pad_token_id
        label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
        
        # Extract bbox and time from both
        pred_bboxes, pred_times = extract_bbox_and_time_from_text(pred_text)
        gt_bboxes, gt_times = extract_bbox_and_time_from_text(label_text)
        
        # Compute bbox loss (IoU-based or L1)
        min_bbox_len = min(len(pred_bboxes), len(gt_bboxes))
        if min_bbox_len > 0:
            for i in range(min_bbox_len):
                pred_bbox = torch.tensor(pred_bboxes[i], dtype=torch.float32, device=device)
                gt_bbox = torch.tensor(gt_bboxes[i], dtype=torch.float32, device=device)
                
                if loss_type == "l1":
                    bbox_loss = F.l1_loss(pred_bbox, gt_bbox)
                else:  # mse
                    bbox_loss = F.mse_loss(pred_bbox, gt_bbox)
                
                total_bbox_loss += bbox_loss
                valid_bbox_count += 1
        
        # Compute time loss
        min_time_len = min(len(pred_times), len(gt_times))
        if min_time_len > 0:
            for i in range(min_time_len):
                pred_time = torch.tensor(pred_times[i], dtype=torch.float32, device=device)
                gt_time = torch.tensor(gt_times[i], dtype=torch.float32, device=device)
                
                if loss_type == "l1":
                    time_loss = F.l1_loss(pred_time, gt_time)
                else:  # mse
                    time_loss = F.mse_loss(pred_time, gt_time)
                
                total_time_loss += time_loss
                valid_time_count += 1
    
    # Average losses
    if valid_bbox_count > 0:
        total_bbox_loss = total_bbox_loss / valid_bbox_count
    if valid_time_count > 0:
        total_time_loss = total_time_loss / valid_time_count
    
    # Combine losses
    aux_loss = bbox_weight * total_bbox_loss + time_weight * total_time_loss
    
    return aux_loss, total_bbox_loss, total_time_loss


# Global variable for aux loss config (set in main)
AUX_LOSS_CONFIG = {
    "enabled": False,
    "weight": 0.0,
    "bbox_weight": 1.0,
    "time_weight": 1.0,
    "loss_type": "l1",
    "tokenizer": None
}


def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    if example['task'] == 'visual QA':
        system_message = "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."
        image_root = TREEVGR_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(image_root,example['image_path'])
                    },
                    {
                        "type": "text",
                        "text": question,
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "image_size": example["image_size"], "task": "visual QA", "source": example["source"], "key_frames":[]}
    
    elif example['task'] == 'temporal-spatial free-form QA':
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`."
        question = example['question']
        video_root = STR_DATA
        if example['source'] == "STR_plm_rdcap":
            video_root = STR_PLM_DATA
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]    
        return {"messages": messages, "key_frames": example["key_frames"], "task": "temporal-spatial free-form QA", "source": example["source"], "image_size":[]}
    
    elif example['task'] == 'temporal QA':
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'"
        video_root = TVG_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "task": "temporal QA", "source": example["source"], "key_frames":[], "image_size":[]}
    elif example["task"] == "General video QA MCQ":
        system_message = "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Only output the correct option in the <answer> </answer> section."
        video_root = GENERAL_VIDEO_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "task": "General video QA MCQ", "source": example["source"], "key_frames":[], "image_size":[]}
    elif example["task"] == "General video QA Free-form":
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
        video_root = GENERAL_VIDEO_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "task": "General video QA Free-form", "source": example["source"], "key_frames":[], "image_size":[]}
    
    raise ValueError(f"Unknown task: {example['task']}")


def convert_coord_format_espressso(bbox, image_size):
    # for videoespresso
    # image_size: (W, H)
    nx, ny, nw, nh = [coord / 1000.0 for coord in bbox]
    x_center = nx * image_size[0]
    y_center = ny * image_size[1]
    width = nw * image_size[0]
    height = nh * image_size[1]

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_size[0], x_max)
    y_max = min(image_size[1], y_max)

    return [x_min, y_min, x_max, y_max]

def convert_coord_format_gemini(coords, image_size):
    # for gemini annotated data
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = coords
    width, height = image_size
    real_x_min = norm_x_min * width
    real_y_min = norm_y_min * height
    real_x_max = norm_x_max * width
    real_y_max = norm_y_max * height
    return [real_x_min, real_y_min, real_x_max, real_y_max]

import re
def resize_bounding_boxes_for_image(text: str, old_image_size: tuple, new_image_size: tuple) -> str:

    old_w, old_h = old_image_size
    new_w, new_h = new_image_size
    ratios = (new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h)

    def resizer(match: re.Match) -> str:
        coords = [int(c) for c in match.group(1).strip('[]').split(',')]
        new_coords = [int(round(c * r)) for c, r in zip(coords, ratios)]
        return f"<box>[{','.join(map(str, new_coords))}]</box>"

    return re.sub(r"<box>(\[.*?\])</box>", resizer, text)

def replace_boxes_for_videoespresso(text, image_size):
    import re
    pattern = re.compile(r'<box>\[([^]]+)\]</box>')
    
    def replacer(match):
        box_str = match.group(1)
        coords = list(map(float, box_str.split(',')))
        new_coords = convert_coord_format_espressso(coords, image_size)
        new_coords = str([round(coord) for coord in new_coords])
        new_coords = new_coords.replace(" ","")
        return '<box>' + new_coords + '</box>'
    
    return pattern.sub(replacer, text)


def replace_boxes_for_gemini_data(text, image_size):
    import re
    pattern = re.compile(r'<box>\[([^]]+)\]</box>')
    
    def replacer(match):
        box_str = match.group(1)
        coords = list(map(float, box_str.split(',')))
        new_coords = convert_coord_format_gemini(coords, image_size)
        new_coords = str([round(coord) for coord in new_coords])
        new_coords = new_coords.replace(" ","")
        return '<box>' + new_coords + '</box>'
    
    return pattern.sub(replacer, text)

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []

    for i, example in enumerate(examples):
        try:
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")
        
    # batch size must be 1
    assert len(texts) == 1

    if example["task"] == "visual QA":
        old_image_size = example["image_size"]
        new_image_size = [image_inputs[0].size[0], image_inputs[0].size[1]] # W * H
        texts[0] = resize_bounding_boxes_for_image(texts[0], old_image_size, new_image_size)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            return_tensors="pt",
            padding=True
        )
        
    elif example["task"] == "temporal-spatial free-form QA":

        width, height = video_inputs[0].size(3), video_inputs[0].size(2)
        image_size = (width, height)

        # Here, we need to add key frames.
        key_frame_root = STR_KF_ROOT
        if example['source'] == "STR_plm_rdcap":
            key_frame_root = STR_PLM_KF_ROOT

        key_frames = []

        for key_frame in example["key_frames"]:
            kf_path = os.path.join(key_frame_root, key_frame["path"])
            kf = Image.open(kf_path)
            kf = kf.convert('RGB')
            resized_kf = kf.resize(image_size)
            resized_kf= np.array(resized_kf)
            resized_kf = np.transpose(resized_kf, (2, 0, 1))
            resized_kf = torch.from_numpy(resized_kf)
            key_frames.append((key_frame["time"], resized_kf))
        
        frame_prompt = ""
        refined_image_inputs = []
        kf_idx = 0
        ori_idx = 0
        frame_idx = 1
        while ori_idx < len(video_inputs[0]):
            time_now = int(ori_idx / video_kwargs['fps'][0])
            if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                refined_image_inputs.append(key_frames[kf_idx][1])
                time_now = key_frames[kf_idx][0]
                frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"   
                kf_idx += 1
            else:
                refined_image_inputs.append(video_inputs[0][ori_idx])
                time_now = round(ori_idx / video_kwargs['fps'][0],1)
                frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"       
                ori_idx += 1
            frame_idx += 1

        refined_image_inputs = torch.stack(refined_image_inputs)
        texts[0] = texts[0].replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
        texts[0] = replace_boxes_for_gemini_data(texts[0], image_size)
        # print(refined_image_inputs.shape, texts[0])

        inputs = processor(
            text=texts,
            images=[refined_image_inputs],   # (16+k)*3*h*w
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False
        )

    elif example["task"] == "temporal QA" or "General video QA" in example["task"]:
        frame_prompt = ""
        ori_idx = 0
        while ori_idx < len(video_inputs[0]):
            time_now = round(ori_idx / video_kwargs['fps'][0], 1)
            frame_prompt += f"Frame {ori_idx + 1} at {time_now}: <|vision_start|><|image_pad|><|vision_end|>\n"    
            ori_idx += 1
        frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
        texts[0] = texts[0].replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)

        # print(texts[0])

        inputs = processor(
            text=texts,
            images=video_inputs,
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False
        )
    else:
        raise ValueError(f"Unknown task: {example['task']}")

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs


class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, aux_loss_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_config = aux_loss_config or AUX_LOSS_CONFIG

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch) 
        
        # Add auxiliary loss if enabled
        if self.aux_loss_config.get("enabled", False) and self.aux_loss_config.get("weight", 0.0) > 0:
            try:
                # Get predicted token ids from logits
                logits = outputs.logits  # (batch, seq_len, vocab_size)
                pred_ids = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                labels = inputs.get("labels")
                
                if labels is not None:
                    aux_loss, bbox_loss, time_loss = compute_aux_regression_loss(
                        pred_ids=pred_ids,
                        label_ids=labels,
                        tokenizer=self.aux_loss_config["tokenizer"],
                        bbox_weight=self.aux_loss_config.get("bbox_weight", 1.0),
                        time_weight=self.aux_loss_config.get("time_weight", 1.0),
                        loss_type=self.aux_loss_config.get("loss_type", "l1")
                    )
                    
                    # Add weighted auxiliary loss
                    aux_weight = self.aux_loss_config.get("weight", 0.0)
                    loss = loss + aux_weight * aux_loss
                    
                    # Log auxiliary losses
                    if self.state.global_step % self.args.logging_steps == 0:
                        self.log({
                            "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                            "bbox_loss": bbox_loss.item() if isinstance(bbox_loss, torch.Tensor) else bbox_loss,
                            "time_loss": time_loss.item() if isinstance(time_loss, torch.Tensor) else time_loss,
                        })
            except Exception as e:
                # If aux loss fails, just use the original loss
                if self.state.global_step % 100 == 0:
                    print(f"Warning: Auxiliary loss computation failed: {e}")
        
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # Parse arguments with custom script arguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure auxiliary loss
    aux_loss_config = {
        "enabled": script_args.aux_loss_weight > 0,
        "weight": script_args.aux_loss_weight,
        "bbox_weight": script_args.bbox_loss_weight,
        "time_weight": script_args.time_loss_weight,
        "loss_type": script_args.aux_loss_type,
        "tokenizer": None  # Will be set after processor is loaded
    }
    
    if aux_loss_config["enabled"]:
        print(f"\n{'='*60}")
        print("Auxiliary Loss Configuration:")
        print(f"  - Aux Loss Weight: {aux_loss_config['weight']}")
        print(f"  - BBox Loss Weight: {aux_loss_config['bbox_weight']}")
        print(f"  - Time Loss Weight: {aux_loss_config['time_weight']}")
        print(f"  - Loss Type: {aux_loss_config['loss_type']}")
        print(f"{'='*60}\n")
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=None,  # Changed for ZeRO-3 compatibility
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Set tokenizer in aux_loss_config
    aux_loss_config["tokenizer"] = processor.tokenizer

    # Prepare dataset
    from tqdm import tqdm 
    prepared_dataset = [prepare_dataset(example) for example in tqdm(dataset['train'], desc="Preparing dataset")]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer with auxiliary loss config
    trainer = MySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        aux_loss_config=aux_loss_config,
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
