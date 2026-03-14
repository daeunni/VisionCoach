"""
GRPO Trainer with IDL (Iterative Distillation Learning) Integration

Extends Qwen2VLGRPOTrainer to implement:
  1. Reward threshold check (k1) → Easy/Hard question classification
  2. VP Selector decision (raw vs darken VP) for hard questions
  3. VP-guided re-generation and reward computation
  4. Self-distillation loss (IDL) on positive responses
  5. Combined loss: L_A2D = -J(θ) + α * L'_IDL(θ) * I[k1 > R̄]

Reference:
  - L'_IDL = -(1/|Y'_Sel|) * Σ log π_θ(y'_j | x_i, p)
  - Y'_Sel = selected positive responses from VP-guided generation
  - q = min(N_pos, k2 * n_rollout)
"""

import os
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
GQA_ROOT = os.path.join(ROOT, "gqa")
TIMERFT_ROOT = os.path.join(ROOT, "timerft")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
VIDEO_ESPRESSO_KF_ROOT = os.path.join(ROOT, "videoespresso/kfs")
VIDEO_ESPRESSO_ROOT = os.path.join(ROOT, "videoespresso/videos")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

import copy
import json
import re
import torch
import numpy as np
from PIL import Image
from typing import Any, Optional, Union
from collections import defaultdict

from transformers import GenerationConfig, PreTrainedModel

from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from src.open_r1.vision_process import process_vision_info
from src.open_r1.vp_selector_utils import (
    VP_HINT_DARKEN,
    VP_HINT_RAW,
    VP_HINT_NUMPRO,
    get_key_object_from_json,
)

VP_HINT_MAP = {
    "darken": VP_HINT_DARKEN,
    "circle": (
        "\n[Hint: The key frames have been visually enhanced — key objects are highlighted "
        "with circles to draw attention to the most relevant areas. Focus on the circled "
        "regions and track these key objects across different time points to answer the question.]"
    ),
    "api_prompt": (
        "\n[Hint: The key frames have been visually enhanced with attention-guiding prompts "
        "that highlight the most task-relevant regions. Focus on the emphasized areas "
        "and track these key objects across different time points to answer the question.]"
    ),
    "numpro": VP_HINT_NUMPRO,
    "raw": VP_HINT_RAW,
}

from .grpo_trainer import Qwen2VLGRPOTrainer


class Qwen2VLGRPOTrainerIDL(Qwen2VLGRPOTrainer):
    """
    GRPO Trainer extended with IDL (Iterative Distillation Learning).

    When the average reward for a sample's rollouts falls below ``idl_k1``,
    the IDL path is activated:
      1. VP Selector prediction is looked up (pre-computed).
      2. If VP=1 (darken), key-frame images are darkened outside key-object bboxes.
      3. A VP guidance hint is appended to the prompt.
      4. New completions are generated with the VP-guided input.
      5. Rewards are computed; positive responses are selected (up to k2 * n_rollout).
      6. NLL (self-distillation) loss is computed on positive responses.
      7. Final loss = GSPO_loss + α * IDL_loss.
    """

    def __init__(
        self,
        # Same args as parent
        model,
        reward_funcs,
        args=None,
        script_args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        max_pixels=12845056,
        min_pixels=3136,
        attn_implementation="flash_attention_2",
        gspo=True,
        # IDL-specific args
        enable_idl: bool = False,
        idl_k1: float = 0.5,
        idl_k2: float = 0.5,
        idl_alpha: float = 0.1,
        vp_predictions: Optional[dict] = None,
        darken_keyframe_dir: Optional[str] = None,
        vp_keyframe_base_dir: Optional[str] = None,
        key_object_json_path: Optional[str] = None,
        wo_vp_selector: bool = False,
        idl_candidate_mode: str = "ans_only",  # "ans_only", "grounding_only", "ans_or_grounding", "total_reward"
        idl_positive_ranking: str = "ans_acc",  # "ans_acc", "grounding", "total_reward"
        idl_top_k: int = 2,  # number of top positive responses (top-1, top-2, top-3, ...)
        numpro_keyframe_dir: Optional[str] = None,
        temporal_reward_threshold: float = 0.9225,
        spatial_reward_threshold: float = 0.3107,
        idl_stgr_filter_json: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            script_args=script_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            attn_implementation=attn_implementation,
            gspo=gspo,
        )

        # IDL configuration
        self.enable_idl = enable_idl
        self.idl_k1 = idl_k1
        self.idl_k2 = idl_k2
        self.idl_alpha = idl_alpha
        self.vp_predictions = vp_predictions or {}
        self.darken_keyframe_dir = darken_keyframe_dir
        self.vp_keyframe_base_dir = vp_keyframe_base_dir
        self.key_object_json_path = key_object_json_path
        self.wo_vp_selector = wo_vp_selector
        self.idl_candidate_mode = idl_candidate_mode
        self.idl_positive_ranking = idl_positive_ranking
        self.idl_top_k = idl_top_k

        self.idl_stgr_filter_ids = None
        if idl_stgr_filter_json and os.path.isfile(idl_stgr_filter_json):
            with open(idl_stgr_filter_json, "r") as f:
                stgr_data = json.load(f)
            self.idl_stgr_filter_ids = frozenset(item.get("id") for item in stgr_data if item.get("id"))
            print(f"[IDL] STGR filter: IDL only for {len(self.idl_stgr_filter_ids)} ids from {idl_stgr_filter_json}")
        elif idl_stgr_filter_json:
            print(f"[IDL] STGR filter path not found: {idl_stgr_filter_json}, IDL runs on all samples.")

        if enable_idl:
            print(f"[IDL] Enabled: k1={idl_k1}, k2={idl_k2}, alpha={idl_alpha}")
            print(f"[IDL] VP predictions loaded: {len(self.vp_predictions)} samples")
            print(f"[IDL] wo_vp_selector: {self.wo_vp_selector}")
            print(f"[IDL] Candidate mode: {idl_candidate_mode}, Positive ranking: {idl_positive_ranking}, top_k: {idl_top_k}")
            if vp_keyframe_base_dir:
                print(f"[IDL] VP keyframe base directory: {vp_keyframe_base_dir}")
            if darken_keyframe_dir:
                print(f"[IDL] Darken keyframe directory (legacy): {darken_keyframe_dir}")
            if key_object_json_path:
                print(f"[IDL] Key object JSON: {key_object_json_path}")

    # -----------------------------------------------------------------
    # compute_loss: full override with GSPO + IDL
    # -----------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # ==============================================================
        # Phase 1: Normal GSPO (identical to parent's compute_loss) 
        # ==============================================================
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        input_copy = [copy.deepcopy(inputs[0]["prompt"][1])]

        if inputs[0]["source"] == "videoespresso_train_video":
            video_root = VIDEO_ESPRESSO_ROOT
            input_copy[0]["content"][0]["video"] = os.path.join(video_root, inputs[0]["video_path"])
        elif inputs[0]["source"] == "timerft":
            video_root = TIMERFT_ROOT
            input_copy[0]["content"][0]["video"] = os.path.join(video_root, inputs[0]["video_path"])
        elif inputs[0]["source"] == "gqa":
            image_root = GQA_ROOT
            input_copy[0]["content"][0]["image"] = os.path.join(image_root, inputs[0]["image_path"])
        elif "STR" in inputs[0]["source"]:
            if "STR_plm" in inputs[0]["source"]:
                video_root = STR_PLM_DATA
            else:
                video_root = STR_DATA
            input_copy[0]["content"][0]["video"] = os.path.join(video_root, inputs[0]["video_path"])
        elif "TVG" in inputs[0]["source"]:
            video_root = TVG_ROOT
            input_copy[0]["content"][0]["video"] = os.path.join(video_root, inputs[0]["video_path"])
        elif "videor1" in inputs[0]["source"]:
            video_root = GENERAL_VIDEO_ROOT
            input_copy[0]["content"][0]["video"] = os.path.join(video_root, inputs[0]["video_path"])
        else:
            raise ValueError(f"Invalid source: {inputs[0]['source']}")

        input_copy = self.remove_none_from_data(input_copy)

        # remove None from key_items
        if "key_items" in inputs[0]:
            keys_to_remove = []
            for key, item in inputs[0]["key_items"].items():
                if item is None:
                    keys_to_remove.append(key)
                elif isinstance(item, dict):
                    sub_keys_to_remove = [k for k, v in item.items() if v is None]
                    for k in sub_keys_to_remove:
                        del item[k]
            for key in keys_to_remove:
                del inputs[0]["key_items"][key]

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                input_copy, return_video_kwargs=True
            )
            if image_inputs is not None:
                inputs[0]["image_size_refine"] = (image_inputs[0].size[0], image_inputs[0].size[1])
                inputs[0]["prompt_text_final"] = prompts_text[0]
            if video_inputs is not None:
                inputs[0]["video_sample_fps"] = video_kwargs["fps"][0]
                inputs[0]["video_duration"] = video_inputs[0].size(0) / video_kwargs["fps"][0]
                inputs[0]["image_size"] = (video_inputs[0].size(3), video_inputs[0].size(2))
                inputs[0]["prompt_text_final"] = prompts_text[0]
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")

        current_step = self.state.global_step + 1
        total_steps = self.state.max_steps
        inputs[0]["step_percent"] = current_step / total_steps

        multi_image = True
        if video_inputs is None:
            multi_image = False

        if multi_image:
            if inputs[0]["task"] != "temporal-spatial free-form QA":
                frame_prompt = ""
                ori_idx = 0
                while ori_idx < len(video_inputs[0]):
                    time_now = round(ori_idx / video_kwargs["fps"][0], 1)
                    frame_prompt += f"Frame {ori_idx + 1} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                    ori_idx += 1
                frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                prompts_text[0] = prompts_text[0].replace(
                    "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
                )
                inputs[0]["prompt_text_final"] = prompts_text[0]
                image_inputs = [video_inputs[0]]
            else:
                width, height = video_inputs[0].size(3), video_inputs[0].size(2)
                image_size = (width, height)

                if inputs[0]["source"] == "videoespresso_train_video":
                    key_frame_root = VIDEO_ESPRESSO_KF_ROOT
                elif "STR_plm" in inputs[0]["source"]:
                    key_frame_root = STR_PLM_KF_ROOT
                else:
                    key_frame_root = STR_KF_ROOT

                # key frame load 
                key_frames = []
                for key_frame in inputs[0]["key_frames"]:
                    kf_path = os.path.join(key_frame_root, key_frame["path"])
                    kf = Image.open(kf_path)
                    kf = kf.convert("RGB")
                    resized_kf = kf.resize(image_size)
                    resized_kf = np.array(resized_kf)
                    resized_kf = np.transpose(resized_kf, (2, 0, 1))
                    resized_kf = torch.from_numpy(resized_kf)
                    key_frames.append((round(key_frame["time"]), resized_kf))

                frame_prompt = ""
                refined_image_inputs = []
                kf_idx = 0
                ori_idx = 0
                frame_idx = 1
                while ori_idx < len(video_inputs[0]):
                    time_now = int(ori_idx / video_kwargs["fps"][0])
                    if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                        refined_image_inputs.append(key_frames[kf_idx][1])
                        time_now = round(key_frames[kf_idx][0], 1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        kf_idx += 1
                    else:
                        refined_image_inputs.append(video_inputs[0][ori_idx])
                        time_now = round(ori_idx / video_kwargs["fps"][0], 1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        ori_idx += 1
                    frame_idx += 1
                frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                image_inputs = torch.stack(refined_image_inputs)
                image_inputs = [image_inputs]
                prompts_text[0] = prompts_text[0].replace(
                    "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
                )
                inputs[0]["prompt_text_final"] = prompts_text[0]

            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs,
            )

        prompt_inputs = super(Qwen2VLGRPOTrainer, self)._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token 
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        if multi_image or image_inputs is not None:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(
                len(prompt_completion_ids), 1
            )
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(
                len(prompt_completion_ids), 1
            )
        else:
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(
                len(prompt_completion_ids), 1
            )
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(
                len(prompt_completion_ids), 1
            )

        if "second_per_grid_ts" in prompt_inputs:
            del prompt_inputs["second_per_grid_ts"]

        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)    
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)

        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, **prompt_inputs
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, **prompt_inputs
                        )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]


        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)   
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        # Compute rewards
        prompts_repeated = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts_repeated), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {
                key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]
            }
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            if self.script_args is not None:
                reward_kwargs["spatial_iou_mode"] = getattr(self.script_args, "spatial_iou_mode", "max")
                reward_kwargs["identity_match_mode"] = getattr(self.script_args, "identity_match_mode", "none")
                reward_kwargs["spatial_norm_mode"] = getattr(self.script_args, "spatial_norm_mode", "all")
                reward_kwargs["correct_tempgate"] = getattr(self.script_args, "correct_tempgate", True)

            output_reward_func = reward_func(
                prompts=prompts_repeated, completions=completions, **reward_kwargs
            )
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = self._apply_reward_ablation_mask(rewards_per_func)
        rewards = rewards_per_func.sum(dim=1)

        # ---- Save local average reward for IDL threshold check ----
        local_avg_reward = rewards.view(-1, self.num_generations).mean(dim=1).mean().item()
        ans_reward_idx = 0
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = (
                reward_func.config._name_or_path.split("/")[-1]
                if isinstance(reward_func, PreTrainedModel)
                else reward_func.__name__
            )
            if "ans_acc" in reward_func_name:
                ans_reward_idx = i
                break
        local_avg_answer_reward = (
            rewards_per_func[:, ans_reward_idx]
            .view(-1, self.num_generations)
            .mean(dim=1)
            .mean()
            .item()
        )

        # Grouped rewards for GSPO
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # GSPO loss
        log_ratio = per_token_logps - per_token_logps.detach()
        if self.gspo:
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            log_importance_weights = log_ratio

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss + self.beta * per_token_kl
        gspo_loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        # Metrics (same as parent)
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        num_devices = gathered_rewards.size(0) // self.num_generations
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # ==============================================================
        # Phase 2: IDL (conditional, only for hard questions)
        # ==============================================================
        final_loss = gspo_loss    

        print(f"local_avg_reward: {local_avg_reward}")

        torch.cuda.empty_cache()
        if self.enable_idl:
            is_hard = local_avg_reward < self.idl_k1
            if is_hard and self.idl_stgr_filter_ids is not None:
                sample_id = inputs[0].get("id", "")
                if sample_id not in self.idl_stgr_filter_ids:
                    is_hard = False
            has_key_frames = bool(inputs[0].get("key_frames", []))
            has_key_items = bool(inputs[0].get("key_items", {}))

            try:
                idl_loss = self._compute_idl_loss(
                    model=model,
                    inputs=inputs,
                    prompts=prompts,
                    local_avg_reward=local_avg_reward,
                    local_avg_answer_reward=local_avg_answer_reward,
                    has_key_frames=has_key_frames,
                    has_key_items=has_key_items,
                    video_inputs=video_inputs,
                    video_kwargs=video_kwargs,
                    multi_image=multi_image,
                    image_inputs_gspo=image_inputs,
                    is_hard=is_hard,
                )

                hard_mask = torch.tensor(
                    1.0 if is_hard else 0.0, device=gspo_loss.device, dtype=gspo_loss.dtype
                )
                finite_mask = torch.isfinite(idl_loss).to(dtype=gspo_loss.dtype)
                idl_loss_safe = torch.nan_to_num(idl_loss, nan=0.0, posinf=0.0, neginf=0.0)
                final_loss = gspo_loss + self.idl_alpha * hard_mask * finite_mask * idl_loss_safe

                if is_hard and bool(torch.isfinite(idl_loss).item()):
                    self._metrics["idl_loss"].append(idl_loss.item())
                    self._metrics["idl_avg_reward_before"].append(local_avg_reward)
                    self._metrics["idl_triggered"].append(1.0)
                else:
                    self._metrics["idl_triggered"].append(0.0)

            except Exception as e:
                print(f"[IDL] Error: {e}")
                import traceback
                traceback.print_exc()
                self._metrics["idl_triggered"].append(0.0)

        return final_loss

    # -----------------------------------------------------------------
    # IDL Loss Computation
    # -----------------------------------------------------------------
    def _compute_idl_loss(
        self,
        model,
        inputs,
        prompts,
        local_avg_reward,
        local_avg_answer_reward,
        has_key_frames,
        has_key_items,
        video_inputs,
        video_kwargs,
        multi_image,
        image_inputs_gspo,
        is_hard=True,
    ):
        """Compute IDL (self-distillation) loss for a hard question.

        Steps:
          1. Look up VP selector prediction (cached)
          2. Prepare VP-guided visual input (darken key frames if VP=1)
          3. Add VP guidance hint to prompt
          4. Generate new completions
          5. Compute rewards for new completions
          6. Select positive responses (up to k2 * n_rollout)
          7. Compute NLL on positive responses

        Returns:
            IDL loss tensor, or None if no positive responses found.
        """
        device = self.accelerator.device
        sample_id = inputs[0].get("id", "")
        vp_pred = self.vp_predictions.get(sample_id, 0)

        if self.wo_vp_selector:
            vp_pred = "raw"

        if isinstance(vp_pred, int):
            vp_pred = "darken" if vp_pred == 1 else "raw"

        if vp_pred != "raw" and not has_key_items:
            vp_pred = "raw"

        hint = VP_HINT_MAP.get(vp_pred, VP_HINT_RAW)

        # ---- Step 1: Reconstruct prompt with VP hint ---- 
        idl_prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
        if assistant_marker in idl_prompts_text[0]:
            marker_pos = idl_prompts_text[0].rfind(assistant_marker)
            idl_prompts_text[0] = (
                idl_prompts_text[0][:marker_pos] + hint + idl_prompts_text[0][marker_pos:]
            )
        else:
            idl_prompts_text[0] = idl_prompts_text[0] + hint

        # ---- Step 2: Process visual input (with optional darken VP) ----
        input_copy = [copy.deepcopy(inputs[0]["prompt"][1])]


        # Resolve video/image path (same as GSPO)
        source_valid = True
        if inputs[0]["source"] == "videoespresso_train_video":
            input_copy[0]["content"][0]["video"] = os.path.join(VIDEO_ESPRESSO_ROOT, inputs[0]["video_path"])
        elif inputs[0]["source"] == "timerft":
            input_copy[0]["content"][0]["video"] = os.path.join(TIMERFT_ROOT, inputs[0]["video_path"])
        elif inputs[0]["source"] == "gqa":
            input_copy[0]["content"][0]["image"] = os.path.join(GQA_ROOT, inputs[0]["image_path"])
        elif "STR" in inputs[0]["source"]:
            vr = STR_PLM_DATA if "STR_plm" in inputs[0]["source"] else STR_DATA
            input_copy[0]["content"][0]["video"] = os.path.join(vr, inputs[0]["video_path"])
        elif "TVG" in inputs[0]["source"]:
            input_copy[0]["content"][0]["video"] = os.path.join(TVG_ROOT, inputs[0]["video_path"])
        elif "videor1" in inputs[0]["source"]:
            input_copy[0]["content"][0]["video"] = os.path.join(GENERAL_VIDEO_ROOT, inputs[0]["video_path"])
        else:
            print(f"[IDL] Unknown source: {inputs[0]['source']}, falling back to text-only IDL path")
            source_valid = False

        if source_valid:
            input_copy = self.remove_none_from_data(input_copy)

        idl_image_inputs, idl_video_inputs, idl_video_kwargs = None, None, {}
        try:
            if source_valid:
                # input을 읽는 부분
                idl_image_inputs, idl_video_inputs, idl_video_kwargs = process_vision_info(
                    input_copy, return_video_kwargs=True
                )
        except Exception as e:
            print(f"[IDL] process_vision_info error: {e}, falling back to text-only IDL path")
            idl_image_inputs, idl_video_inputs, idl_video_kwargs = None, None, {}

        idl_multi_image = idl_video_inputs is not None

        if idl_multi_image:
            if inputs[0]["task"] != "temporal-spatial free-form QA":
                frame_prompt = ""
                ori_idx = 0
                while ori_idx < len(idl_video_inputs[0]):
                    time_now = round(ori_idx / idl_video_kwargs["fps"][0], 1)
                    frame_prompt += f"Frame {ori_idx + 1} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                    ori_idx += 1
                frame_prompt += f"The video is in total {int(idl_video_inputs[0].size(0) / idl_video_kwargs['fps'][0])} seconds.\n"
                idl_prompts_text[0] = idl_prompts_text[0].replace(
                    "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
                )
                idl_image_inputs = [idl_video_inputs[0]]

            else:
                # temporal-spatial free-form QA with key frames
                width, height = idl_video_inputs[0].size(3), idl_video_inputs[0].size(2)
                image_size = (width, height)

                if inputs[0]["source"] == "videoespresso_train_video":
                    key_frame_root = VIDEO_ESPRESSO_KF_ROOT
                elif "STR_plm" in inputs[0]["source"]:
                    key_frame_root = STR_PLM_KF_ROOT
                else:
                    key_frame_root = STR_KF_ROOT

                key_frames = []
                question_id = str(inputs[0].get("id", "")).strip()
                folder_id = re.sub(r'^STR_', '', question_id)
                vp_question_dir = None
                use_vp_keyframes = False

                if vp_pred != "raw" and folder_id:
                    vp_kf_dir = None
                    if self.vp_keyframe_base_dir:
                        if vp_pred == "numpro":
                            vp_kf_dir = os.path.join(self.vp_keyframe_base_dir, "numpro", "keyframes", folder_id)
                        else:
                            vp_kf_dir = os.path.join(self.vp_keyframe_base_dir, vp_pred, folder_id)
                    elif vp_pred == "darken" and self.darken_keyframe_dir:
                        vp_kf_dir = os.path.join(self.darken_keyframe_dir, folder_id)

                    if vp_kf_dir and os.path.isdir(vp_kf_dir):
                        vp_question_dir = vp_kf_dir
                        use_vp_keyframes = True
                    else:
                        if not hasattr(self, "_idl_vp_folder_miss_logged"):
                            self._idl_vp_folder_miss_logged = set()
                        key = (folder_id, vp_pred)
                        if key not in self._idl_vp_folder_miss_logged and len(self._idl_vp_folder_miss_logged) < 20:
                            self._idl_vp_folder_miss_logged.add(key)
                            print(
                                f"[IDL] VP keyframe folder not found for '{folder_id}' "
                                f"(vp_pred={vp_pred}), falling back to raw. (further misses logged only up to 20 unique.)"
                            )

                for kf_idx, key_frame in enumerate(inputs[0]["key_frames"]):

                    if use_vp_keyframes and vp_question_dir is not None:
                        orig_name = os.path.basename(key_frame["path"])
                        match = re.match(r".*?(\d+)_time_([\d.]+)\.jpg", orig_name)
                        vp_path = None
                        time_fmt = ""
                        time_str = ""
                        if match:
                            idx_str, time_str = match.groups()
                            time_fmt = f"{float(time_str):.2f}"
                            candidates = [
                                os.path.join(vp_question_dir, f"frame_{idx_str}_time_{time_fmt}.jpg"),
                                os.path.join(vp_question_dir, f"frame_{idx_str}_time_{time_str}.jpg"),
                                os.path.join(vp_question_dir, f"frame_{kf_idx}_time_{time_fmt}.jpg"),
                                os.path.join(vp_question_dir, f"frame_{kf_idx}_time_{time_str}.jpg"),
                                os.path.join(vp_question_dir, f"{folder_id}_{idx_str}_time_{time_fmt}.jpg"),
                                os.path.join(vp_question_dir, f"{folder_id}_{idx_str}_time_{time_str}.jpg"),
                                os.path.join(vp_question_dir, orig_name),
                            ]
                            if idx_str and int(idx_str) > 0:
                                candidates.extend([
                                    os.path.join(vp_question_dir, f"frame_{int(idx_str) - 1}_time_{time_fmt}.jpg"),
                                    os.path.join(vp_question_dir, f"frame_{int(idx_str) - 1}_time_{time_str}.jpg"),
                                ])
                            for cand in candidates:
                                if os.path.exists(cand):
                                    vp_path = cand
                                    break
                        if vp_path is None:
                            cand = os.path.join(vp_question_dir, orig_name)
                            if os.path.exists(cand):
                                vp_path = cand
                        if vp_path is None and time_fmt:
                            try:
                                files = os.listdir(vp_question_dir)
                                same_time = [f for f in files if f.endswith(".jpg") and (time_fmt in f or time_str in f)]
                                same_time.sort()
                                if same_time and kf_idx < len(same_time):
                                    vp_path = os.path.join(vp_question_dir, same_time[kf_idx])
                                elif same_time:
                                    vp_path = os.path.join(vp_question_dir, same_time[0])
                            except OSError:
                                pass

                        if vp_path is not None:
                            kf = Image.open(vp_path).convert("RGB")
                        else:
                            if not hasattr(self, "_idl_vp_file_miss_logged"):
                                self._idl_vp_file_miss_logged = set()
                            key = (vp_question_dir, orig_name)
                            if key not in self._idl_vp_file_miss_logged and len(self._idl_vp_file_miss_logged) < 30:
                                self._idl_vp_file_miss_logged.add(key)
                                print(f"[IDL] VP keyframe file not found for {orig_name} in {vp_question_dir}, using raw")
                            kf_path = os.path.join(key_frame_root, key_frame["path"])
                            kf = Image.open(kf_path).convert("RGB")
                    else:
                        kf_path = os.path.join(key_frame_root, key_frame["path"])
                        kf = Image.open(kf_path).convert("RGB")
                    
                    resized_kf = kf.resize(image_size)
                    resized_kf = np.array(resized_kf)
                    resized_kf = np.transpose(resized_kf, (2, 0, 1))
                    kf_tensor = torch.from_numpy(resized_kf)
                    key_frames.append((round(key_frame["time"]), kf_tensor))

                frame_prompt = ""
                refined_image_inputs = []
                kf_idx = 0
                ori_idx = 0
                frame_idx = 1

                
                while ori_idx < len(idl_video_inputs[0]):
                    time_now = int(ori_idx / idl_video_kwargs["fps"][0])
                    if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                        refined_image_inputs.append(key_frames[kf_idx][1])
                        time_now = round(key_frames[kf_idx][0], 1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        kf_idx += 1
                    else:
                        refined_image_inputs.append(idl_video_inputs[0][ori_idx])
                        time_now = round(ori_idx / idl_video_kwargs["fps"][0], 1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        ori_idx += 1
                    frame_idx += 1

                frame_prompt += f"The video is in total {int(idl_video_inputs[0].size(0) / idl_video_kwargs['fps'][0])} seconds.\n"
                idl_image_inputs = [torch.stack(refined_image_inputs)]
                idl_prompts_text[0] = idl_prompts_text[0].replace(
                    "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
                )

            idl_prompt_inputs = self.processing_class(
                text=copy.deepcopy(idl_prompts_text),
                images=idl_image_inputs,
                videos=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            idl_prompt_inputs = self.processing_class(
                text=copy.deepcopy(idl_prompts_text),
                images=idl_image_inputs,
                videos=idl_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **idl_video_kwargs,
            )

        idl_prompt_inputs = super(Qwen2VLGRPOTrainer, self)._prepare_inputs(idl_prompt_inputs)

        # Truncate prompt
        if self.max_prompt_length is not None:
            idl_prompt_inputs["input_ids"] = idl_prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            idl_prompt_inputs["attention_mask"] = idl_prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        idl_prompt_length = idl_prompt_inputs["input_ids"].size(1)

        # ---- Step 3: Generate new completions with VP-guided input ----
        idl_gen_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=self.processing_class.pad_token_id,
        )

        # Save prompt_inputs before generation modifies them
        idl_prompt_inputs_for_fwd = {
            k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
            for k, v in idl_prompt_inputs.items()
        }

        torch.cuda.empty_cache()

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                idl_prompt_completion_ids = unwrapped_model.generate(
                    **idl_prompt_inputs, generation_config=idl_gen_config
                )

        idl_completion_ids = idl_prompt_completion_ids[:, idl_prompt_length:]

        # EOS masking
        is_eos = idl_completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        idl_completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # ---- Step 4: Compute rewards for IDL completions ----
        idl_completions = self.processing_class.batch_decode(
            idl_completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            idl_completions = [[{"role": "assistant", "content": c}] for c in idl_completions]

        idl_prompts_repeated = [p for p in prompts for _ in range(self.num_generations)]
        idl_rewards_per_func = torch.zeros(
            len(idl_prompts_repeated), len(self.reward_funcs), device=device
        )

        for i, (reward_func, _) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {
                key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]
            }
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            if self.script_args is not None:
                reward_kwargs["spatial_iou_mode"] = getattr(self.script_args, "spatial_iou_mode", "max")
                reward_kwargs["identity_match_mode"] = getattr(self.script_args, "identity_match_mode", "none")
                reward_kwargs["spatial_norm_mode"] = getattr(self.script_args, "spatial_norm_mode", "all")
                reward_kwargs["correct_tempgate"] = getattr(self.script_args, "correct_tempgate", True)
            output_reward = reward_func(
                prompts=idl_prompts_repeated, completions=idl_completions, **reward_kwargs
            )
            idl_rewards_per_func[:, i] = torch.tensor(output_reward, dtype=torch.float32, device=device)

        idl_rewards_per_func = self._apply_reward_ablation_mask(idl_rewards_per_func)
        idl_rewards = idl_rewards_per_func.sum(dim=1)     # [# generation]
        self._metrics["idl_rewards_mean"].append(idl_rewards.mean().item())

        # ---- Build reward function index mapping ----
        reward_idx_map = {}
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = (
                reward_func.config._name_or_path.split("/")[-1]
                if isinstance(reward_func, PreTrainedModel)
                else reward_func.__name__
            )
            if "ans_acc" in reward_func_name:
                reward_idx_map["ans_acc"] = i
            elif "thk_temporal_point" in reward_func_name:
                reward_idx_map["thk_temporal_point"] = i
            elif "thk_spatial" in reward_func_name:
                reward_idx_map["thk_spatial"] = i

        ans_reward_idx = reward_idx_map.get("ans_acc", 0)
        
        # Compute grounding reward (temporal + spatial) for STGR tasks
        grounding_reward = torch.zeros(len(idl_rewards), device=device)
        if "thk_temporal_point" in reward_idx_map:
            grounding_reward += idl_rewards_per_func[:, reward_idx_map["thk_temporal_point"]]
        if "thk_spatial" in reward_idx_map:
            grounding_reward += idl_rewards_per_func[:, reward_idx_map["thk_spatial"]]
        
        idl_answer_rewards = idl_rewards_per_func[:, ans_reward_idx]
        
        # Compute baselines
        baseline_ans = local_avg_answer_reward
        baseline_grounding = grounding_reward.mean().item()
        
        # ---- Step 5: Select positive responses for IDL ----
        # Candidate selection based on idl_candidate_mode
        #   - "ans_only":        ans_acc > baseline (original)
        #   - "grounding_only":  grounding > baseline (for STGR task)
        #   - "ans_or_grounding": ans_acc > baseline OR grounding > baseline
        #   - "total_reward":    total reward > baseline total
        
        ans_mask = idl_answer_rewards > baseline_ans
        grounding_mask = grounding_reward > baseline_grounding
        total_mask = idl_rewards > idl_rewards.mean()
        
        if self.idl_candidate_mode == "ans_only":
            candidate_mask = ans_mask
        elif self.idl_candidate_mode == "grounding_only":
            candidate_mask = grounding_mask
        elif self.idl_candidate_mode == "ans_or_grounding":
            candidate_mask = ans_mask | grounding_mask
        elif self.idl_candidate_mode == "total_reward":
            candidate_mask = total_mask
        else:
            candidate_mask = ans_mask
            
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        
        # Log candidate selection stats
        self._metrics["idl_candidate_ans_pass"].append(float(ans_mask.sum().item()))
        self._metrics["idl_candidate_grounding_pass"].append(float(grounding_mask.sum().item()))
        self._metrics["idl_candidate_final"].append(float(len(candidate_indices)))

        # Determine ranking score based on idl_positive_ranking
        #   - "ans_acc":     rank by ans_acc reward
        #   - "grounding":   rank by grounding reward (temporal + spatial)
        #   - "total_reward": rank by total reward
        if self.idl_positive_ranking == "grounding":
            ranking_scores = grounding_reward
        elif self.idl_positive_ranking == "total_reward":
            ranking_scores = idl_rewards
        else:  # "ans_acc" (default)
            ranking_scores = idl_answer_rewards

        ci_nonempty = len(candidate_indices) > 0
        if not ci_nonempty:
            if is_hard:
                print(
                    f"[IDL] No responses passed candidate_mask (mode={self.idl_candidate_mode}) "
                    f"for sample {sample_id}. Baselines: ans={baseline_ans:.4f}, grounding={baseline_grounding:.4f}. "
                    f"Forcing dummy IDL path and masking IDL loss to zero (L = L_GSPO)."
                )
            top_k = min(self.idl_top_k, ranking_scores.numel())
            _, positive_indices = torch.topk(ranking_scores, k=top_k, largest=True, sorted=True)
        else:
            candidate_ranking_scores = ranking_scores[candidate_indices]
            top_k = min(self.idl_top_k, len(candidate_indices))
            _, top_idx = torch.topk(candidate_ranking_scores, k=top_k, largest=True, sorted=True)
            positive_indices = candidate_indices[top_idx]

        if len(positive_indices) == 0:
            if is_hard:
                print(f"[IDL] No positive responses for sample {sample_id}, using index 0 fallback")
            positive_indices = torch.zeros(1, dtype=torch.long, device=device)

        self._metrics["idl_num_positives"].append(float(len(positive_indices)))

        # ---- Detailed IDL log (only for hard STGR samples) ----
        if is_hard:
            n_gen = idl_rewards.numel()
            print(
                f"[IDL-Detail] sample={sample_id} | source={inputs[0].get('source','')} | "
                f"vp_pred={vp_pred} | gspo_avg_R={local_avg_reward:.4f} | "
                f"idl_avg_R={idl_rewards.mean().item():.4f} | "
                f"ans_baseline={baseline_ans:.4f} grounding_baseline={baseline_grounding:.4f}"
            )
            per_gen_info = []
            for gi in range(n_gen):
                per_gen_info.append(
                    f"  gen{gi}: ans={idl_answer_rewards[gi].item():.4f} "
                    f"grounding={grounding_reward[gi].item():.4f} "
                    f"total={idl_rewards[gi].item():.4f} "
                    f"ans_pass={'Y' if ans_mask[gi] else 'N'} "
                    f"grd_pass={'Y' if grounding_mask[gi] else 'N'} "
                    f"candidate={'Y' if candidate_mask[gi] else 'N'}"
                )
            print("\n".join(per_gen_info))
            print(
                f"[IDL-Detail] candidates={len(candidate_indices)}/{n_gen} | "
                f"positives={len(positive_indices)} (idx={positive_indices.tolist()}) | "
                f"ranking_by={self.idl_positive_ranking}"
            )

        # ---- Step 6: Compute NLL (self-distillation loss) on positive responses ----
        # L'_IDL = -(1/|Y'_Sel|) * Σ log π_θ(y'_j | x_i, p)
        positive_completion_ids = idl_prompt_completion_ids[positive_indices]
        positive_completion_mask = idl_completion_mask[positive_indices]

        # Prepare visual features for forward pass (expand for positive count)
        idl_fwd_inputs = {}
        n_pos = len(positive_indices)

        # Remove input_ids / attention_mask (we use the full prompt+completion ids)
        for k, v in idl_prompt_inputs_for_fwd.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if isinstance(v, torch.Tensor):
                idl_fwd_inputs[k] = v.repeat(n_pos, *([1] * (v.dim() - 1)))
            else:
                idl_fwd_inputs[k] = v

        # Remove second_per_grid_ts if present
        idl_fwd_inputs.pop("second_per_grid_ts", None)

        # Forward pass to get per-token log probs for positive responses
        try:
            idl_per_token_logps = self._get_per_token_logps(   
                model, positive_completion_ids, **idl_fwd_inputs
            )
            idl_per_token_logps = idl_per_token_logps[:, idl_prompt_length - 1 :]

        except Exception as e:
            print(f"[IDL] Error computing logps: {e}")
            try:
                idl_per_token_logps = self._get_per_token_logps(model, positive_completion_ids)
                idl_per_token_logps = idl_per_token_logps[:, idl_prompt_length - 1 :]
            except Exception as e2:
                print(f"[IDL] Fallback logps failed: {e2}, using zero logps")
                idl_per_token_logps = torch.zeros(
                    positive_completion_mask.size(0),
                    positive_completion_mask.size(1),
                    dtype=torch.float32,
                    device=device,
                )

        # Align mask length with logps
        min_len = min(idl_per_token_logps.size(1), positive_completion_mask.size(1))
        idl_per_token_logps = idl_per_token_logps[:, :min_len]
        positive_completion_mask = positive_completion_mask[:, :min_len]   
        nll_per_sample = -(idl_per_token_logps * positive_completion_mask).sum(dim=1) / positive_completion_mask.sum(dim=1).clamp(min=1.0)
        idl_loss = nll_per_sample.mean()
        if not ci_nonempty:
            idl_loss = idl_loss * torch.zeros((), dtype=idl_loss.dtype, device=idl_loss.device)

        return idl_loss
