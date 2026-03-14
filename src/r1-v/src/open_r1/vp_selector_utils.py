"""
VP Selector Utilities for IDL (Iterative Distillation Learning) Integration

Provides:
1. VP Selector model loading and batch inference
2. Darken VP application to image tensors (key frames)
3. Key object text extraction from key_items
4. Pre-computation and caching of VP predictions
5. VP guidance prompt constants
"""

import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm


# ==============================
# VP Guidance Prompt Hints
# ==============================

VP_HINT_DARKEN = (
    "\n[Hint: The key frames have been visually enhanced — regions outside the key objects "
    "are darkened to draw attention to the most relevant areas. Focus on the bright (highlighted) "
    "regions and track these key objects across different time points to answer the question.]"
)

VP_HINT_RAW = (
    "\n[Hint: Carefully observe the key objects in the video and track their spatial positions "
    "and temporal changes across different frames to answer the question accurately.]"
)

VP_HINT_NUMPRO = (
    "\n[Hint: The key frames have been annotated with frame numbers to help you identify "
    "the temporal order and timing of events. Pay close attention to these numbered frames "
    "to accurately determine when key events occur in the video.]"
)

VP_HINT_DARKEN_AND_NUMPRO = (
    "\n[Hint: The key frames have been visually enhanced in two ways — regions outside the key objects "
    "are darkened, and frames are annotated with numbers to help temporal identification. "
    "Focus on the bright (highlighted) regions for spatial grounding, and use frame numbers "
    "for accurate temporal grounding.]"
)


# ==============================
# Helper Functions
# ==============================

def build_vp_selector_prompt(question: str, key_object: Optional[str]) -> str:
    """Build prompt for VP selector binary classification.
    
    Same format as used in VP selector training
    (train_vp_selector_qwen_7b_lora_binary.py).
    """
    if key_object and key_object.strip():
        return (
            "You are determining whether 'raw' (no visual prompting) is the best method "
            "for video question answering.\n"
            f"Question: {question}\n"
            f"Key object: {key_object}\n"
            "Answer: yes (raw is best) or no (other VP method is better)."
        )
    return (
        "You are determining whether 'raw' (no visual prompting) is the best method "
        "for video question answering.\n"
        f"Question: {question}\n"
        "Answer: yes (raw is best) or no (other VP method is better)."
    )


def extract_key_object_text(key_items: dict) -> Optional[str]:
    """Extract key object names from key_items dict.

    key_items format examples:
      Case A (temporal-spatial): {"1": {"monitors": [[bbox]], "data": [[bbox]]}, "2": {...}}
      Case B (visual QA):       {"girl": [[bbox]]}

    Returns:
        Comma-separated unique object names, or None if empty.
    """
    if not key_items:
        return None

    object_names = set()
    for frame_key, objects in key_items.items():
        if isinstance(objects, dict):
            # Case A: frame_key -> {obj_name: [bboxes]}
            for obj_name in objects.keys():
                if obj_name and isinstance(obj_name, str):
                    object_names.add(obj_name)
        elif isinstance(objects, list):
            # Case B: obj_name -> [[bbox], ...]
            if frame_key and isinstance(frame_key, str):
                object_names.add(frame_key)

    if not object_names:
        return None
    return ", ".join(sorted(object_names))


# ==============================
# Key Object JSON Loader
# ==============================

_key_object_cache: Optional[Dict[str, Dict]] = None
_key_object_json_path: Optional[str] = None


def load_key_object_json(json_path: str) -> Dict[str, Dict]:
    """Load key object extraction JSON and cache it.
    
    Expected format:
    [
      {
        "id": "000000",
        "key_object_extraction": {
          "primary": "man",
          "secondary": ["beer glass"],
          ...
        },
        ...
      },
      ...
    ]
    
    Returns:
        Dict mapping sample_id -> sample dict
    """
    global _key_object_cache, _key_object_json_path
    
    if _key_object_cache is not None and _key_object_json_path == json_path:
        return _key_object_cache
    
    if not os.path.exists(json_path):
        print(f"[VPSelector] Warning: Key object JSON not found: {json_path}")
        _key_object_cache = {}
        _key_object_json_path = json_path
        return _key_object_cache
    
    print(f"[VPSelector] Loading key object JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Build id -> sample mapping
    _key_object_cache = {sample.get("id"): sample for sample in data if "id" in sample}
    _key_object_json_path = json_path
    print(f"[VPSelector] Loaded {len(_key_object_cache)} key object entries")
    
    return _key_object_cache


def get_key_object_from_json(sample_id: str, json_path: str) -> Optional[str]:
    """Get primary key object text from JSON for a given sample_id.
    
    Args:
        sample_id: Sample ID (e.g., "000000")
        json_path: Path to key object extraction JSON
        
    Returns:
        Primary key object name (e.g., "man"), or None if not found
    """
    cache = load_key_object_json(json_path)
    sample = cache.get(sample_id)
    if not sample:
        return None
    
    key_obj_ext = sample.get("key_object_extraction", {})
    primary = key_obj_ext.get("primary")
    
    if primary and isinstance(primary, str):
        return primary.strip()
    
    return None


def extract_all_bboxes(key_items: dict) -> List[List[float]]:
    """Extract ALL bounding boxes from key_items (flattened across all frames/objects).

    Returns:
        List of normalized bboxes [[x1, y1, x2, y2], ...]
    """
    bboxes = []
    if not key_items:
        return bboxes

    for frame_key, objects in key_items.items():
        if isinstance(objects, dict):
            for obj_name, bbox_list in objects.items():
                if isinstance(bbox_list, list):
                    for bbox in bbox_list:
                        if isinstance(bbox, list) and len(bbox) == 4:
                            bboxes.append(bbox)
        elif isinstance(objects, list):
            for bbox in objects:
                if isinstance(bbox, list) and len(bbox) == 4:
                    bboxes.append(bbox)
    return bboxes


def apply_darken_to_tensor(
    image_tensor: torch.Tensor,
    bboxes: List[List[float]],
    darken_factor: float = 0.3,
    padding: int = 10,
) -> torch.Tensor:
    """Apply darken VP to an image tensor.

    Darkens regions OUTSIDE bounding boxes, keeping bbox regions bright.

    Args:
        image_tensor: (C, H, W) tensor (uint8 or float)
        bboxes: list of normalized bboxes [[x1, y1, x2, y2], ...]
        darken_factor: 0~1, lower = darker background
        padding: pixel padding around bboxes

    Returns:
        Darkened image tensor (same shape and dtype as input)
    """
    if not bboxes:
        return image_tensor

    C, H, W = image_tensor.shape
    device = image_tensor.device
    dtype = image_tensor.dtype

    img_float = image_tensor.float()

    # Create mask: 1.0 for bbox regions (bright), 0.0 for background (darken)
    mask = torch.zeros(H, W, device=device, dtype=torch.float32)
    for bbox in bboxes:
        x1 = max(0, int(bbox[0] * W) - padding)
        y1 = max(0, int(bbox[1] * H) - padding)
        x2 = min(W, int(bbox[2] * W) + padding)
        y2 = min(H, int(bbox[3] * H) + padding)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

    mask_3d = mask.unsqueeze(0).expand(C, -1, -1)  # (C, H, W)
    result = img_float * mask_3d + img_float * darken_factor * (1.0 - mask_3d)

    return result.to(dtype)


# ==============================
# VP Selector Model
# ==============================

class VPSelectorPredictor:
    """VP Selector for batch prediction.

    Loads the trained VP selector model (Qwen2.5-7B-Instruct + LoRA + cls head)
    and provides inference. Used during training initialization to pre-compute
    VP predictions for all training samples.
    """

    def __init__(self, model_dir: str, device: str = "cpu", batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size

        # Load training meta
        meta_path = os.path.join(model_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            model_name = meta.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        else:
            model_name = "Qwen/Qwen2.5-7B-Instruct"

        print(f"[VPSelector] Loading base model: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Note:
        # In ZeRO-3 distributed training runs, low_cpu_mem_usage=True is incompatible
        # with from_pretrained model loading and can crash rank0 before barrier sync.
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        print(f"[VPSelector] Loading LoRA adapter from: {model_dir}")
        backbone = PeftModel.from_pretrained(backbone, model_dir)

        hidden = backbone.config.hidden_size
        head = nn.Linear(hidden, 2)
        cls_head_path = os.path.join(model_dir, "cls_head.pt")
        if os.path.exists(cls_head_path):
            head.load_state_dict(
                torch.load(cls_head_path, map_location="cpu", weights_only=True)
            )
            print(f"[VPSelector] Classification head loaded from: {cls_head_path}")

        # Wrap into a simple module
        class _VPModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, input_ids, attention_mask):
                out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                h = out.hidden_states[-1]  # (B, L, H)
                idx = attention_mask.sum(dim=-1) - 1  # (B,)
                pooled = h[torch.arange(h.size(0), device=h.device), idx]  # (B, H)
                return self.head(pooled.float())  # (B, 2)

        self.model = _VPModel(backbone, head).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[VPSelector] Model loaded on {device}")

    @torch.no_grad()
    def predict_batch(
        self, questions: List[str], key_objects: List[Optional[str]]
    ) -> List[int]:
        """Batch prediction: 0 = raw, 1 = VP (darken)."""
        prompts = [
            build_vp_selector_prompt(q, ko)
            for q, ko in zip(questions, key_objects)
        ]
        predictions = []
        total = len(prompts)
        for i in tqdm(
            range(0, total, self.batch_size),
            desc="[VPSelector] Predicting",
            total=(total + self.batch_size - 1) // self.batch_size,
        ):
            batch = prompts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
            predictions.extend(logits.argmax(-1).tolist())
        return predictions

    def unload(self):
        """Free model memory."""
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        print("[VPSelector] Model unloaded")


# ==============================
# Pre-computation
# ==============================

def precompute_vp_predictions(
    vp_selector_model_dir: str,
    dataset,
    cache_path: str,
    device: str = "cpu",
    batch_size: int = 16,
    key_object_json_path: Optional[str] = None,
) -> Dict[str, int]:
    """Pre-compute VP predictions for all samples in the dataset.

    Caches results to ``cache_path`` (JSON).  If the cache file already exists,
    loads it directly without running the model.

    Args:
        vp_selector_model_dir: Path to trained VP selector model directory
        dataset: HuggingFace Dataset object (train split)
        cache_path: Path to save/load cached predictions JSON
        device: Device for VP selector inference ('cpu' or 'cuda:X')
        batch_size: Batch size for VP selector inference
        key_object_json_path: Optional path to key object extraction JSON.
            If provided, key objects are loaded from this JSON instead of key_items.

    Returns:
        Dict mapping sample_id -> prediction (0=raw, 1=VP/darken)
    """
    # Check cache
    if os.path.exists(cache_path):
        print(f"[VPSelector] Loading cached predictions from {cache_path}")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        print(f"[VPSelector] Loaded {len(cached)} cached predictions")
        return cached

    print(f"[VPSelector] Pre-computing VP predictions for {len(dataset)} samples...")

    # Load key object JSON if provided
    if key_object_json_path:
        load_key_object_json(key_object_json_path)

    # Load model
    predictor = VPSelectorPredictor(
        vp_selector_model_dir, device=device, batch_size=batch_size
    )

    # Extract questions and key objects
    sample_ids, questions, key_objects = [], [], []
    for sample in tqdm(dataset, desc="[VPSelector] Extracting features"):
        sample_id = sample.get("id", "")
        sample_ids.append(sample_id)
        questions.append(sample.get("question", ""))
        
        # Get key object: prefer JSON, fallback to key_items
        key_object = None
        if key_object_json_path:
            key_object = get_key_object_from_json(sample_id, key_object_json_path)
        
        if not key_object:
            key_items = sample.get("key_items", {})
            key_object = extract_key_object_text(key_items) if key_items else None
        
        key_objects.append(key_object)

    # Batch prediction
    predictions = predictor.predict_batch(questions, key_objects)

    # Build result dict
    result = {sid: pred for sid, pred in zip(sample_ids, predictions)}

    # Save cache
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f)

    raw_count = sum(1 for v in result.values() if v == 0)
    vp_count = sum(1 for v in result.values() if v == 1)
    print(f"[VPSelector] Predictions cached to {cache_path}")
    print(f"[VPSelector] Distribution: raw={raw_count}, VP(darken)={vp_count}")

    # Free memory
    predictor.unload()
    return result
