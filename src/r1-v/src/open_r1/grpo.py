import os
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
GQA_ROOT = os.path.join(ROOT, "gqa")
TIMERFT_ROOT = os.path.join(ROOT, "timerft")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
VIDEO_ESPRESSO_ROOT = os.path.join(ROOT, "videoespresso/videos")
VIDEO_ESPRESSO_KF_ROOT = os.path.join(ROOT, "videoespresso/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

from dataclasses import dataclass, field
from typing import Optional
from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from data_loader import get_data
# from reward_func import accuracy_reward, format_reward, temporal_reward, spatial_reward
from reward_func import ans_acc_reward, ans_tiou_reward, ans_viou_reward, thk_temporal_point_reward, thk_temporal_segment_reward, thk_spatial_reward, format_reward


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["ans_acc", "ans_tiou", "ans_viou", "thk_temporal_point", "thk_temporal_segment", "thk_spatial", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    spatial_iou_mode: Optional[str] = field(
        default="max",
        metadata={"help": "IoU aggregation mode for spatial reward across multiple GT objects: 'max' (default) or 'avg'"},
    )
    identity_match_mode: Optional[str] = field(
        default="none",
        metadata={"help": "Object identity matching mode: 'none' (ignore identity), 'soft' (flexible matching), 'strict' (exact match)"},
    )
    spatial_norm_mode: Optional[str] = field(
        default="all",
        metadata={"help": "Spatial reward normalization: 'all' (divide by all claims), 'matched' (divide by matched claims only)"},
    )
    correct_tempgate: Optional[bool] = field(
        default=True,
        metadata={"help": "Temporal gating condition: True (abs diff <= threshold), False (gt - pred < threshold)"},
    )
    # Ablation: exclude grounding rewards
    wo_spatial: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, exclude thk_spatial reward (spatial grounding ablation)"},
    )
    wo_tempspatial: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, exclude thk_temporal_point + thk_temporal_segment + thk_spatial (full grounding ablation)"},
    )
    wo_acc: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, exclude ans_acc reward (answer accuracy ablation)"},
    )
    # ============================================================
    # IDL (Iterative Distillation Learning) arguments
    # ============================================================
    enable_idl: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable IDL (Iterative Distillation Learning) with VP guidance"},
    )
    idl_k1: Optional[float] = field(
        default=0.5,
        metadata={"help": "Reward threshold k1: IDL activated when avg reward < k1 (hard question)"},
    )
    idl_k2: Optional[float] = field(
        default=0.5,
        metadata={"help": "Max positive ratio k2: retain at most k2 * n_rollout positive responses"},
    )
    idl_alpha: Optional[float] = field(
        default=0.1,
        metadata={"help": "Alpha: balancing weight for IDL loss (L_A2D = -J + alpha * L_IDL * I[k1 > R_bar])"},
    )
    vp_selector_model_dir: Optional[str] = field(
        default="",
        metadata={"help": "Path to trained VP selector model directory (Qwen2.5-7B + LoRA + cls_head)"},
    )
    vp_predictions_cache: Optional[str] = field(
        default="",
        metadata={"help": "Path to cached VP predictions JSON. Auto-generated if not provided."},
    )
    darken_keyframe_dir: Optional[str] = field(
        default="",
        metadata={"help": "Path to directory containing pre-computed darken key frames"},
    )
    numpro_keyframe_dir: Optional[str] = field(
        default="",
        metadata={"help": "Path to directory containing pre-computed numpro (numbered) key frames"},
    )
    vp_keyframe_base_dir: Optional[str] = field(
        default="",
        metadata={"help": "Base directory for all VP keyframes (contains api_prompt/, circle/, darken/, numpro/, raw/ subdirs)"},
    )
    key_object_json_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to key object extraction JSON (for VP selector input)"},
    )
    idl_stgr_filter_json: Optional[str] = field(
        default="",
        metadata={"help": "Path to STGR JSON (e.g. STGR-RL-filtered.json). If set, IDL runs only for samples whose id is in this file."},
    )
    temporal_reward_threshold: Optional[float] = field(
        default=0.9225,
        metadata={"help": "Adaptive VP gating: if GSPO temporal reward < threshold, use numpro VP"},
    )
    spatial_reward_threshold: Optional[float] = field(
        default=0.3107,
        metadata={"help": "Adaptive VP gating: if GSPO spatial reward < threshold, use darken VP"},
    )
    wo_vp_selector: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, bypass VP selector/darken and always use raw key frames"},
    )
    idl_candidate_mode: Optional[str] = field(
        default="ans_only",
        metadata={
            "help": "IDL candidate selection mode: "
                    "'ans_only' (ans_acc > baseline), "
                    "'grounding_only' (temporal+spatial > baseline, for STGR tasks), "
                    "'ans_or_grounding' (ans_acc > baseline OR grounding > baseline), "
                    "'ans_and_grounding' (ans_acc > baseline AND grounding > baseline, same as ans_and_both), "
                    "'total_reward' (total reward > baseline), "
                    "'task_aware' (STGR→grounding, others→ans_acc, RECOMMENDED)"
        },
    )
    idl_positive_ranking: Optional[str] = field(
        default="ans_acc",
        metadata={
            "help": "IDL positive ranking mode (how to rank candidates for top-k selection): "
                    "'ans_acc' (rank by answer accuracy), "
                    "'grounding' (rank by temporal+spatial reward), "
                    "'total_reward' (rank by total reward), "
                    "'task_aware' (STGR→grounding, others→ans_acc, RECOMMENDED)"
        },
    )
    idl_top_k: Optional[int] = field(
        default=2,
        metadata={"help": "IDL: number of top positive responses to select (top-1, top-2, top-3, ...). Default 2."},
    )
    idl_trainer_module: Optional[str] = field(
        default="",
        metadata={
            "help": "Python module path for IDL trainer class (e.g. trainer.grpo_trainer_idl_v3_nofallback_grounding). "
                    "Default: use trainer.grpo_trainer_idl (from trainer/__init__.py). "
                    "Class name must be Qwen2VLGRPOTrainerIDL."
        },
    )



reward_funcs_registry = {
    "ans_acc": ans_acc_reward,
    "ans_tiou": ans_tiou_reward,
    "ans_viou": ans_viou_reward,
    "thk_temporal_point": thk_temporal_point_reward,
    "thk_temporal_segment": thk_temporal_segment_reward,
    "thk_spatial": thk_spatial_reward,
    "format": format_reward
}





def main(script_args, training_args, model_args):
    # Get reward functions (wo_spatial/wo_tempspatial ablation applied in trainer reward computation)
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = get_data(script_args)

    # ============================================================
    # IDL: Pre-compute VP predictions if IDL is enabled
    # ============================================================
    vp_predictions = {}
    if getattr(script_args, "enable_idl", False):
        import importlib
        idl_module = getattr(script_args, "idl_trainer_module", "") or ""
        if idl_module.strip():
            mod = importlib.import_module(idl_module.strip())
            trainer_cls = getattr(mod, "Qwen2VLGRPOTrainerIDL")
            print(f"[IDL] IDL mode enabled, using {idl_module}.Qwen2VLGRPOTrainerIDL")
        else:
            from trainer import Qwen2VLGRPOTrainerIDL
            trainer_cls = Qwen2VLGRPOTrainerIDL
            print("[IDL] IDL mode enabled, using Qwen2VLGRPOTrainerIDL (default: trainer.grpo_trainer_idl)")
        from vp_selector_utils import precompute_vp_predictions

        # Determine VP predictions cache path
        vp_cache_path = script_args.vp_predictions_cache
        if not vp_cache_path:
            vp_cache_path = os.path.join(training_args.output_dir, "vp_predictions_cache.json")

        # Pre-compute VP predictions (rank 0 computes, others load from cache)
        import torch.distributed as dist
        is_main = not dist.is_initialized() or dist.get_rank() == 0

        if is_main and script_args.vp_selector_model_dir and not script_args.wo_vp_selector:
            vp_predictions = precompute_vp_predictions(
                vp_selector_model_dir=script_args.vp_selector_model_dir,
                dataset=dataset[script_args.dataset_train_split],
                cache_path=vp_cache_path,
                device="cuda:0",
                batch_size=16,
                key_object_json_path=script_args.key_object_json_path if script_args.key_object_json_path else None,
            )
        
        # Sync: wait for rank 0 to finish writing cache
        if dist.is_initialized():
            dist.barrier()

        # All ranks load from cache
        if not is_main and os.path.exists(vp_cache_path):
            import json
            with open(vp_cache_path, "r") as f:
                vp_predictions = json.load(f)
            print(f"[IDL] Rank {dist.get_rank()}: loaded {len(vp_predictions)} VP predictions from cache")
        elif not vp_predictions and os.path.exists(vp_cache_path):
            import json
            with open(vp_cache_path, "r") as f:
                vp_predictions = json.load(f)

        print(f"[IDL] Config: k1={script_args.idl_k1}, k2={script_args.idl_k2}, "
              f"alpha={script_args.idl_alpha}")
    else:
        trainer_cls = Qwen2VLGRPOTrainer  # if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified

    print("using: ", trainer_cls)
    print("training_args:", training_args)
    print("script_args:", script_args)
    print("model_args:", model_args)

    # Initialize the GRPO trainer
    # Build common kwargs
    trainer_kwargs = dict(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Add IDL-specific kwargs if enabled
    if getattr(script_args, "enable_idl", False):
        trainer_kwargs.update(
            enable_idl=True,
            idl_k1=script_args.idl_k1,
            idl_k2=script_args.idl_k2,
            idl_alpha=script_args.idl_alpha,
            vp_predictions=vp_predictions,
            darken_keyframe_dir=script_args.darken_keyframe_dir if script_args.darken_keyframe_dir else None,
            numpro_keyframe_dir=script_args.numpro_keyframe_dir if script_args.numpro_keyframe_dir else None,
            vp_keyframe_base_dir=getattr(script_args, "vp_keyframe_base_dir", "") or None,
            key_object_json_path=script_args.key_object_json_path if script_args.key_object_json_path else None,
            wo_vp_selector=script_args.wo_vp_selector,
            idl_candidate_mode=getattr(script_args, "idl_candidate_mode", "ans_only"),
            idl_positive_ranking=getattr(script_args, "idl_positive_ranking", "ans_acc"),
            idl_top_k=getattr(script_args, "idl_top_k", 2),
            temporal_reward_threshold=getattr(script_args, "temporal_reward_threshold", 0.9225),
            spatial_reward_threshold=getattr(script_args, "spatial_reward_threshold", 0.3107),
            idl_stgr_filter_json=getattr(script_args, "idl_stgr_filter_json", "") or None,
        )

    trainer = trainer_cls(**trainer_kwargs)
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
