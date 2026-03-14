# #!/bin/bash

SCRIPT_PATH="$(realpath "$0")"

cd src/r1-v


export VIDEO_PIXELS_FACTOR=128

MODEL_PATH="./results_train/sft_train_all/checkpoint-7500"
EXP_NAME="0226_grpo_idl_sft_train_all_alpha_0.1_can_ans_and_grounding_pos_ans_acc_VPSelector"
OUT_DIR="./results_rl_idl/${EXP_NAME}"

spatial_iou_mode="avg"
identity_match_mode="soft"       # "none", "soft", "strict"
spatial_norm_mode="matched"      # "all", "matched"
correct_tempgate="true"          # "true", "false"

enable_idl="true"                                   
idl_k1="2.21"
idl_alpha="0.1"                                     
wo_vp_selector="false"                            

idl_candidate_mode="ans_and_grounding"
idl_positive_ranking="ans_acc" 

VP_PREDICTIONS_CACHE="./0226_vp_predictions_cache.json"
VP_KEYFRAME_BASE_DIR="./results/open_o3_video_vp"
DARKEN_KEYFRAME_DIR="./results/open_o3_video_vp/darken"
NUMPRO_KEYFRAME_DIR="./results/open_o3_video_vp/numpro/keyframes"

KEY_OBJECT_JSON_PATH="./results/open_o3_video_vp/key_obj/STGR-SFT-filtered_key_obj_extracted.json"
IDL_STGR_FILTER_JSON="./data/json_data/STGR-RL-filtered.json"

IDL_TRAINER_MODULE="trainer.grpo_trainer_idl_v3_nofallback_grounding"

#----------------------------------------------------------
# Environment Setup
#----------------------------------------------------------
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)

BASE_MODEL_DIR="./model"
for fname in preprocessor_config.json chat_template.json; do
  if [ -f "${BASE_MODEL_DIR}/${fname}" ] && [ ! -f "${MODEL_PATH}/${fname}" ]; then
    echo "Copying ${fname} from ${BASE_MODEL_DIR} to ${MODEL_PATH}"
    cp "${BASE_MODEL_DIR}/${fname}" "${MODEL_PATH}/"
  fi
done

DATA_ROOT=$(python -c "from configs.data_root import DATA_ROOT; print(DATA_ROOT)")
mkdir -p ./train_logs
mkdir -p $OUT_DIR

cp "$SCRIPT_PATH" "${OUT_DIR}/train_script.sh"
echo "Training script saved to ${OUT_DIR}/train_script.sh ($(date))"

LOG_FILE="${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: ${LOG_FILE}"

#----------------------------------------------------------
# Launch Training
#----------------------------------------------------------
echo "=============================================="
echo " GRPO + IDL Training"
echo "=============================================="
echo " Model:         ${MODEL_PATH}"
echo " Experiment:    ${EXP_NAME}"
echo " IDL enabled:   ${enable_idl}"
echo " IDL k1:        ${idl_k1}"
echo " IDL alpha:     ${idl_alpha}"
echo " IDL candidate: ${idl_candidate_mode}"
echo " IDL ranking:   ${idl_positive_ranking}"
echo " WO VP Select:  ${wo_vp_selector}"
echo " VP Cache:      ${VP_PREDICTIONS_CACHE}"
echo " VP KF Base:    ${VP_KEYFRAME_BASE_DIR}"
echo " IDL STGR filter: ${IDL_STGR_FILTER_JSON:- (none)}"
echo " IDL Trainer:   ${IDL_TRAINER_MODULE:-trainer.grpo_trainer_idl (default)}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12323" \
    src/open_r1/grpo.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "${DATA_ROOT}/json_data/STGR-RL.json" \
    --deepspeed "local_scripts/zero3.json" \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    --save_steps 5000 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 4 \
    --spatial_iou_mode $spatial_iou_mode \
    --identity_match_mode $identity_match_mode \
    --spatial_norm_mode $spatial_norm_mode \
    --correct_tempgate $correct_tempgate \
    --enable_idl $enable_idl \
    --idl_k1 $idl_k1 \
    --idl_alpha $idl_alpha \
    --wo_vp_selector $wo_vp_selector \
    --idl_candidate_mode $idl_candidate_mode \
    --idl_positive_ranking $idl_positive_ranking \
    --vp_predictions_cache $VP_PREDICTIONS_CACHE \
    --darken_keyframe_dir $DARKEN_KEYFRAME_DIR \
    --vp_keyframe_base_dir $VP_KEYFRAME_BASE_DIR \
    --key_object_json_path $KEY_OBJECT_JSON_PATH \
    ${IDL_STGR_FILTER_JSON:+--idl_stgr_filter_json $IDL_STGR_FILTER_JSON} \
    ${IDL_TRAINER_MODULE:+--idl_trainer_module $IDL_TRAINER_MODULE}
