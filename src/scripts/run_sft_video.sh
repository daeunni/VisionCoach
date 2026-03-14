

SCRIPT_PATH="$(realpath "$0")"

cd src/r1-v
# Fix CUDA path for DeepSpeed
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Fix memory fragmentation
export VIDEO_PIXELS_FACTOR=128 
mkdir -p $TRITON_CACHE_DIR

MODEL_PATH="/.cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
EXP_NAME="sft_test_bbox"
OUT_DIR="./results_train_loss/${EXP_NAME}"

DATA_ROOT=$(python -c "from configs.data_root import DATA_ROOT; print(DATA_ROOT)")
mkdir -p $OUT_DIR

cp "$SCRIPT_PATH" "${OUT_DIR}/train_script.sh"
echo "Training script saved to ${OUT_DIR}/train_script.sh ($(date))"

# Auxiliary Loss Configuration
AUX_LOSS_WEIGHT=0.01  
BBOX_LOSS_WEIGHT=0.1
TIME_LOSS_WEIGHT=0.0
AUX_LOSS_TYPE="l1"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/sft_multi_task.py \
    --output_dir $OUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name "${DATA_ROOT}/json_data/STGR-SFT-filtered.json" \
    --deepspeed "local_scripts/zero3.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 100 \
    --run_name $EXP_NAME \
    --save_steps 2000 \
    --max_grad_norm 5 \
    --save_only_model true \
    --aux_loss_weight $AUX_LOSS_WEIGHT \
    --bbox_loss_weight $BBOX_LOSS_WEIGHT \
    --time_loss_weight $TIME_LOSS_WEIGHT \
    --aux_loss_type $AUX_LOSS_TYPE
