#!/bin/bash
LLM_PATH="Qwen/Qwen2.5-72B-Instruct"
echo "LLM_PATH: $LLM_PATH"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export HF_HOME=/nas-ssd2/daeun/.cache/huggingface
export VLLM_USAGE_STATS_DISABLE=1
export PYTHONPATH="${PYTHONPATH}:/nas-ssd2/daeun/TrackGRPO/Open-o3-Video/eval"

MODEL_PATHS=(
    # your model path here 
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "=========================================="
    echo "** Running eval for: $MODEL_PATH"
    echo "=========================================="

    MODEL_PATH_CLEAN="${MODEL_PATH%/}"
    EXP_NAME=$(basename "${MODEL_PATH_CLEAN%%/checkpoint-*}")

    # Fix preprocessor_config.json (Qwen2_5_VLImageProcessor -> Qwen2VLImageProcessor)
    if [ -f "${MODEL_PATH_CLEAN}/preprocessor_config.json" ]; then
        sed -i 's/Qwen2_5_VLImageProcessor/Qwen2VLImageProcessor/g' "${MODEL_PATH_CLEAN}/preprocessor_config.json"
        echo "** Fixed preprocessor_config.json in ${MODEL_PATH_CLEAN}"
    fi
    echo "** EXP_NAME: $EXP_NAME"

    # for v-star
    mkdir -p ./logs/vstar_logs
    MODEL_KWARGS="./config/vstar.yaml"
    NUM_GPUS=4 CUDA_VISIBLE_DEVICES=8,5,6,7 python ./test/test_vstar_multi_images.py \
        --video_folder "/nas-ssd2/daeun/TrackGRPO/V-STaR/videos/" \
        --anno_file "/nas-ssd2/daeun/TrackGRPO/V-STaR/V_STaR_test.json" \
        --result_file "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
        --model_path "$MODEL_PATH" \
        --model_kwargs $MODEL_KWARGS \
        --think_mode

    vLLM version - much faster with batch processing
    CUDA_VISIBLE_DEVICES=8,5,6,7 python ./test/eval_vstar_vllm.py \
        --result_file  "./logs/vstar_logs/${EXP_NAME}_vstar.json" \
        --model_path $LLM_PATH \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        > "./logs/vstar_logs/eval_${EXP_NAME}_vstar.log" 2>&1
done


# for videomme
mkdir -p ./logs/videomme_logs
MODEL_KWARGS="./config/video_mme.yaml"
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=5,6,7,8 python ./test/test_videomme.py \
    --exp_name "${EXP_NAME}_mme" \
    --data_dir "./Video-MME" \
    --model_path $MODEL_PATH \
    --model_kwargs $MODEL_KWARGS \
    --N 1 \
    --vote 'majority_voting' 

# for videommmu
mkdir -p ./logs/videommmu_logs
MODEL_KWARGS="./config/video_mmmu.yaml"
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=5,6,7,8 python ./test/test_videommmu.py \
    --exp_name "${EXP_NAME}_videommmu" \
    --data_dir "./VideoMMMU/" \
    --model_path $MODEL_PATH \
    --model_kwargs $MODEL_KWARGS \
    --N 1 \
    --vote 'majority_voting' \
    --think_mode 
    > "./logs/videommmu_logs/${EXP_NAME}_videommmu.log" 2>&1


# for worldsense
mkdir -p ./logs/world_logs
MODEL_KWARGS="./config/world_sense.yaml"
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=5,6,7,8 python ./test/test_worldsense.py \
    --exp_name "${EXP_NAME}_wds" \
    --data_dir "./WorldSense" \
    --model_path $MODEL_PATH \
    --model_kwargs $MODEL_KWARGS \
    --N 1 \
    --vote 'majority_voting' \
    --think_mode > "./logs/world_logs/${EXP_NAME}_wds.log" 2>&1


# for perceptiontest
mkdir -p ./logs/perceptiontest_logs
MODEL_KWARGS="./config/perceptiontest.yaml"
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=5,6,7,8 python ./test/test_perceptiontest.py \
    --exp_name "${EXP_NAME}_pt" \
    --data_path "./mc_question_val/validation-00000-of-00001.parquet" \
    --video_dir "./perceptiontest_val/videos" \
    --model_path $MODEL_PATH \
    --model_kwargs $MODEL_KWARGS \
    --N 1 \
    --vote 'majority_voting' \
    --think_mode 
    > "./logs/perceptiontest_logs/${EXP_NAME}_pt.log" 2>&1


