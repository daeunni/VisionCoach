import os

os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
from datetime import datetime
from dataloader.perceptiontest import (
    PerceptionTest_Bench,
    perceptiontest_aggregate_results,
    perceptiontest_process_results,
    parse_answer
)
import argparse
import multiprocessing
import re


def parse_args():
    parser = argparse.ArgumentParser(description="PerceptionTest Evaluation")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/nas-ssd2/daeun/TrackGRPO/PerceptionTest/mc_question_val/validation-00000-of-00001.parquet",
        help="Path to the parquet data file.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/nas-ssd2/daeun/.cache/perceptiontest_val/videos",
        help="Path to the video directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/your/model",
        help="Path to the model.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="perceptiontest", help="Experiment name."
    )
    parser.add_argument("--models_per_gpu", type=int, default=1)
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="Path to YAML file containing model keyword arguments.",
    )
    parser.add_argument(
        "--vote", type=str, default="majority_voting", help="Voting method."
    )
    parser.add_argument("--think_mode", action="store_true", help="Use Chain-of-Thought prompting.")
    parser.add_argument("--N", type=int, default=1, help="Number of paths for voting")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N examples")

    args = parser.parse_args()

    try:
        import yaml
        with open(args.model_kwargs, "r") as f:
            args.model_kwargs = yaml.safe_load(f)
        if not isinstance(args.model_kwargs, dict):
            raise ValueError("YAML file must contain a dictionary")
    except ImportError:
        parser.error("PyYAML is required for YAML parsing. Install with: pip install pyyaml")
    except FileNotFoundError:
        parser.error(f"Model kwargs file not found: {args.model_kwargs}")
    except Exception as e:
        parser.error(f"Error parsing YAML file: {e}")
    
    return args


def get_cuda_visible_devices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible_devices:
        return []
    gpu_list = [
        int(gpu_id.strip())
        for gpu_id in cuda_visible_devices.split(",")
        if gpu_id.strip()
    ]
    return gpu_list


def build_model(model_path, **model_kwargs):
    from models.model_vllm import QwenVL_VLLM

    model = QwenVL_VLLM(
        model_path,
        rt_shape=True,
        **model_kwargs,
    )
    return model


def evaluate_chunk(video_paths, image, text_input, docs, gpu_id, args, queue, save_interval=100):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {str(gpu_id)}] Processing {len(video_paths)} examples.")

        results = []
        bs = 1
        model = build_model(args.model_path, **args.model_kwargs)

        think_contents = []
        frame_shape_list = []
        
        # For intermediate saving
        os.makedirs("./logs/perceptiontest_logs", exist_ok=True)
        checkpoint_path = f"./logs/perceptiontest_logs/checkpoint_{args.exp_name}_gpu{gpu_id}.json"

        def save_checkpoint(current_idx):
            """Save intermediate results every save_interval examples"""
            checkpoint_metrics = [
                perceptiontest_process_results(docs[i], results[i], think_contents[i], frame_shape_list[i])
                for i in range(len(results))
            ]
            checkpoint_data = {
                "gpu_id": gpu_id,
                "processed": current_idx,
                "total": len(video_paths),
                "metrics": checkpoint_metrics
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            print(f"[GPU {gpu_id}] Checkpoint saved: {current_idx}/{len(video_paths)}")

        def process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc):
            N = args.N
            score_list = []
            pred_list = []
            n_think = ["" for _ in range(N)]
            frame_shape = None

            for path_idx in range(N):
                think_mode = args.think_mode

                output_list, frames, fps, shape = model(
                    batch_video_paths,
                    batch_text_input,
                    query_image=batch_image,
                )
                request_output = output_list[0]
                completions = request_output.outputs
                pred_text = completions[0].text
                frame_shape = shape

                if think_mode:
                    match = re.search(r"<answer>(.*?)</answer>", pred_text, re.DOTALL)
                    ans = None
                    if match:
                        ans = match.group(1).strip()
                        if ans not in ["A", "B", "C"]:
                            score = 0
                            pred_list.append("NA")
                            score_list.append(score)
                            continue
                        else:
                            pred_list.append(ans)
                    else:
                        score = 0
                        pred_list.append("NA")
                        score_list.append(score)
                        continue
                    
                    match = re.search(r"<think>(.*?)</think>", pred_text, re.DOTALL)
                    if match:
                        think_process = match.group(1).strip()
                        n_think[path_idx] = think_process
                    else:
                        score_list.append(0)
                        continue

                    score = 1.0
                else:
                    ans = parse_answer(pred_text, batch_doc[0])
                    if ans in ["A", "B", "C"]:
                        pred_list.append(ans)
                        score = 1.0
                    else:
                        pred_list.append("NA")
                        score = 0.0
                score_list.append(score)

            # Majority voting
            choice_score = {"A": 0, "B": 0, "C": 0}
            for i in range(len(pred_list)):
                if pred_list[i] == "NA":
                    continue
                choice_score[pred_list[i]] += score_list[i]
            
            pred_text = max(choice_score, key=choice_score.get)

            print(
                batch_doc[0]["video_name"],
                "GT:", batch_doc[0]["answer"],
                "Pred:", pred_text,
            )

            think_text = ""
            for idx in range(len(pred_list)):
                if pred_list[idx] == pred_text:
                    think_text = n_think[idx]
                    break

            results.append(pred_text)
            think_contents.append(think_text)
            frame_shape_list.append(frame_shape)

        idx = 0
        while idx < len(video_paths):
            batch_size = min(bs, len(video_paths) - idx)
            batch_video_paths = video_paths[idx : idx + batch_size]
            batch_text_input = text_input[idx : idx + batch_size]
            batch_image = image[idx : idx + batch_size]
            batch_doc = docs[idx : idx + batch_size]

            process_batch(batch_video_paths, batch_text_input, batch_image, batch_doc)
            idx += batch_size
            print(f"GPU ID:{gpu_id}, {idx}/{len(video_paths)}")
            
            # Save checkpoint every save_interval examples
            if idx % save_interval == 0:
                save_checkpoint(idx)

        # Final save
        save_checkpoint(len(video_paths))

        metrics = [
            perceptiontest_process_results(docs[i], results[i], think_contents[i], frame_shape_list[i])
            for i in range(len(docs))
        ]
        queue.put((metrics, results, None))
        print(f"[GPU {gpu_id}] Finished processing.")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        queue.put((None, None, error_msg))


def evaluate(args, num_gpus, gpu_list):
    benchmark = PerceptionTest_Bench(
        args.data_path,
        args.video_dir,
        think_mode=args.think_mode
    )

    if len(gpu_list) == 0:
        gpu_list = list(range(num_gpus))

    video_paths, image, text_input, docs = benchmark.get_data()
    total = len(video_paths)

    models_per_gpu = args.models_per_gpu
    gpu_list_new = [
        gpu_list[i] for i in range(len(gpu_list)) for j in range(models_per_gpu)
    ]
    gpu_list = gpu_list_new

    print(f"Total examples: {total}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"GPU list: {gpu_list}")

    num_gpus = num_gpus * models_per_gpu
    chunk_size = (total + num_gpus - 1) // num_gpus
    chunks = []

    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= end:
            break
        chunks.append((
            video_paths[start:end],
            image[start:end],
            text_input[start:end],
            docs[start:end],
        ))

    queue = multiprocessing.Queue()
    processes = []

    for i, (vp_chunk, img_chunk, txt_chunk, docs_chunk) in enumerate(chunks):
        p = multiprocessing.Process(
            target=evaluate_chunk,
            args=(vp_chunk, img_chunk, txt_chunk, docs_chunk, gpu_list[i], args, queue, args.save_interval),
        )
        p.start()
        processes.append(p)

    all_metrics = []
    all_results = []
    for _ in processes:
        metrics, results, error = queue.get()
        if error is not None:
            print(f"Worker error: {error}")
            for p in processes:
                p.terminate()
            exit(1)
        all_metrics.extend(metrics)
        all_results.extend(results)

    for p in processes:
        p.join()

    acc = perceptiontest_aggregate_results(all_metrics)
    print("Final accuracy:", acc)

    queue.close()
    return all_metrics, all_results


def main():
    print("Start Time:", datetime.now())
    args = parse_args()
    print(args)

    num_gpus = int(os.getenv("NUM_GPUS", 1))
    gpu_list = get_cuda_visible_devices()

    metrics, results = evaluate(args, num_gpus=num_gpus, gpu_list=gpu_list)

    os.makedirs("./logs/perceptiontest_logs", exist_ok=True)
    
    metrics_path = f"./logs/perceptiontest_logs/metrics_{args.exp_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    results_path = f"./logs/perceptiontest_logs/results_{args.exp_name}.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print("End Time:", datetime.now())


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
