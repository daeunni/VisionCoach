import os
import pandas as pd
import numpy as np
from loguru import logger as eval_logger


class PerceptionTest_Bench:
    def __init__(self, data_path, video_dir, think_mode=False):
        self.data_path = data_path
        self.video_dir = video_dir
        self.think_mode = think_mode
        print("think mode:", self.think_mode)

    def get_data(self):
        print("Loading PerceptionTest data...")
        df = pd.read_parquet(self.data_path)
        
        video_paths, image_input, text_input, all_docs = [], [], [], []
        
        for index, row in df.iterrows():
            doc = row.to_dict()
            # Convert answer_id to letter (0->A, 1->B, 2->C)
            doc["answer"] = chr(ord("A") + int(doc["answer_id"]))
            all_docs.append(doc)
            
            video_p, img, txt = self.process_data(doc)
            video_paths.extend(video_p)
            image_input.extend(img)
            text_input.extend(txt)
        
        print(f"Data loaded: {len(all_docs)} examples")
        return video_paths, image_input, text_input, all_docs

    def process_data(self, doc):
        video_path = self.get_video_path(doc)
        text = self.doc_to_text(doc)
        return [video_path], [None], [text]
    
    def get_video_path(self, doc):
        video_name = doc["video_name"]
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
        return video_path
    
    def doc_to_text(self, doc):
        question = doc["question"]
        options = doc["options"]
        option_str = "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)])
        question = question + "\n" + option_str

        if not self.think_mode:
            option_prompt = "Select the best answer to the multiple-choice question based on the video. Respond with only the letter (A, B, or C) of the correct option."
            full_prompt = option_prompt + "\n" + question + "\n"
        else:
            option_prompt = "Select the best answer to the multiple-choice question based on the video. You must first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The reasoning process MUST NOT be longer than 100 words. In the answer part, respond with only the letter (A, B, or C) of the correct option."
            full_prompt = "Question:" + question + "\n" + option_prompt
        return full_prompt


def perceptiontest_process_results(doc, pred, think=None, frame_shape=None):
    """Process results for a single example"""
    index2ans, all_choices = get_multi_choice_info(doc["options"])
    pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)
    
    data_dict = {
        "question_id": doc["question_id"],
        "video_name": doc["video_name"],
        "area": doc["area"],
        "reasoning": doc["reasoning"],
        "tag": list(doc["tag"]) if hasattr(doc["tag"], '__iter__') else doc["tag"],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
        "response": pred,
        "reasoning_process": think,
        "frame_shape": frame_shape,
    }
    return data_dict


def perceptiontest_aggregate_results(results):
    """Aggregate results and compute accuracy by category"""
    
    # By area (physics, semantics)
    area_scores = {}
    for result in results:
        area = result["area"]
        if area not in area_scores:
            area_scores[area] = {"correct": 0, "total": 0}
        area_scores[area]["total"] += 1
        if result["pred_answer"] == result["answer"]:
            area_scores[area]["correct"] += 1
    
    for area, scores in area_scores.items():
        acc = 100 * scores["correct"] / scores["total"] if scores["total"] > 0 else 0
        eval_logger.info(f"Area [{area}]: {acc:.1f}% ({scores['correct']}/{scores['total']})")
    
    # By reasoning type
    reasoning_scores = {}
    for result in results:
        reasoning = result["reasoning"]
        if reasoning not in reasoning_scores:
            reasoning_scores[reasoning] = {"correct": 0, "total": 0}
        reasoning_scores[reasoning]["total"] += 1
        if result["pred_answer"] == result["answer"]:
            reasoning_scores[reasoning]["correct"] += 1
    
    for reasoning, scores in reasoning_scores.items():
        acc = 100 * scores["correct"] / scores["total"] if scores["total"] > 0 else 0
        eval_logger.info(f"Reasoning [{reasoning}]: {acc:.1f}% ({scores['correct']}/{scores['total']})")
    
    # Overall
    total_correct = sum(1 for r in results if r["pred_answer"] == r["answer"])
    total = len(results)
    overall_acc = 100 * total_correct / total if total > 0 else 0
    eval_logger.info(f"Overall Performance: {overall_acc:.1f}% ({total_correct}/{total})")
    
    return overall_acc


def parse_answer(pred, doc):
    index2ans, all_choices = get_multi_choice_info(doc["options"])
    pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)
    return pred_ans


def get_multi_choice_info(options):
    """Get index to answer mapping"""
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """Parse the prediction from the generated response"""
    if response == "API Error" or response == "":
        return "API Error"

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates = []
    
    # Look for various patterns
    patterns = [
        lambda c: f"{c}.",
        lambda c: f"{c}:",
        lambda c: f"({c})",
        lambda c: f"{c} ",
        lambda c: f"\n{c}\n",
        lambda c: f" {c}\n",
        lambda c: f"\n{c} ",
        lambda c: f": {c}",
        lambda c: f":{c}",
        lambda c: f":\n{c}",
        lambda c: f"\n\n{c}",
        lambda c: f"**{c}**",
        lambda c: f"{{{c}}}",
    ]
    
    for choice in all_choices:
        for pattern in patterns:
            if pattern(choice) in response:
                candidates.append(pattern(choice))
    
    # Content-based matching
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
    
    if len(candidates) == 0:
        return "No Answer Found"
    elif len(candidates) > 1:
        start_indexes = [response.rfind(can) for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
        for choice in all_choices:
            if choice in pred_index:
                return choice
        return pred_index
    else:
        pred_index = candidates[0]
        for choice in all_choices:
            if choice in pred_index:
                return choice
        return pred_index





