import re
import torch
import numpy as np

MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv"
]

NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]

SPECIAL_QUESTION_TYPES = [
    "view_change_infer",
]

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

QUESTION_TEMPLATE = (
    "Question: {Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

PROMPT_TEMPLATES = {
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1).",
    "special_post_prompt": "Please output the answer directly.",
}

Low = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]

Middle = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

High = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv"
]

# ====================== helpers ====================== # 
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def exact_match(pred, target):
    # return 1. if pred.lower() == target.lower() else 0.
    pred = pred.lower()
    target = target.lower()
    if pred.lower() == target.lower():
        return 1.
    elif pred in target:
        return 1.
    elif pred[0] == target:
        return 1.
    else:
        return 0

def abs_dist_norm(pred, target):
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def compute_vci_metric(pred, answer):
    acion_list = ["move_right", "move_left", 
                  "move_forward", "move_backward", 
                  "move_up", "move_down", 
                  "rotate_right", "rotate_left",
                  "rotate_up", "rotate_down"]
    action_order = ["move_right_left",
                    "move_up_down",
                    "move_forward_backward",
                    "rotate_right_left",
                    "rotate_up_down"]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []

    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))
    
    mra_list = []
    for gt, answer in zip(gt_list, answer_list):
        mra = mean_relative_accuracy(gt, answer, start=.5, end=.95, interval=.05)
        mra_list.append(mra)

    return np.mean(mra_list)

def process_na(pred, task):
    numbers = re.findall(r'(?<!\^)\d+\.\d+|(?<!\^)\d+', pred)
    # Convert the matched numbers to float or int
    extracted_numbers = [float(num) if '.' in num else int(num) for num in numbers]
    if task in ["depth_prediction_oc_mv", 
                "depth_prediction_oo_mv",
                "distance_prediction_oc_mv",
                "distance_prediction_oo_mv",
                ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]

def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}

# ====================== exported functions ====================== # 

def build_sparbench_prompt(row_dict) -> str:
    prompt = row_dict["question"]
    if row_dict['task'] in NA_QUESTION_TYPES:
        prompt = prompt + '\n' + PROMPT_TEMPLATES['na_post_prompt']
    elif row_dict['task'] in MCA_QUESTION_TYPES:
        post_prompt = ""
        if row_dict['task']in ['position_matching', "camera_motion_infer"]:
            post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
        prompt = prompt + "\n" + post_prompt + "\n" + PROMPT_TEMPLATES['mca_post_prompt']
    elif row_dict['task'] in SPECIAL_QUESTION_TYPES:
        prompt = prompt + '\n' + PROMPT_TEMPLATES['special_post_prompt']
    else:
        raise ValueError(f"Unknown question type: {row_dict['task']}")
    
    return prompt


def calculate_sparbench_reward(clean_ans_gt: str, clean_ans_pred: str, row_dict: dict) -> float:
    reward = 0
    if row_dict['task'] in MCA_QUESTION_TYPES:
        reward = exact_match(clean_ans_pred, clean_ans_gt)
    elif row_dict['task'] in NA_QUESTION_TYPES:
        reward = mean_relative_accuracy(
            float(process_na(clean_ans_pred, row_dict['task'])), 
            float(clean_ans_gt), 
            start=.5, end=.95, interval=.05
        )
    elif row_dict['task'] in SPECIAL_QUESTION_TYPES:
        assert row_dict['task'] == "view_change_infer"
        reward =  compute_vci_metric(clean_ans_pred, clean_ans_gt)
    else:
        raise ValueError(f"Unknown question type: {row_dict['task']}")
    
    return convert_numpy_types(reward)

def calculate_sparbench_metrics(results):
    import pandas as pd
    
    # Convert to DataFrame for easier processing
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        return {"overall": {}, "by_img_type": {}}
    
    output = {}
    
    # Helper function for calculating weighted metrics
    def calculate_weighted_metrics(df):
        metrics = {}
        
        # Calculate per-task average scores
        task_metrics = {}
        task_counts = {}
        
        for task_name, task_group in df.groupby('task'):
            if 'reward' in task_group.columns:
                mean_score = task_group['reward'].mean()
                task_metrics[task_name] = mean_score
                task_counts[task_name] = len(task_group)
        
        # Calculate WEIGHTED overall_accuracy
        weighted_sum_overall = 0.0
        total_weight_overall = 0.0
        
        for task_name, avg_score in task_metrics.items():
            if pd.notna(avg_score):
                task_weight = task_counts[task_name]
                weighted_sum_overall += avg_score * task_weight
                total_weight_overall += task_weight
        
        metrics['overall_accuracy'] = weighted_sum_overall / total_weight_overall if total_weight_overall > 0 else 0.0
        
        # Calculate WEIGHTED Low, Middle, High category scores
        categories = {"Low": Low, "Middle": Middle, "High": High}
        
        for cat_name, tasks_in_category in categories.items():
            weighted_sum_category = 0.0
            total_weight_category = 0.0
            
            for task_name in tasks_in_category:
                if task_name in task_metrics and pd.notna(task_metrics[task_name]):
                    task_weight = task_counts[task_name]
                    weighted_sum_category += task_metrics[task_name] * task_weight
                    total_weight_category += task_weight
            
            metrics[cat_name] = weighted_sum_category / total_weight_category if total_weight_category > 0 else 0.0
        
        # Add individual task metrics
        for task_name, avg_score in task_metrics.items():
            metrics[task_name] = avg_score
        
        return metrics
    
    # Calculate overall metrics
    output['overall'] = calculate_weighted_metrics(results_df)
    
    # Calculate metrics by image_type if column exists
    if 'img_type' in results_df.columns:
        output['by_img_type'] = {}
        for img_type, img_type_df in results_df.groupby('img_type'):
            output['by_img_type'][img_type] = calculate_weighted_metrics(img_type_df)
    else:
        output['by_img_type'] = {}
    
    return output
