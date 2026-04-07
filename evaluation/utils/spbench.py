import torch
from typing import Optional

# ====================== helpers ====================== # 

def normalize_number(num_str: str) -> Optional[float]:
    """Convert string number to float, handling commas."""
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None

def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Calculate mean relative accuracy for regression tasks."""
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def calculate_final_metrics(results):
    """Calculate metrics from a list of results."""
    mean_acc_rewards = [s["reward"] for s in results if s.get("problem_type") != "regression" and "reward" in s]
    mean_mra_rewards = [s["reward"] for s in results if s.get("problem_type") == "regression" and "reward" in s and s.get("prediction") != "error"]
    task_list = [
        "object_counting",
        "object_size_estimation",
        "object_abs_distance",
        "object_rel_distance",
        "object_rel_direction"
    ]

    final_metrics = {"mean_acc": 0.0, "mean_mra": 0.0, "mean_all": 0.0, "tasks": []}
    if mean_acc_rewards:
            final_metrics["mean_acc"] = torch.tensor(mean_acc_rewards, dtype=torch.float32).mean().item()
    if mean_mra_rewards:
            final_metrics["mean_mra"] = torch.tensor(mean_mra_rewards, dtype=torch.float32).mean().item()
    if mean_acc_rewards or mean_mra_rewards:
        all_rewards = torch.cat([torch.tensor(mean_acc_rewards, dtype=torch.float32), torch.tensor(mean_mra_rewards, dtype=torch.float32)])
        final_metrics["mean_all"] = all_rewards.mean().item()
    
    for task in task_list:
        task_results = [s["reward"] for s in results if s.get("original_question_type") == task]
        task_metrics = torch.tensor(task_results, dtype=torch.float32).mean().item()
        final_metrics["tasks"].append({"task": task, "reward": task_metrics, "count": len(task_results)})

        if task == "object_rel_direction_easy":
            # merge object_rel_direction
            merged_rel_direction_results = [s["reward"] for s in results if "object_rel_direction" in s.get("original_question_type")]
            merged_rel_direction_metrics = torch.tensor(merged_rel_direction_results, dtype=torch.float32).mean().item()
            final_metrics["tasks"].append({"task": "object_rel_direction", "reward": merged_rel_direction_metrics, "count": len(merged_rel_direction_results)})

    return final_metrics


# ====================== exported functions ====================== # 

def build_spbench_prompt(row_dict) -> str:
    prompt = row_dict['question']
    SFT_TYPE_TEMPLATE = {
        "multiple choice": " Please answer with the option's letter from the given choices (e.g., A, B, etc.).",
        "numerical": " Please answer with the only numerical value (e.g., 42, 3.14, etc.).",
        "regression": " Please answer with the only numerical value (e.g., 42, 3.14, etc.).",
        "verbal": " Please answer the question simply.",
    }
    prompt += SFT_TYPE_TEMPLATE[row_dict["problem_type"]]
    return prompt

def calculate_spbench_reward(clean_ans_gt: str, clean_ans_pred: str, row_dict: dict) -> float:
    question_type = row_dict['problem_type']
    """Calculate reward based on question type and model output."""
    if question_type == "multiple choice":
        return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0
    elif question_type == "regression" or question_type == "numerical":
        gt_number = normalize_number(clean_ans_gt)
        pred_number = normalize_number(clean_ans_pred)
        if gt_number is None or pred_number is None:
            return 0.0
        return mean_relative_accuracy(pred_number, gt_number)
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


def calculate_spbench_metrics(results):
    """Calculate metrics from a list of results."""
    final_metrics = {
        "Overall": calculate_final_metrics(results),
        "MV": calculate_final_metrics([s for s in results if s.get("modal_type") == "MV"]),
        "SI": calculate_final_metrics([s for s in results if s.get("modal_type") == "SI"]),
    }

    return final_metrics
