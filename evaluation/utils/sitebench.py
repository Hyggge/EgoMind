import torch

# ====================== exported functions ====================== # 

def build_sitebench_prompt(row_dict) -> str:
    prompt = row_dict['question']
    prompt += "Please answer with the option's letter from the given choices (e.g., A, B, etc.)."
    return prompt

def calculate_sitebench_reward(clean_ans_gt: str, clean_ans_pred: str, row_dict: dict) -> float:
    return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0

def calculate_sitebench_metrics(results):
    """Calculate metrics from a list of results."""
    # Get all unique task types and datasets
    all_tasks = set(s.get("category") for s in results if s.get("category"))
    all_datasets = set(s.get("dataset") for s in results if s.get("dataset"))
    final_metrics = {"mean_all": 0.0, "tasks": [], "datasets": []}
  
    # Calculate overall average accuracy
    all_rewards = [s["reward"] for s in results if "reward" in s]
    if all_rewards:
        final_metrics["mean_all"] = torch.tensor(all_rewards, dtype=torch.float32).mean().item()
  
    # Calculate accuracy for each task
    for task in sorted(all_tasks):  # Sort to ensure consistent output order
        task_results = [s["reward"] for s in results if s.get("category") == task and "reward" in s]
        if task_results:
            task_metrics = torch.tensor(task_results, dtype=torch.float32).mean().item()
        else:
            task_metrics = 0.0
      
        final_metrics["tasks"].append({
            "task": task, 
            "reward": task_metrics, 
            "count": len(task_results)
        })
    
    # Calculate accuracy for each dataset
    for dataset in sorted(all_datasets):  # Sort to ensure consistent output order
        dataset_results = [s["reward"] for s in results if s.get("dataset") == dataset and "reward" in s]
        if dataset_results:
            dataset_metrics = torch.tensor(dataset_results, dtype=torch.float32).mean().item()
        else:
            dataset_metrics = 0.0
      
        final_metrics["datasets"].append({
            "dataset": dataset, 
            "reward": dataset_metrics, 
            "count": len(dataset_results)
        })
  
    return final_metrics