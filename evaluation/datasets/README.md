# Dataset Preparation

Paths below are relative to the **EgoMind** repository root. Each benchmark lives under `evaluation/datasets/<name>/` and includes a `qa_processed.jsonl` file. Media paths inside the JSONL are relative to that directory (see `run_eval.resolve_media_paths`).

You need to download raw videos/images and mirror the `data/` layout so those relative paths resolve. A fresh clone may only contain `qa_processed.jsonl` until you add `data/`.

## VSI-Bench

**Source**: [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)

**JSONL fields:** `answer`, `id`, `original_question_type`, `problem_type`, `question`, `video`  
**Media:** `video` → `data/arkitscenes/*.mp4`, `data/scannet/*.mp4`, `data/scannetpp/*.mp4`

```
evaluation/datasets/VSI-Bench/
├── qa_processed.jsonl
└── data/
    ├── arkitscenes/
    ├── scannet/
    └── scannetpp/
```

## SPAR-Bench

**Source**: [SPAR-Bench](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)

**JSONL fields:** `answer`, `format_type`, `id`, `image`, `img_type`, `question`, `source`, `task`  
**Media:** `image` → `data/images/*.png`

```
evaluation/datasets/SPAR-Bench/
├── qa_processed.jsonl
└── data/
    └── images/
```

## SITE-Bench

**Source**: [SITE-Bench](https://huggingface.co/datasets/franky-veteran/SITE-Bench)

**JSONL fields:** `answer`, `category`, `dataset`, `id`, `question`, `video`  
**Media:** `video` → under `data/` (layout per row in `qa_processed.jsonl`)

```
evaluation/datasets/SITE-Bench/
├── qa_processed.jsonl
└── data/
    ├── ActivityNet/
    ├── MLVU/
    ├── MVBench/
    ├── TGIF_Zero_Shot_QA/
    ├── VSI-Bench/
    ├── VideoMME/
    ├── openeqa/
    └── tvqa/
```

## SPBench

**Source**: [SPBench](https://huggingface.co/datasets/hongxingli/SPBench)

**JSONL fields:** `answer`, `id`, `images`, `modal_type`, `original_question_type`, `problem_type`, `question`  
**Media:** `images` (list) → `data/SPBench-MV-images/...`, `data/SPBench-SI-images/...`

```
evaluation/datasets/SPBench/
├── qa_processed.jsonl
└── data/
    ├── SPBench-MV-images/
    └── SPBench-SI-images/
```
