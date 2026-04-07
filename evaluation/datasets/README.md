# Dataset Preparation

Paths below are relative to the **EgoMind** repository root. Each benchmark lives under `evaluation/datasets/<name>/` and includes a `qa_processed.jsonl` file. Media paths inside the JSONL are relative to that directory).

You need to download raw videos/images and mirror the `data/` layout so those relative paths resolve. A fresh clone may only contain `qa_processed.jsonl` until you add `data/`.

## VSI-Bench

**Source**: [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)

**JSONL fields (per row)**: `id`, `video`, `question`, `answer`, `problem_type`, `original_question_type` — `video` paths look like `data/arkitscenes/<id>.mp4`, `data/scannet/<scene>.mp4`, `data/scannetpp/<id>.mp4`.

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

**JSONL fields**: includes `image` (path like `data/images/SPAR_Bench_*.png`), plus `format_type`, `img_type`, `source`, `task`.

```
evaluation/datasets/SPAR-Bench/
├── qa_processed.jsonl
└── data/
    └── images/
```

## SITE-Bench

**Source**: [SITE-Bench](https://huggingface.co/datasets/franky-veteran/SITE-Bench)

**JSONL fields**: `id`, `question`, `answer`, `video`, `dataset`, `category`. Place media under `data/` following the paths in `qa_processed.jsonl`.

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

**JSONL fields**: `images` (list of paths), `modal_type`, `problem_type`, etc. Paths look like `data/SPBench-MV-images/<scene>/<frame>.jpg` or under `data/SPBench-SI-images/`.

```
evaluation/datasets/SPBench/
├── qa_processed.jsonl
└── data/
    ├── SPBench-MV-images/
    └── SPBench-SI-images/
```
