# TMLU

The TMLU benchmark is an evaluation suit tailored for assessing advanced knowledge
and reasoning capability in LLMs under Taiwanese Mandarin, in the format of multiple choice questions. TMLU contains a wide range of subjects spanning social science, STEM, humanities, Taiwan-specific content, and others, across middle school to professional levels.

Repository: https://github.com/MiuLab/TMLU


## Installation

We recommend using `uv` for fast dependency management:

```bash
uv pip venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```


## Baselines

- `Gemma-3-27B-IT`: `google/gemma-3-27b-it`
- `Phi-4`: `microsoft/phi-4`
- `Phi-4-MM-IT`: `microsoft/Phi-4-multimodal-instruct`
- `Qwen-3-32B`: `Qwen/Qwen3-32B`
- `Qwen-3-14B`: `Qwen/Qwen3-14B`
- `Deepseek-R1-Llama-8B`: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `Deepseek-R1-Qwen-32B`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

## Usage

Run the following command to evaluate the model:

```bash
python3 tmlu_eval.py \
    --backend vllm \
    --model <model_name> \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_length 8192 \
    --max_tokens 4096 \
    --subsets ALL \
    --tensor_parallel_size 1 \
    --log_dir <log_dir> \
    --few_shot_num 0 \ # To prevent <think> from being suppressed
    --apply_chat_template \ # New: add chat template
    --cot
```

## Results

|                 | Gemma-3-27B-IT | Phi-4  | Phi-4-MM-IT | Qwen-3-32B | Qwen-3-14B | Deepseek-R1-Llama-8B | Deepseek-R1-Qwen-32B |
|-----------------|----------------|--------|-------------|------------|------------|----------------------|----------------------|
| Social Science  | 74.29%         | 68.04% | 44.51%      | 78.17%     | 75.14%     | 48.49%               | 78.52%               |
| STEM            | 72.73%         | 62.09% | 32.85%      | 87.26%     | 82.24%     | 43.64%               | 82.03%               |
| Humanities      | 72.21%         | 65.32% | 37.73%      | 80.06%     | 79.90%     | 46.51%               | 76.56%               |
| Taiwan Specific | 78.81%         | 71.86% | 49.25%      | 80.77%     | 84.76%     | 54.04%               | 79.10%               |
| Others          | 53.31%         | 45.13% | 37.19%      | 70.02%     | 65.25%     | 33.34%               | 63.94%               |
| Average         | 70.27%         | 62.49% | 40.30%      | 79.26%     | 77.46%     | 45.20%               | 76.03%               |

## Citation

This repository is adapted from TMLU, which is a benchmark for evaluating Taiwanese Mandarin language understanding. 

```
@article{DBLP:journals/corr/abs-2403-20180,
  author       = {Po{-}Heng Chen and
                  Sijia Cheng and
                  Wei{-}Lin Chen and
                  Yen{-}Ting Lin and
                  Yun{-}Nung Chen},
  title        = {Measuring Taiwanese Mandarin Language Understanding},
  journal      = {CoRR},
  volume       = {abs/2403.20180},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.20180},
  doi          = {10.48550/ARXIV.2403.20180},
  eprinttype    = {arXiv},
  eprint       = {2403.20180},
  timestamp    = {Wed, 10 Apr 2024 17:37:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-20180.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
