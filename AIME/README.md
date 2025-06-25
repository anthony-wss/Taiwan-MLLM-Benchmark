# Benchmark Name

This benchmark focuses on evaluating models' **math reasoning abilities**.
The dataset is organized by year. Each year subset contains **15 or 30 math problems**.

## Installation

To install dependencies:

```bash
pip install -r requirements.txt
```

## Baselines

- `Gemma-3-27B-it`: `google/gemma-3-27b-it`
- `Earica-omni`: `voidful/earica-omni-27b`
- `Earica-text`: `Allen172/gemma-text-7700`

## Usage

1. Clone the AIME corpus from [this repo](https://github.com/anthony-wss/AIME-Preview.git)

```bash
git clone https://github.com/anthony-wss/AIME-Preview.git
```

2. Run the command to get model performance(accuracy)

```bash
python aime_eval.py \
    --model_name_or_path google/gemma-3-27b-it \
    --split test2024 \
    --output_file result_gemma_3_27b_it.json \
    --use_flash_attn
```

## Results

> Add results in the table below. Ensure they match exactly with our Google sheets.

|           | model A | model B | model C |
| --------- | ------- | ------- | ------- |
| AIME 2024 | 98.5%   | 41.5%   | 71.5%   |
| AIME 2025-I | 76.5%   | 32.5%   | 68.5%   |
| AIME 2025-II | 76.5%   | 32.5%   | 68.5%   |

