# Common Voice 16

This benchmark contains ASR (Automatic Speech Recognition) tasks for the Traditional Chinese subset, test split of the Common Voice 16 dataset.

Repository: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0

You can run `load_cv16.py` to load the dataset.

## Installation

We recommend using `uv` for fast dependency management:

```bash
uv pip venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

For `microsoft/Phi-4-multimodal-instruct` **ONLY**, please install flash attention and downgrade transformers:
```bash
uv pip uninstall transformers
uv pip install transformers==4.48.2
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
``` 

## Baselines

- `Qwen2.5-Omni-7B`: `Qwen/Qwen2.5-Omni-7B`
- `Qwen2-Audio-7B-IT`: `Qwen/Qwen2-Audio-7B-Instruct`
- `Qwen-Audio-Chat`: `Qwen/Qwen-Audio-Chat`
- `Phi-4-MM`: `microsoft/Phi-4-multimodal-instruct`

## Usage

### Part 1: Responses Generation

For models not listed above, please follow the format of `template.py` and check the official repository of your model to implement the `inference` function.

The prompts are adapted from examples in the [Qwen 2.5 Omni cookbook](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb).

Results will be saved to `./results/cv16_<your_model_name_for_file>.json`.

#### Qwen 2.5 Omni

We use the `transformers` library to run Qwen 2.5 Omni.
 
Our implementation closely follows the [Qwen 2.5 Omni cookbook](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb), with the logic implemented in `qwen2_5_omni.py`.

To run the inference:

```bash
python qwen2_5_omni.py
```

#### Qwen Audio Chat

We use the `transformers` library to run Qwen Audio Chat.

Our implementation follows the [Qwen-Audio-Chat Official Huggingface Repo](https://huggingface.co/Qwen/Qwen-Audio-Chat), with the logic implemented in `qwen_audio_chat.py`.

To run the inference:

```bash
python qwen_audio_chat.py
```

#### Qwen 2 Audio Instruct

We use the `transformers` library to run Qwen 2 Audio Instruct.

Our implementation follows the [Qwen2-Audio-7B-Instruct Official Huggingface Repo](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct), with the logic implemented in `qwen2_audio_it.py`.

To run the inference:

```bash
python qwen2_audio_it.py
```

#### Phi 4 Multimodal Instruct

We use the `transformers` library to run Phi 4 Multimodal Instruct. Note that the transformers version should be 4.48.2 for this model.

Our implementation follows the [Phi-4-multimodal-instruct Official Huggingface Repo](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), with the logic implemented in `phi4_mm_it.py`.

To run the inference:

```bash
python phi4_mm_it.py
```

### Part 2: Calculate CER (Character Error Rate)

In this part, two types of CER are calculated:
- **Original CER**: CER with the cleaned responses.
- **Force-S2T CER**: CER with the cleaned responses, **AND** all characters converted to traditional Chinese using the `OpenCC` package.

To calculate these two CERs, run the following script:

```bash
python3 cer.py --input_json <your_inference_result>.json \
--reference <your_dataset>/metadata.csv
```

Please ensure the `file` column in your prediction file and metadata are aligned.

## Results

| CER type   | Qwen2.5-Omni-7B | Qwen2-Audio-7B-IT | Qwen-Audio-Chat | Phi-4-MM |
| ---------- | --------------- | ----------------- | --------------- | -------- |
| Original   | 9.00%           | 23.01%            | 23.25%          | 33.89%   |
| Force-S2T  | 5.97%           | 11.70%            | 10.43%          | 9.37%    |