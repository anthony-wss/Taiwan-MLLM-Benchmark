import torch
import os
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig 
from tqdm import tqdm
import json
import pandas as pd
import librosa

# if you do not use Ampere or later GPUs, turn this to False
USE_FLASH_ATTENTION = True

MAX_SAMPLE = -1
MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
model_name_for_file = "phi4_mm_it"
CV16_DATA_DIR = "data_cv16"
attn_impl = "flash_attention_2" if USE_FLASH_ATTENTION else "eager"


# Define prompt structure
system_prompt = '<|system|>You are a speech recognition model.<|end|>'
user_prefix = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
speech_prompt = "請將這段繁體中文語音轉換為純文本，並且去掉標點符號。"
prompt = f'{system_prompt}{user_prefix}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

if __name__ == "__main__":

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation=attn_impl
    ).cuda()
    generation_config = GenerationConfig.from_pretrained(MODEL_PATH)

    # Load dataset
    df = pd.read_csv(os.path.join(CV16_DATA_DIR, "metadata.csv"))
    df.head()

    ids = []
    hypotheses = []
    references = []

    if MAX_SAMPLE == -1:
        max_sample = len(df)
    else:
        max_sample = MAX_SAMPLE
        
    model.eval()
    
    for i in tqdm(range(max_sample)):
        
        sample = df.iloc[i]
        audio_file = sample["file"]
        audio_path = f"{CV16_DATA_DIR}/audio/{audio_file}"
        audio, sr = librosa.load(audio_path)
        
        with torch.no_grad():

            inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors='pt').to('cuda:0')

            generate_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                generation_config=generation_config,
            )

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            hypotheses.append(response)
            ids.append(audio_file)

    # Save results as json
    results = []
    for id, hyp in zip(ids, hypotheses):
        result = {
            "file": id,
            "response": hyp
        }
        results.append(result)

    os.makedirs("results", exist_ok=True)

    with open(f"results/cv16_{model_name_for_file}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)