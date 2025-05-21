import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import pandas as pd
import librosa

MAX_SAMPLE = -1
CV16_DATA_DIR = "data_cv16"
MODEL_PATH = "Qwen/Qwen-Audio-Chat"
model_name_for_file = "qwen_audio_chat"

# Define prompt structure
system_prompt = 'You are a speech recognition model.'
speech_prompt = "請將這段繁體中文語音轉換為純文本，並且去掉標點符號。"

conversation = [
    {'role': 'system', "content": [{"type": "text", "text": system_prompt}]}, 
    {"role": "user", "content": [
        {"type": "audio", "audio": "input.wav"},
        {"type": "text", "text": speech_prompt},
    ]}
]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="cuda", 
        trust_remote_code=True, 
        bf16=True
    )

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
        query = tokenizer.from_list_format([
            {'audio': audio_path}, # Either a local path or an url
            {'text': speech_prompt},
        ])
        
        with torch.no_grad():

            response, _ = model.chat(
                tokenizer, 
                query=query, 
                system=system_prompt, 
                history=None, 
                do_sample=False,
                top_p=None,
                top_k=None
            )
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