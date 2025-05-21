import torch
import os
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, GenerationConfig
from tqdm import tqdm
import json
import pandas as pd
import librosa

MAX_SAMPLE = -1
CV16_DATA_DIR = "data_cv16"
MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
model_name_for_file = "qwen2_audio_7b_it"

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
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
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
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    for i in tqdm(range(max_sample)):

        sample = df.iloc[i]
        audio_file = sample["file"]
        audio_path = f"{CV16_DATA_DIR}/audio/{audio_file}"
        audio, sr = librosa.load(audio_path, sr=16000)

        with torch.no_grad():

            inputs = processor(text=prompt, audio=[audio], sampling_rate=sr, return_tensors='pt').to('cuda:0')

            generate_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None
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