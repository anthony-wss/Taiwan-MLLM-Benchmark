import os
import soundfile as sf
from tqdm import tqdm
import csv
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_16_0", "zh-TW", split="test", trust_remote_code=True)

DATASET_DIR = "data_cv16"

os.makedirs(os.path.join(DATASET_DIR, "audio"), exist_ok=True)
csv_data = [['file', 'sentence']]

for sample in tqdm(dataset):
    filename = f"{os.path.basename(sample['path']).split('.')[0]}.wav"
    filepath = os.path.join(DATASET_DIR, "audio", filename)
    sf.write(filepath, sample["audio"]["array"], sample["audio"]["sampling_rate"])
    csv_entry = [filename, sample['sentence']]
    csv_data.append(csv_entry)

    
with open(os.path.join(DATASET_DIR, "metadata.csv"), "+w", encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)