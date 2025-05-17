import jiwer
from opencc import OpenCC
from typing import List
import re
import argparse
import json
import pandas as pd

def clean_transcription(text, language="zh"):
    """
    Simple function to extract transcription after ":" or "：" and remove punctuation
    """
    # Find text after ":" or "："
    if "：" in text:
        text = text.split("：", 1)[1].strip()
    elif ":" in text:
        text = text.split(":", 1)[1].strip()

    # Remove punctuation
    punctuation = r"""！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏.!?,;:"""
    translator = str.maketrans('', '', punctuation)
    text = text.translate(translator)

    # Remove spaces for Chinese
    if language == "zh":
        text = re.sub(r'\s+', '', text)

    return text

def calculate_cer(references: List[str], hypotheses: List[str]):

    transform = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars()
    ])
    
    cer = jiwer.cer(
        references,
        hypotheses,
        truth_transform=transform,
        hypothesis_transform=transform
    )

    return cer

def calculate_cer_s2t(references: List[str], hypotheses: List[str]):
    cc = OpenCC('s2t')
    clean_references = [clean_transcription(ref) for ref in references]
    clean_hypotheses = [clean_transcription(hyp) for hyp in hypotheses]
    converted_hypotheses = [cc.convert(hyp) for hyp in clean_hypotheses]

    json_data = [
        {
            "reference": ref,
            "hypothesis": hyp,
        } for ref, hyp in zip(clean_references, clean_hypotheses)
    ]
    with open("tmp.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    # Simplified Chinese vs. Traditional Chinese
    original_cer = calculate_cer(clean_references, clean_hypotheses)
    converted_cer = calculate_cer(clean_references, converted_hypotheses)

    return original_cer, converted_cer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--reference", type=str)
    args = parser.parse_args()
    with open(args.input_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    ref_data = pd.read_csv(args.reference)
    
    references = []
    hypotheses = []
    for result in json_data:
        ref = ref_data.loc[ref_data['file'] == result['file'], 'sentence'].values[0]
        hyp = result['response']
        references.append(ref)
        hypotheses.append(hyp)

    orig_cer, s2t_cer = calculate_cer_s2t(references, hypotheses)
    print("Original CER:", orig_cer)
    print("Force S2T (Simplified-to-Traditional) CER:", s2t_cer)       
