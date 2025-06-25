import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from utils.data_loader import load_data
from utils.parser import parse_question, parse_ground_truth, extract_answer
from utils.grader import check_is_correct
from math import comb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help="model path")
    parser.add_argument('--split', type=str, default="test", help="data split to evaluate on")
    # parser.add_argument('--k', type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument('--output_file', type=str, default="results.jsonl", help="output file path")
    parser.add_argument('--use_flash_attn', action='store_true', help="whether to use flash attention")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(0)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Configure model loading with Flash Attention if requested
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if args.use_flash_attn:
        model_kwargs.update({
            "use_flash_attention_2": True,
            "attn_implementation": "flash_attention_2"
        })

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )

    # Load data
    examples = load_data("aime", args.split, data_dir="./AIME-Preview/eval/data")
    
    # Prepare prompt template
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    results = []
    correct_count = 0
    inputs = []

    for example in examples:
        question = parse_question(example, "aime")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        inputs.append(messages)

    
    for i in range(len(examples)):
        # Get question and ground truth
        example = examples[i]
        gt_cot, gt_ans = parse_ground_truth(example, "aime")
        
        # Prepare prompt
        question = parse_question(example, "aime")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=32768,
                # max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for faster generation
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        print(generated_text)
        
        # Extract and check answer
        generated_answer = extract_answer(generated_text, "aime")
        is_correct = check_is_correct(generated_answer, gt_ans)
        
        if is_correct:
            correct_count += 1
        
        # Save result
        result = {
            "question": question,
            "prompt": prompt,
            "generated_response": generated_text,
            "generated_answer": generated_answer,
            "gold_answer": gt_ans,
            "is_correct": is_correct
        }
        results.append(result)

    # Calculate metrics
    accuracy = correct_count / len(examples)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()

