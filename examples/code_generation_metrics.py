import json
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer
from datasets import load_dataset
from typing import Dict, List

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

humaneval_result: Dict[str, List]=  {}
mbpp_result: Dict[str, List] = {}
mathqa_dataset: Dict[str, List] = {}

def get_length_of_tokens(dataset_name: str, tokenizer: LlamaTokenizer,label: str = None):
    # Code to get code metrics
    prompts = []
    truths = []

    if dataset_name == "humaneval":
        assert label == "test"
        dataset = load_dataset("openai_humaneval")[label]
        for row in dataset:
            prompts.append(row["prompt"])
            truths.append(row["canonical_solution"])
    elif dataset_name == "mbpp":
        assert label in ["train", "validation", "test", "prompt"]
        dataset = load_dataset("mbpp")[label]
        for row in dataset:
            prompts.append(row["text"])
            truths.append(row["code"])
        
        MATHQAPYTHON_PROMPT = "Here are 3 examples:\nHere is your task: {} Your code should calculate the answer of the task.\n[BEGIN]\n{}\n[DONE]\n"
        extra_prompt = ""
        prompt_data = load_dataset("mbpp")["prompt"]
        for i in range(3):
            row = prompt_data[i]
            extra_prompt += MATHQAPYTHON_PROMPT.format(row["text"], row["code"]) + '\n'
        extra_prompt = "\'\'\'\n\n" + extra_prompt + "\n Give the code for last task to fill between [BEGIN] and [DONE] \n"+ "\n\'\'\'\n\n"
        extra_len = len(tokenizer(extra_prompt)["input_ids"])
        print(f"Extra mbpp prompt length: {extra_len} ")
        print(extra_prompt)
    elif dataset_name == "mathqa":
        # mathqa_python dataset
        with open("/home/zhaorunyuan/mathqa/mathqa-python.json", "r") as f:
            dataset = json.load(f)
        for row in dataset:
            prompts.append(row["text"])
            truths.append(row["code"])
    else:
        return NotImplementedError()
    
    prompts_lens = []
    truths_lens = []
    for prompt in prompts:
        tokenized_prompt = tokenizer(prompt)
        prompts_lens.append(len(tokenized_prompt["input_ids"]))
    for truth in truths:
        tokenized_truth = tokenizer(truth)
        truths_lens.append(len(tokenized_truth["input_ids"]))
    
    if label is not None:
        prompt_title = f"{dataset_name}_{label}_prompt"
    else:
        prompt_title = f"{dataset_name}_prompt"
    plt.hist(prompts_lens, bins=10, edgecolor='black')  # 设置bins来控制直方图的柱子数量
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.title(prompt_title)
    plt.grid(True)

    plt.savefig(f'{prompt_title}.png')

    plt.clf()

    if label is not None:
        truth_title = f"{dataset_name}_{label}_truth"
    else:
        truth_title = f"{dataset_name}_truth"
    plt.hist(truths_lens, bins=10, edgecolor='black')  # 设置bins来控制直方图的柱子数量
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.title(truth_title)
    plt.grid(True)

    plt.savefig(f'{truth_title}.png')

    plt.clf()

    print(f"Prompts: {dataset_name} {label} mean length: {sum(prompts_lens) / len(prompts_lens)}")
    print(f"Truths: {dataset_name} {label} mean length: {sum(truths_lens) / len(truths_lens)}")

if __name__ == "__main__":
    get_length_of_tokens("humaneval", tokenizer, label="test")
    get_length_of_tokens("mbpp", tokenizer, label="train")
    get_length_of_tokens("mbpp", tokenizer, label="validation")
    get_length_of_tokens("mbpp", tokenizer, label="test")
    get_length_of_tokens("mathqa", tokenizer)
