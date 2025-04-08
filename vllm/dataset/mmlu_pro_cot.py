from typing import Optional, Tuple, List, Dict
from datasets import load_dataset
import re
import string
import collections
import json
import pathlib

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams


class MMLUProCoTQuestion:
    def __init__(
        self,
        category: str,
        prompt: str,
        answer: str,
        max_tokens: int,
    ) -> None:
        self.category = category
        self.prompt = prompt
        self.answer = answer
        self.max_tokens = max_tokens
        self.result = None
        
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (
            self.prompt,
            SamplingParams(n=1, 
                           temperature=0.0, 
                           max_tokens=self.max_tokens, 
                           stop=["</s>", "Q:", "<|im_end|>", "\nQ:"]),
        )

    def get_scores(self) -> bool:
        return self.result
    
    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        self.result = output.outputs[0].text
        return _is_correct(self.result, self.answer)
    
    def __repr__(self) -> str:
        if self.result is None:
            return (
                f"*** Prompt: {self.prompt}\n"
                f"*** Answer: {self.answer}\n"
            )
        else:
            return (
                f"*** Prompt: {self.prompt}\n"
                f"*** Answer: {self.answer}\n"
                f"*** Result: {self.result}\n"
            )

def format_cot_example(example, choices, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

class MMLUProCoTDataset(LLMDataset):    
    def __init__(
        self, 
        max_tokens: int = 1024,
        sample_percent: int = 100
    ):
        super().__init__(LLMDatasetType.COT_QA)
        self.dataset = load_dataset('TIGER-Lab/MMLU-Pro')
        self.sample_percent = sample_percent
        assert self.sample_percent > 0 and self.sample_percent <= 100
        
        self.valid_set = self.dataset['validation']
        self.test_set = self.dataset['test']
        if self.sample_percent < 100:
            self.test_set = self.test_set.select(
                    range(0, len(self.test_set), 100 // self.sample_percent))
        
        self.max_tokens = max_tokens
        self.choices = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
        ]
        
        self.categories = [
            "biology",
            "business",
            "chemistry",
            "computer science",
            "economics",
            "engineering",
            "health",
            "history",
            "law",
            "math",
            "other",
            "philosophy",
            "physics",
            "psychology",
        ]
        
        # create few shot prompts
        self.few_shot_examples = {c: [] for c in self.categories}
        for example in self.valid_set:
            c = example["category"]
            self.few_shot_examples[c].append(example)
        
        self.few_shot_prompts = {}
        for c in self.categories:
            self.few_shot_prompts[c] = ''
            for example in self.few_shot_examples[c]:
                self.few_shot_prompts[c] += format_cot_example(example, self.choices)
        
        self.request_to_question: Dict[str, MMLUProCoTQuestion] = {}
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
    
    def register_request(
        self,
        question: MMLUProCoTQuestion,
        request_id: str,
    ):
        self.request_to_question[request_id] = question
    
    def complete_request(self, output: RequestOutput) -> None:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        # print('-------------------')
        if question.update_request_output(output):
            self.num_correct += 1
        # print(question)
        # print('-------------------')
        
        self.num_total += 1
    
    def get_scores_str(self) -> str:
        return f"Accuracy: {round(self.num_correct / self.num_total * 100, 2)}%, {self.num_correct}/{self.num_total}"

    def get_scores(self) -> Tuple[int, int]:
        return self.num_correct, self.num_total
    
    def sample(
        self, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MMLUProCoTQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        dataset = self.test_set
        # sample row indices
        if not continued:
            sample_start_pos = 0
        else:
            assert n is not None, 'Number of samples required for continued sampling'
            sample_start_pos = self.data_ptr
            assert sample_start_pos >= 0
            self.data_ptr += n
        if not indices:
            indices = self.index_sampler.sample(n, is_random, len(dataset), sample_start_pos)
            
        # make up questions
        questions = []   
        for index in indices:
            item = dataset[index]
            category = item['category']
            
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt = self.few_shot_prompts[category] + format_cot_example(
                item, self.choices, including_answer=False)
            answer = item['answer']            
            # print(prompt)
            questions.append(MMLUProCoTQuestion(category, prompt, answer, self.max_tokens))
        return questions


def _extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        # print("1st answer extract failed\n" + text)
        return _extract_again(text)

def _extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return None
        # return _extract_final(text)

def _extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def _is_correct(model_completion: str, answer: str):
    match = _extract_answer(model_completion)
    # print(f"*********** gt_answer:{answer} | result:{match} ***********")
    return match == answer