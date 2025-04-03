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


class MMLUCoTQuestion:
    def __init__(
        self,
        prompt: str,
        answer: str,
        max_tokens: int,
    ) -> None:
        self.prompt = prompt
        self.answer = answer
        self.max_tokens = max_tokens
        self.result = None
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (
            self.prompt,
            SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, 
                           stop=['</s>', 'Q:', '\nQ:', '<|im_end|>']),
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
                # f"*** Prompt: {self.prompt}\n"
                f"*** Answer: {self.answer}\n"
            )
        else:
            return (
                # f"*** Prompt: {self.prompt}\n"
                f"*** Answer: {self.answer}\n"
                f"*** Result: {self.result}\n"
            )

PROMPTS_JSON = pathlib.Path(__file__).parent.resolve() / '_mmlu_cot_prompts.json'

def get_mmlu_subsets() -> List[str]:
    with open(PROMPTS_JSON, 'r') as f:
        prompts = json.load(f)
    return list(prompts.keys())

class MMLUCoTDataset(LLMDataset):
    def __init__(self, subset: str = 'all', max_tokens: int = 1024):
        super().__init__(LLMDatasetType.COT_QA)
        self.dataset = load_dataset('cais/mmlu', subset)
        self.subset_label = subset
        self.max_tokens = max_tokens
        self.labels = ['A', 'B', 'C', 'D']
        with open(PROMPTS_JSON, 'r') as f:
            self.few_shot_prompts = json.load(f)
        
        self.request_to_question: Dict[str, MMLUCoTQuestion] = {}
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
    
    def register_request(
        self,
        question: MMLUCoTQuestion,
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
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MMLUCoTQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        assert label in ['test', 'validation', 'dev', 'auxiliary_train']
        dataset = self.dataset[label]
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
            subject = item['subject']
            
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt =  self.few_shot_prompts[subject] + "Q: " + item['question'].strip() + "\n"            
            choices = ''
            for label, choice in zip(self.labels, item['choices']):
                choices += f'({label}): {choice} '
            prompt += choices.strip() + "\nA: Let's think step by step."
            answer = self.labels[item['answer']]            
            # print(prompt)
            questions.append(MMLUCoTQuestion(prompt, answer, self.max_tokens))
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