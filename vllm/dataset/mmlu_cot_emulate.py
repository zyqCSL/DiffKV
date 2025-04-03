from typing import Optional, Tuple, List, Dict, Union
from datasets import load_dataset
import re
import string
import collections
import json
import pathlib

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)


class MMLUCoTEmulateQuestion:
    def __init__(
        self,
        prompt: str,
        answer: str,
        prefill_token_ids: List[int],   # token ids for prefill
        truth_token_ids: List[int],     # full tokens in the prompt
        max_tokens: int,
    ) -> None:
        self.prompt = prompt
        self.answer = answer
        self.prefill_token_ids = prefill_token_ids
        self.truth_token_ids = truth_token_ids
        self.max_tokens = max_tokens
        self.result = None
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (
            # self.prompt,
            self.prefill_token_ids,
            SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, 
                           emulate_seq=True, truth_token_ids=self.truth_token_ids,
                           stop=['\n']),
        )
    
    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        # self.result = output.outputs[0].text
        self.result = output.outputs[0].emulated_text
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

PROMPTS_JSON = pathlib.Path(__file__).parent.resolve() / '_mmlu_cot_prompts.json'

def get_mmlu_subsets() -> List[str]:
    with open(PROMPTS_JSON, 'r') as f:
        prompts = json.load(f)
    return list(prompts.keys())

class MMLUCoTEmulateDataset(LLMDataset):
    def __init__(self, 
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 subset: str = 'all', 
                 prefill_tokens: int = 64, 
                 max_tokens: int = 512):
        super().__init__(LLMDatasetType.COT_QA)
        self.dataset = load_dataset('cais/mmlu', subset)
        self.subset_label = subset
        self.max_tokens = max_tokens
        self.labels = ['A', 'B', 'C', 'D']
        with open(PROMPTS_JSON, 'r') as f:
            self.few_shot_prompts = json.load(f)
        
        # emulation config
        self.tokenizer = tokenizer
        self.prefill_tokens = prefill_tokens
        
        # runtime stats
        self.request_to_question: Dict[str, MMLUCoTEmulateQuestion] = {}
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
        # total number of attended tokens across all steps of all seqs
        self.num_attended_tokens = 0
    
    def register_request(
        self,
        question: MMLUCoTEmulateQuestion,
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
        
        # log sparsity
        prompt_len = len(output.prompt_token_ids)
        assert len(output.outputs) == 1
        for out in output.outputs:
            seq_len = len(out.token_ids) + prompt_len
            self.num_attended_tokens += (seq_len + prompt_len) * (seq_len - prompt_len + 1) / 2
            # print(f'request_id {request_id} theoretically attended tokens = ',
            #       (seq_len + prompt_len) * (seq_len - prompt_len + 1) / 2
            #       )
    
    def get_scores_str(self) -> str:
        return f"Accuracy: {round(self.num_correct / self.num_total * 100, 2)}%, {self.num_correct}/{self.num_total}"

    def get_scores(self) -> Tuple[int, int]:
        return self.num_correct, self.num_total
    
    def get_num_attended_tokens(self) -> int:
        return self.num_attended_tokens
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MMLUCoTEmulateQuestion]:
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
            
            # ground truth token ids for prompt
            truth_token_ids = self.tokenizer.encode(prompt)
            prefill_token_ids = truth_token_ids[:self.prefill_tokens]
            
            questions.append(MMLUCoTEmulateQuestion(
                prompt, answer, 
                prefill_token_ids, truth_token_ids,
                self.max_tokens))
        return questions


_MMLU_COT_RE = re.compile("(?<=The answer is )(.*)(?=.)")
_MMLU_COT_FALLBACK_RE = re.compile(":[\s]*(A|B|C|D)")

def _find_match(regex: re.Pattern, resp: str, select_index: int = 0) -> str:
    match = regex.findall(resp)
    if match:
        match = match[select_index]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        match = match.replace('(', '').replace(')', '')
        match = match.strip()
    return match


def _is_correct(model_completion: str, answer: str):
    match = _find_match(_MMLU_COT_RE, model_completion)
    if not match:
        match = _find_match(_MMLU_COT_FALLBACK_RE, model_completion)
    # print(f"*********** gt_answer:{answer} | result:{match} ***********")
    return match == answer