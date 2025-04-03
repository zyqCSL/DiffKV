from typing import Optional, Tuple, List, Dict, Union
from datasets import load_dataset, concatenate_datasets

import re
import pathlib
import numpy as np
import enum

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)


class LongBenchV2CoTStatus(enum.Enum):
    """Status of a longbench-v2 cot question."""
    COT = enum.auto()
    EXTRACTION = enum.auto()

# With COT, each LongBench question is processed two times
# The 1st round generates the COT contents, and the 2nd round formats the COT contents to answers
class LongBenchV2EmulateQuestion:
    def __init__(
        self,
        index: int,
        _id: str,
        domain: str,
        sub_domain: str,
        difficulty: str,
        status: LongBenchV2CoTStatus,
        prompt: str,
        answer: str,
        prefill_token_ids: List[int],   # token ids for prefill
        truth_token_ids: List[int],     # full tokens in the prompt
        max_tokens: int,
        stop: List[str] = [],
        cot_text: Optional[str] = None,
    ) -> None:
        self.index = index
        self._id = _id
        self.domain = domain
        self.sub_domain = sub_domain
        self.difficulty = difficulty
        
        self.status = status
        self.prompt = prompt
        self.cot_text = cot_text    # valid for extraction phase
        self.answer = answer
        self.prefill_token_ids = prefill_token_ids
        self.truth_token_ids = truth_token_ids
        self.max_tokens = max_tokens
        self.stop = stop
        self.result = None
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (
            # self.prompt,
            self.prefill_token_ids,
            SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, 
                           emulate_seq=True, truth_token_ids=self.truth_token_ids,
                           stop=self.stop),
        )
    
    def update_request_output(self, output: RequestOutput) -> bool:
        assert self.status == LongBenchV2CoTStatus.EXTRACTION
        
        assert output.finished
        assert len(output.outputs) == 1
        # self.result = output.outputs[0].text
        
        self.result = output.outputs[0].emulated_text
        assert self.result is not None
        
        # print(f'Emulated longbench response: {self.result}')
        
        return _is_correct(self.result, self.answer)
    
    def __repr__(self) -> str:
        x = (f"*** _id: {self._id}, index: {self.index}\n"
             f"*** Domain: {self.domain}\n"
             f"*** Difficulty: {self.difficulty}\n"
             f"*** Status: {self.status}\n"
             f"*** Prompt: {self.prompt}\n"
             f"*** Answer: {self.answer}\n")
        
        if self.cot_text is not None:
            x += f"*** COT: {self.cot_text}\n"
        if self.result is not None:
            x += f"*** Result: {self.result}\n"
        return x


PROMPTS_DIR = pathlib.Path(__file__).parent.resolve() / 'longbench_v2_prompts'
COT_PROMPT_TEMPLATE = PROMPTS_DIR / '0shot_cot.txt'
COT_ANS_PROMPT_TEMPLATE = PROMPTS_DIR / '0shot_cot_ans.txt'


class LongBenchV2EmulateDataset(LLMDataset):
    def __init__(self, 
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 split: str = 'train', 
                 prefill_tokens: int = 64, 
                 max_model_len: Optional[int] = None):
        super().__init__(LLMDatasetType.COT_QA)
        
        self.split_label = split
        self.dataset = load_dataset('THUDM/LongBench-v2', split=split)
        
        self.cot_format = open(COT_PROMPT_TEMPLATE, encoding='utf-8').read()
        self.extract_format = open(COT_ANS_PROMPT_TEMPLATE, encoding='utf-8').read()
        # max generation len for different phases
        self.max_model_len = max_model_len
        self.cot_maxlen = 1024
        # self.cot_maxlen = 4096
        self.extraction_maxlen = 128
        self.stop = []
        
        # emulation config
        self.tokenizer = tokenizer
        self.prefill_tokens = prefill_tokens
        
        # sampling stats
        self.request_to_question: Dict[str, LongBenchV2EmulateQuestion] = {}
        self.data_ptr = 0
        # total number of attended tokens across all steps of all seqs WITHOUT sparsity
        self.num_attended_tokens = 0
        
        # only for questions fully completed (extraction is done)
        # accuracy stats
        self.num_correct = 0
        self.num_total = 0
        # breakdown w.r.t difficulty
        self.num_correct_difficulty = {}
        self.num_total_difficulty = {}
        # breakdown w.r.t domain
        self.num_correct_domain = {}
        self.num_total_domain = {}
        
    def register_request(
        self,
        question: LongBenchV2EmulateQuestion,
        request_id: str,
    ):
        self.request_to_question[request_id] = question
    
    def __make_extraction_question(
        self, 
        cot_question: LongBenchV2EmulateQuestion, 
        output: RequestOutput,
    ) -> LongBenchV2EmulateQuestion:
        assert cot_question.status == LongBenchV2CoTStatus.COT        
        assert output.finished
        assert len(output.outputs) == 1
        # self.result = output.outputs[0].text
        
        # print(f'LongBench make_extraction_request: {cot_question}')
        
        cot_text = output.outputs[0].emulated_text
        assert cot_text is not None
        
        item = self.dataset[cot_question.index]
        extraction_prompt = self.extract_format.replace('$DOC$', item['context'].strip())
        extraction_prompt = extraction_prompt.replace('$Q$', item['question'].strip())
        extraction_prompt = extraction_prompt.replace('$C_A$', item['choice_A'].strip())
        extraction_prompt = extraction_prompt.replace('$C_B$', item['choice_B'].strip())
        extraction_prompt = extraction_prompt.replace('$C_C$', item['choice_C'].strip())
        extraction_prompt = extraction_prompt.replace('$C_D$', item['choice_D'].strip())
        extraction_prompt = extraction_prompt.replace('$COT$', cot_text)
        
        # ground truth token ids for prompt
        truth_token_ids = self.tokenizer.encode(extraction_prompt)
        # print(f'[DEBUG] Original extraction prompt token len = {len(truth_token_ids)}')
        if self.max_model_len is not None and \
           len(truth_token_ids) + self.extraction_maxlen > self.max_model_len:
            max_len = self.max_model_len - self.extraction_maxlen
            assert max_len > 0
            truth_token_ids = truth_token_ids[:max_len//2] + truth_token_ids[-max_len//2:]
        # print(f'[DEBUG] Truncated extraction prompt token len = {len(truth_token_ids)}')
        
        prefill_token_ids = truth_token_ids[:self.prefill_tokens]
        
        extract_question = LongBenchV2EmulateQuestion(
            index=cot_question.index,
            _id=item['_id'],
            domain=item['domain'],
            sub_domain=item['sub_domain'],
            difficulty=item['difficulty'],
            status=LongBenchV2CoTStatus.EXTRACTION,
            prompt=extraction_prompt,
            answer=item['answer'],
            prefill_token_ids=prefill_token_ids,   # token ids for prefill
            truth_token_ids=truth_token_ids,       # full tokens in the prompt
            max_tokens=self.extraction_maxlen,
            stop=self.stop,
            cot_text=cot_text)

        # print('************** Extraction question ******************')
        # print(extract_question)
        # print(f"*** Emulated decoding texts: {output.outputs[0].emulated_text}\n")
        # print('********************************')
        
        return extract_question
    
    # TODO: handle question in CoT & extraction phases separately
    def complete_request(self, output: RequestOutput) -> Optional[LongBenchV2EmulateQuestion]:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        
        # print('************** Completed question ******************')
        # print(question)
        # print(f"*** Emulated decoding texts: {output.outputs[0].emulated_text}\n")
        # print('********************************')
        
        if question.status == LongBenchV2CoTStatus.COT:
            # NOTE: we should log spasrity for the CoT phase as well
            prompt_len = len(output.prompt_token_ids)
            assert len(output.outputs) == 1
            for out in output.outputs:
                seq_len = len(out.token_ids) + prompt_len
                self.num_attended_tokens += (seq_len + prompt_len) * (seq_len - prompt_len + 1) / 2
                # print(f'request_id {request_id} (CoT) theoretically attended tokens = ',
                #       (seq_len + prompt_len) * (seq_len - prompt_len + 1) / 2
                #       )
                
            # NOTE: we need to run the question again to extract the correct answer
            return self.__make_extraction_question(question, output)
        
        else:
            assert question.status == LongBenchV2CoTStatus.EXTRACTION
            # the final result should be contained in generated texts
            _score = question.update_request_output(output)
            # update stats
            self.num_correct += _score
            self.num_total += 1
            
            # difficulty breakdown
            _d = question.difficulty
            if _d not in self.num_correct_difficulty:
                self.num_correct_difficulty[_d] = 0
                self.num_total_difficulty[_d] = 0
            self.num_correct_difficulty[_d] += _score
            self.num_total_difficulty[_d] += 1
            
            # domain breakdown
            _d = question.domain
            if _d not in self.num_correct_domain:
                self.num_correct_domain[_d] = 0
                self.num_total_domain[_d] = 0
            self.num_correct_domain[_d] += _score
            self.num_total_domain[_d] += 1
            
            # log sparsity
            prompt_len = len(output.prompt_token_ids)
            assert len(output.outputs) == 1
            for out in output.outputs:
                seq_len = len(out.token_ids) + prompt_len
                self.num_attended_tokens += (seq_len + prompt_len) * (seq_len - prompt_len + 1) / 2
                # print(f'request_id {request_id} (extraction) theoretically attended tokens = ',
                #       (seq_len + prompt_len) * (seq_len - prompt_len + 1) / 2
                #       )
            return None
    
    def get_scores_str(self) -> str:
        return f"Score: {round(self.num_correct / self.num_total * 100, 2)}%"

    def get_scores(self) -> Tuple[int, int]:
        return self.num_correct, self.num_total

    def get_num_attended_tokens(self) -> int:
        return self.num_attended_tokens
    
    def get_scores_difficulty_breakdown(self) -> Dict[str, float]:
        scores = {} 
        for d in self.num_correct_difficulty:
            scores[d] = self.num_correct_difficulty[d] / self.num_total_difficulty[d]
        return scores

    def get_scores_domain_breakdown(self) -> Dict[str, Tuple[int, int]]:
        scores = {} 
        for d in self.num_correct_domain:
            scores[d] = (self.num_correct_domain[d], self.num_total_domain[d])
        return scores
    
    def sample(
        self, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[LongBenchV2EmulateQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        dataset = self.dataset
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
            # print(f'LongBench sample index {index}')
            
            # raw data
            item = dataset[index]
            prompt = self.cot_format.replace('$DOC$', item['context'].strip())
            prompt = prompt.replace('$Q$', item['question'].strip())
            prompt = prompt.replace('$C_A$', item['choice_A'].strip())
            prompt = prompt.replace('$C_B$', item['choice_B'].strip())
            prompt = prompt.replace('$C_C$', item['choice_C'].strip())
            prompt = prompt.replace('$C_D$', item['choice_D'].strip())
            
            # ground truth token ids for prompt
            truth_token_ids = self.tokenizer.encode(prompt)
            # print(f'[DEBUG] Original CoT prompt token len = {len(truth_token_ids)}')
            if self.max_model_len is not None and \
               len(truth_token_ids) + self.cot_maxlen > self.max_model_len:
                max_len = self.max_model_len - self.cot_maxlen
                assert max_len > 0
                truth_token_ids = truth_token_ids[:max_len // 2] + truth_token_ids[-max_len // 2:]
            # print(f'[DEBUG] Truncated CoT prompt token len = {len(truth_token_ids)}')
            
            prefill_token_ids = truth_token_ids[:self.prefill_tokens]
            
            # print(f'LongBench item {index} encoded, len = {len(truth_token_ids)}')
            
            # print(prompt)
            questions.append(LongBenchV2EmulateQuestion(
                index=index,
                _id=item['_id'],
                domain=item['domain'],
                sub_domain=item['sub_domain'],
                difficulty=item['difficulty'],
                status=LongBenchV2CoTStatus.COT,
                prompt=prompt,
                answer=item['answer'],
                prefill_token_ids=prefill_token_ids,   # token ids for prefill
                truth_token_ids=truth_token_ids,       # full tokens in the prompt
                max_tokens=self.cot_maxlen,
                stop=self.stop))

        return questions

#------------- LongBench-v2 correctness -------------#
def _extract_answer(result: str) -> str:
    result = result.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', result)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', result)
        if match:
            return match.group(1)
        else:
            return None


def _is_correct(
    result: str, 
    answer: str, 
) -> bool:
    return _extract_answer(result) == answer