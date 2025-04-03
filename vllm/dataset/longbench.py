from typing import Optional, Tuple, List, Dict
from datasets import load_dataset, concatenate_datasets

import json
import pathlib
import numpy as np

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams

from vllm.dataset.metrics_longbench import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

SUBSETS_EVEN = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
              "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

SUBSETS_ALL = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", 
               "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
               "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

SUBSET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

class LongBenchQuestion:
    def __init__(
        self,
        dataset: str,
        prompt: str,
        answer: str,
        max_tokens: int,
        all_classess: Optional[List[str]],
        stop: List[str] = [],
    ) -> None:
        self.dataset = dataset
        self.prompt = prompt
        self.answer = answer
        self.max_tokens = max_tokens
        self.all_classes = all_classess
        self.stop = stop
        self.result = None
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (
            self.prompt,
            SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, stop=self.stop),
        )
    
    def update_request_output(self, output: RequestOutput) -> float:
        assert output.finished
        assert len(output.outputs) == 1
        self.result = output.outputs[0].text
        return _get_score(self.dataset, self.result, self.answer, self.all_classes)
    
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

PROMPTS_JSON = pathlib.Path(__file__).parent.resolve() / '_longbench_prompts.json'
MAX_LEN_JSON = pathlib.Path(__file__).parent.resolve() / '_longbench_max_len.json'

class LongBenchDataset(LLMDataset):
    def __init__(self, 
                 subset: str = 'all', 
                 use_even_dist: bool = False,
                 sample_rate: int = 100):
        super().__init__(LLMDatasetType.COT_QA)
        
        self.subset_label = subset
        self.dataset = None
        self.use_even_dist = use_even_dist
        self.sample_rate = sample_rate
        assert self.sample_rate > 0 and self.sample_rate <= 100
        
        if self.subset_label == 'all':
            all_datasets = []
            if use_even_dist:
                all_subsets = SUBSETS_EVEN
            else:
                all_subsets = SUBSETS_ALL
            for subset in all_subsets:
                _dataset = load_dataset('THUDM/LongBench', subset)['test']
                if self.sample_rate < 100:
                    _dataset = _dataset.select(
                        range(0, len(_dataset), 100 // self.sample_rate))
                all_datasets.append(_dataset)
            self.dataset = concatenate_datasets(all_datasets)
        else:
            assert self.subset_label in SUBSETS_ALL
            self.dataset = load_dataset('THUDM/LongBench', self.subset_label)['test']
            if self.sample < 100:
                self.dataset = self.dataset.select(
                    range(0, len(self.dataset), 100 // self.sample_rate))
        assert self.dataset is not None
        
        with open(PROMPTS_JSON, 'r') as f:
            self.prompt_formats = json.load(f)
        with open(MAX_LEN_JSON, 'r') as f:
            self.subset_to_maxlens = json.load(f)
        
        # sampling stats
        self.request_to_question: Dict[str, LongBenchQuestion] = {}
        self.data_ptr = 0
        
        # accuracy stats
        self.cum_score = 0
        self.num_total = 0
        # for use_even_dist
        self.scores_breakdown = {
            '0-4k': [],
            '4-8k': [],
            '8k+': [],
        }
        
    def register_request(
        self,
        question: LongBenchQuestion,
        request_id: str,
    ):
        self.request_to_question[request_id] = question
    
    def complete_request(self, output: RequestOutput) -> None:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        # print('-------------------')
        _score = question.update_request_output(output)
        self.cum_score += _score
        seq_len = len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
        if seq_len < 4000:
            self.scores_breakdown['0-4k'].append(_score)
        elif seq_len < 8000:
            self.scores_breakdown['4-8k'].append(_score)
        else:
            self.scores_breakdown['8k+'].append(_score)
        # print(question)
        # print('-------------------')
        
        self.num_total += 1
    
    def get_scores_str(self) -> str:
        return f"Score: {round(self.cum_score / self.num_total * 100, 2)}%"

    def get_scores(self) -> Tuple[int, int]:
        return self.cum_score, self.num_total

    def get_scores_breakdown(self) -> Dict[str, float]:
        assert self.use_even_dist
        scores = {}
        for dist in self.scores_breakdown:
            scores[dist] = np.mean(self.scores_breakdown[dist]) * 100
        return scores
    
    def sample(
        self, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[LongBenchQuestion]:
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
            # raw data
            item = dataset[index]
            # features
            subset = item['dataset']
            stop = []
            if subset == 'samsum':
                stop = ['\n']
            # print(prompt)
            questions.append(LongBenchQuestion(
                dataset=subset, 
                prompt=self.prompt_formats[subset].format(**item),
                answer=item['answers'], 
                max_tokens=self.subset_to_maxlens[subset], 
                all_classess=item['all_classes'],
                stop=stop))
            
        return questions

#------------- LongBench score -------------#
def _get_score(
    dataset: str, 
    result: str, 
    answers: List[str], 
    all_classes: Optional[List[str]],
) -> float:
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        result = result.lstrip('\n').split('\n')[0]
    score = 0.0
    for answer in answers:
        score = max(score, 
                    SUBSET_TO_METRIC[dataset](result, answer, all_classes=all_classes))
    return score