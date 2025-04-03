from typing import Optional, Dict, Tuple, List
from datasets import load_dataset
import evaluate

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams

import re
import string
import collections


class SquadQuestion:
    def __init__(
        self,
        uid: str,
        prompt: str,
        truths: List[str],
        max_tokens: int,
        metric: evaluate.EvaluationModule,
    ) -> None:
        self.uid = uid
        self.prompt = prompt
        self.truths = truths  
        assert len(self.truths) > 0
        self.max_tokens = max_tokens
        self.answer = None
        self.metric = metric

    def make_request(self) -> Tuple[str, SamplingParams]:
        return (
            self.prompt,
            SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, stop=['.', '\n', ';']),
        )

    def get_scores(self) -> Tuple[float, float]:
        pred = {
            'prediction_text': self.answer, 
            'id': self.uid, 
            'no_answer_probability': 0.,
        }
        scores = self.metric.compute(
            predictions=[pred],
            references=[self.truths])
        return scores['exact'], scores['f1'] / 100

    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        self.answer = output.outputs[0].text
        if self.answer is None:
            self.answer = ""
            print(f"Warning: empty answer for prompt: {self.prompt}")
        return self.get_scores()

    def __repr__(self) -> str:
        if self.answer:
            return (
                f"prompt: {self.prompt}\n"
                f"answer: {self.answer}\n"
                f"truth: {self.truths}\n"
                f"scores: {self.get_scores()}\n"
            )
        else:
            return (
                f"prompt = {self.prompt}; "
                f"truths = {self.truths}; "
                f"answer = {self.answer}"
            )


class SquadDataset(LLMDataset):
    def __init__(self):
        super().__init__(LLMDatasetType.SQUAD)
        # mapping between questions and requests
        self.request_to_question: Dict[str, SquadQuestion] = {}

        # score
        self.exact_scores = []
        self.f1_scores = []

    def register_request(
        self,
        question: SquadQuestion,
        request_id: str,
    ):
        self.request_to_question[request_id] = question

    def complete_request(self, output: RequestOutput):
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        if question.update_request_output(output):
            self.exact_scores.append(question.get_scores()[0])
            self.f1_scores.append(question.get_scores()[1])            
        # print('----------------')
        # print(question)
        # print('----------------')

    def get_scores(self) -> Tuple[float, float]:
        return (
            sum(self.exact_scores) / len(self.exact_scores),
            sum(self.f1_scores) / len(self.f1_scores),
        )

# https://github.com/meta-llama/llama/issues/867
# https://arxiv.org/pdf/2203.02155
GPT_SQUAD_PROMPT = "Answer each question using information in the preceding background paragraph. " \
                   "If there is not enough information provided, answer with “Not in background.”"

# SQUAD_PROMPT = "Answer the question using information in the preceding background paragraph. " \
#                "If there is not enough information provided, answer with “Not in background.”"

SQUAD_PROMPT = "Answer the question concisely using information in the preceding background paragraph. " \
               "If there is not enough information provided, answer with “Not in background.”"

class SquadDatasetV1(SquadDataset):
    def __init__(self, max_tokens: int=64):
        super().__init__()
        self.dataset = load_dataset("rajpurkar/squad")
        self.metric = evaluate.load("squad")
        self.max_tokens = max_tokens
        self.data_ptr = 0

    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[SquadQuestion]:
        assert label in ["train", "validation"]
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
            
        # create questions
        questions = []
        for idx in indices:
            row = dataset[idx]
            uid = row['id']
            prompt = SQUAD_PROMPT + '\n\n' + \
                     "Title: " + row["title"] + '\n' + \
                     "Background: " + row["context"] + '\n' + \
                     "Question: " + row["question"] + '\n' + \
                     "Answer: " 
            truths = {
                "answers": row["answers"],
                "id": uid,
            }
            # print(f'truths = {truths}')
            questions.append(SquadQuestion(uid, prompt, truths, self.max_tokens, self.metric))
        return questions


class SquadDatasetV2(SquadDataset):
    def __init__(self, max_tokens: int=64):
        super().__init__()
        self.dataset = load_dataset("rajpurkar/squad_v2")
        self.metric = evaluate.load("squad_v2")
        self.max_tokens = max_tokens
        self.data_ptr = 0

    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[SquadQuestion]:
        assert label in ["train", "validation"]
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
            
        # create questions
        questions = []
        for idx in indices:
            row = dataset[idx]
            uid = row['id']
            prompt = SQUAD_PROMPT + '\n\n' + \
                     "Title: " + row["title"] + '\n' + \
                     "Background: " + row["context"] + '\n' + \
                     "Question: " + row["question"] + '\n' + \
                     "Answer: " 
            truths = {
                "answers": row["answers"],
                "id": uid,
            }
            # print(f'truths = {truths}')
            questions.append(SquadQuestion(uid, prompt, truths, self.max_tokens, self.metric))
        return questions