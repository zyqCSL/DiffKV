import json
from typing import Optional, Dict, Tuple, List, Callable
from datasets import load_dataset
import evaluate as hf_evaluate

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from vllm.dataset.base_dataset import LLMDatasetType, LLMDataset
from vllm.outputs import RequestOutput
from vllm import SamplingParams


def setup_env():
    try:
        pass_at_k = hf_evaluate.load("code_eval")

        # run simple test to check code execution is enabled before model generation
        test_cases = ["assert add(2, 3)==5"]
        candidates = [["def add(a,b): return a*b"]]
        results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
        return pass_at_k
    except Exception as e:
        raise e

def pass_at_1(pass_at_k, references, predictions):    
    if isinstance(references, str):
        references = [references]
        
    if isinstance(predictions, str):
        predictions = [predictions]
    
    return pass_at_k.compute(
        references=references,
        predictions=[predictions],
        k=[1],
    )[0]["pass@1"]

class CodeGenerationQuestion:
    def __init__(
        self,
        prompt: str,
        truth: str,
        tests: str,
        max_tokens: int,
        sampling_n: int,
        temperature: float,
        stop_signs: List[str],
    ) -> None:
        self.prompt = prompt
        self.truth = truth
        assert len(self.truth) > 0
        self.max_tokens = max_tokens
        self.answer = None
        self.tests = tests
        
        self.sampling_n = sampling_n
        self.temperature = temperature
        self.stop_signs = stop_signs
        self.requests = []
        # mapping request id to answer
        self.candidate_answers: Dict[str, str] = {}
        self.program_result = []
        self.finished_requests = 0
        self.passed = False
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (self.prompt,
                SamplingParams(temperature=self.temperature, 
                               max_tokens=self.max_tokens, 
                               use_beam_search=False,
                               stop=self.stop_signs))
           
    def update_request_output(self, output: RequestOutput) -> None:
        ''' NOTE: this function needs benchmark specific implementation
        '''
        assert output.finished
        assert len(output.outputs) == 1
        answer = output.outputs[0].text
        if answer is None:
            self.answer = ''
            print(f'Warning: empty answer for request id {output.request_id} of prompt: {self.prompt}')
        
        self.candidate_answers[output.request_id] = answer
        self.finished_requests += 1
        
    
    def __repr__(self) -> str:
        if self.answer:
            return (f'prompt = {self.prompt}; '
                    f'truth = {self.truth}; '
                    f'answer = {self.answer}')
        else:
            return (f'prompt = {self.prompt}; '
                    f'truth = {self.truth}; '
                    f'answer = None')


class CodeGenerationDataset(LLMDataset):
    """
    Dataset for code generation,
    including HumanEval, mbpp
    """
    def __init__(self, sampling_n: int = 1, temperature: float = 0.0):
        super().__init__(LLMDatasetType.CODE_GENERATION)
        # mapping between questions and requests
        # every question should have sampling_n requests according to the mbpp paper,
        # which is used to evaluate the effect of sampling algorithm on synthesis
        # performance
        self.request_to_question: Dict[str, CodeGenerationQuestion] = {}
        
        self.sampling_n = sampling_n
        self.temperature = temperature
        self.correctness = []
        self.total_correct = 0
        
        # number of samples (number of indices from index_sampler)
        self.n = None
        self.data_ptr = 0
        
        # eval
        self.pass_at_k = setup_env()

    def register_request(
        self,
        question: CodeGenerationQuestion,
        request_id: str,
    ) -> None:
        self.request_to_question[request_id] = question
        question.requests.append(request_id)
        question.candidate_answers[request_id] = ""

    def complete_request(self, output: RequestOutput) -> None:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[output.request_id]
        assert question.sampling_n == self.sampling_n
        question.update_request_output(output)
        
        # print('predictions:')
        # print(question.candidate_answers[request_id])
        # print('**************************')
        # print('tests:')
        # print(question.tests)
        # print('---------------------------------------')
        
        result = pass_at_1(
            self.pass_at_k,
            references=question.tests,
            predictions=question.candidate_answers[request_id],
        )
        question.program_result.append(result)
        self.total_correct += result

    def get_scores_str(self) -> str:
        assert self.n is not None
        return f"Total correctness:{self.total_correct / self.n * 100}% {self.total_correct}/{self.n}"

    def get_scores(self) -> Tuple[int, int]:
        assert self.n is not None
        return self.total_correct, self.n


# https://arxiv.org/pdf/2107.03374
# Sampling params set according to Fig.1
# sampling_n = 100, temperature = 0.8
class HumanEvalDataset(CodeGenerationDataset):
    # def __init__(self, sampling_n: int = 100, temperature: float = 0.8):
    def __init__(self, sampling_n: int = 1, temperature: float = 0.0):
        super().__init__(sampling_n=sampling_n, temperature=temperature)
        # self.dataset = load_dataset("openai_humaneval")
        self.dataset = load_dataset("evalplus/humanevalplus")
        self.max_tokens = 512
        self.stop_signs = [
            "\nclass",
            "\ndef",
            "\n#",
            "\nif",
            "\nprint",
        ]

    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ):
        assert label == "test"
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

        self.n = len(indices)
        
        questions = []
        for idx in indices:
            row = dataset[idx]
            prompt = row["prompt"]
            truth = row["canonical_solution"]
            tests = row["test"] + f"check({row['entry_point']})"
            questions.append(CodeGenerationQuestion(
                prompt, truth, tests, 
                self.max_tokens, self.sampling_n, 
                self.temperature, self.stop_signs))
        return questions
    
    def complete_request(self, output: RequestOutput) -> None:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[output.request_id]
        assert question.sampling_n == self.sampling_n
        question.update_request_output(output)
        result = pass_at_1(
            self.pass_at_k,
            references=question.tests,
            predictions=question.prompt + question.candidate_answers[request_id],
        )
        question.program_result.append(result)
        self.total_correct += result


MBPP_PROMPT = "You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{test_list[0]}\n{test_list[1]}\n{test_list[2]}\n[BEGIN]\n"
# https://arxiv.org/pdf/2108.07732
# default sampling params set according to Section 3. (Page 5)
# sampling_n = 80, temperature = 0.5
class MbppDataset(CodeGenerationDataset):
    ''' Dataset for trivia questions, 
    including TriviaQA, MathQA
    '''
    def __init__(self, sampling_n: int = 1, temperature: float = 0.0):
        super().__init__(sampling_n=sampling_n, temperature=temperature)
        self.dataset = load_dataset("mbpp")
        self.max_tokens = 512
        self.stop_signs = ["[DONE]", "[DONE]\n"]
    
    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ):
        assert label in ["train", "validation", "test", "prompt"]
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

        self.n = len(indices)

        questions = []
        for idx in indices:
            row = dataset[idx]
            prompt = MBPP_PROMPT.format(**row)
            truth = row["code"]
            questions.append(CodeGenerationQuestion(
                prompt, truth, row["test_list"], 
                self.max_tokens, self.sampling_n, 
                self.temperature, self.stop_signs))
        return questions
    

def list_mbpp_plus_fewshot_samples():
    return [
        {
            "task_id": 2,
            "prompt": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "prompt": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "prompt": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]


MBPP_PLUS_PROMPT = "You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{test_list[0]}\n{test_list[1]}\n{test_list[2]}\n[BEGIN]\n"
MBPP_PLUS_TEMPLATE = "You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{test_list[0]}\n{test_list[1]}\n{test_list[2]}\n[BEGIN]\n{code}\n[DONE]\n"

# https://arxiv.org/pdf/2108.07732
# default sampling params set according to Section 3. (Page 5)
# sampling_n = 80, temperature = 0.5
class MbppPlusDataset(CodeGenerationDataset):
    ''' Dataset for trivia questions, 
    including TriviaQA, MathQA
    '''
    def __init__(self, sampling_n: int = 1, temperature: float = 0.0):
        super().__init__(sampling_n=sampling_n, temperature=temperature)
        self.dataset = load_dataset("evalplus/mbppplus")
        self.max_tokens = 512
        self.stop_signs = ["[DONE]", "[DONE]\n"]
        
        self.fewshot_prompt = ''
        for sample in list_mbpp_plus_fewshot_samples():
            self.fewshot_prompt += MBPP_PLUS_TEMPLATE.format(**sample)
    
    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ):
        assert label in ["test"]
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

        self.n = len(indices)

        questions = []
        for idx in indices:
            row = dataset[idx]
            prompt = self.fewshot_prompt + MBPP_PLUS_PROMPT.format(**row)
            truth = row["code"]
            questions.append(CodeGenerationQuestion(
                prompt, truth, row["test"], 
                self.max_tokens, self.sampling_n, 
                self.temperature, self.stop_signs))
        return questions