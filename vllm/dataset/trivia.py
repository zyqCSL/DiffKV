from typing import Optional, Dict, Tuple, List
from datasets import load_dataset

from vllm.dataset.base_dataset import LLMDatasetType, LLMDataset
from vllm.outputs import RequestOutput
from vllm import SamplingParams


# question classes
class TriviaQuestion:
    def __init__(
        self, 
        prompt: str,
        truths: List[str],
        max_tokens: int,
    ) -> None:
        self.prompt = prompt
        self.truths = truths
        assert len(self.truths) > 0
        self.max_tokens = max_tokens
        self.answer = None
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (self.prompt, 
                SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens))
    
    def accurate(self) -> bool:
        assert self.answer is not None
        for truth in self.truths:
            if truth in self.answer:
                return True
        return False
            
    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        self.answer = output.outputs[0].text
        if self.answer is None:
            self.answer = ''
            print(f'Warning: empty answer for prompt: {self.prompt}')
        # TODO: check how ROGUE-2 is computed
        return self.accurate()
    
    def __repr__(self) -> str:
        return (f'prompt = {self.prompt}; '
                f'truths = {self.truths}; '
                f'answer = {self.answer}')


class TriviaDataset(LLMDataset):
    ''' Dataset for trivia questions, 
    including TriviaQA, MathQA
    '''
    def __init__(self):
        super().__init__(LLMDatasetType.TRIVIA)
        # mapping between questions and requests
        self.request_to_question: Dict[str, TriviaQuestion] = {}
        
        # accuracy
        self.num_finished = 0
        self.num_correct = 0
    
    def register_request(
        self, 
        question: TriviaQuestion, 
        request_id: str,
    ) -> None:
        self.request_to_question[request_id] = question
    
    def complete_request(self, output: RequestOutput) -> None:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        self.num_finished += 1
        if question.update_request_output(output):
            self.num_correct += 1
        print(question)
    
    def get_accuracy(self) -> float:
        assert self.num_finished > 0
        return self.num_correct / self.num_finished


class TriviaQADataset(TriviaDataset):
    def __init__(self, max_tokens: int=50) -> None:
        super().__init__()
        self.dataset = load_dataset('mandarjoshi/trivia_qa', 'rc')
        self.max_tokens = max_tokens
    
    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
    ) -> List[TriviaQuestion]:
        assert label in ['train', 'validation', 'test']
        dataset = self.dataset[label]
        # sample row indices
        indices = self.index_sampler.sample(n, is_random, len(dataset))
        # create questions 
        questions = []
        for index in indices:
            item = dataset[index]
            question = TriviaQuestion(
                item['question'],
                item['answer']['aliases'],
                self.max_tokens,
            )
            questions.append(question)
        return questions