from typing import Optional, Dict, Tuple, List
from datasets import load_dataset, Dataset
from rouge import Rouge

from vllm.dataset.base_dataset import LLMDatasetType, LLMDataset
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup
from vllm import SamplingParams

# NOTE: Maximum recursion depth exceeded in comparison
# https://github.com/pltrdy/rouge/issues/19

class SummaryQuestion:
    def __init__(
        self, 
        prompt: str,
        truth: str,
        rouge: Rouge,
        max_tokens: int,
    ):
        self.prompt = prompt
        self.truth = truth
        self.answer = None
        self.rouge = rouge
        self.max_tokens = max_tokens
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        return (self.prompt, 
                SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, stop=["```"]))
    
    def get_scores(self) -> Optional[Tuple[float, float, float]]:
        if self.answer:
            if self.answer[-1] == "```":
                self.answer = self.answer[:-1]
            scores = self.rouge.get_scores(hyps=self.answer, refs=self.truth)
            assert len(scores) == 1
            return (scores[0]['rouge-1']['f'], 
                    scores[0]['rouge-2']['f'],
                    scores[0]['rouge-l']['f'])
        else:
            return (0, 0, 0)
    
    def update_request_output(
        self, 
        output: RequestOutput
    ) -> Tuple[float, float, float]:
        assert output.finished
        assert len(output.outputs) == 1
        self.answer = output.outputs[0].text
        if self.answer is None:
            self.answer = ''
            print(f'Warning: empty answer for prompt: {self.prompt}')
        # TODO: check how ROGUE-2 is computed
        return self.get_scores()
    
    def __repr__(self) -> str:
        if self.answer:
            return (f'prompt: {self.prompt}\n'
                    f'answer: {self.answer}\n'
                    f'truth: {self.truth}\n'
                    f'scores: {self.get_scores()}\n')
        else:
            return (f'prompt: {self.prompt}\n'
                    f'answer: {self.answer}\n'
                    f'truth: {self.truth}\n')


class SummaryDataset(LLMDataset):
    ''' Dataset for summarization questions, 
    including CNN/DailyMail, XSum
    '''
    def __init__(self):
        super().__init__(LLMDatasetType.SUMMARY)
        # mapping between questions and requests
        self.request_to_question: Dict[str, SummaryQuestion] = {}
        
        # rogue score
        self.rouge = Rouge()
        self.rouge_1_scores = []
        self.rouge_2_scores = []
        self.rouge_l_scores = []
    
    def register_request(
        self, 
        question: SummaryQuestion, 
        request_id: str,
    ) -> None:
        self.request_to_question[request_id] = question
    
    def complete_request(self, output: RequestOutput) -> None:
        assert output.finished
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        r1, r2, rl = question.update_request_output(output)
        self.rouge_1_scores.append(r1)
        self.rouge_2_scores.append(r2)
        self.rouge_l_scores.append(rl)
        # print(question)
    
    def get_scores(self) -> Tuple[float, float, float]:
        return (sum(self.rouge_1_scores) / len(self.rouge_1_scores),
                sum(self.rouge_2_scores) / len(self.rouge_2_scores),
                sum(self.rouge_l_scores) / len(self.rouge_l_scores),)


# CNN_DAILYMAIL_PROMPT = ('Summarize the article in 2-3 sentences: ')
# CNN_DAILYMAIL_PROMPT = 'Summarize this text in 2-3 sentences: '
CNN_DAILYMAIL_PROMPT = (
    "Write a concise summary of the text\n"
    "Return your responses with 3 lines that cover the key points of the text.")

CNN_DAILYMAIL_PROMPT = """Write a concise summary of the following text delimited by triple backticks.
Return your response in 3 lines which covers the key points of the text.
```{text}```
SUMMARY:
"""

class CNNDailyMailDataset(SummaryDataset):
    def __init__(
        self, 
        version: str='3.0.0',
        prompt: str = CNN_DAILYMAIL_PROMPT, 
        max_tokens: int=512,
    ) -> None:
        super().__init__()
        self.version = version
        self.dataset = load_dataset('cnn_dailymail', version)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.data_ptr = 0
    
    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[SummaryQuestion]:
        assert label in ['train', 'validation', 'test']
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
        for index in indices:
            item = dataset[index]
            
            question = SummaryQuestion(
                # prompt=self.prompt + '\n\"' + item['article'] + '\"',
                prompt=self.prompt.replace("{text}", item['article']),
                truth=item['highlights'],
                rouge=self.rouge,
                max_tokens=self.max_tokens,
            )
            questions.append(question)
        return questions


XSUM_PROMPT = 'Summarize the following article in one sentence: '

class XSumDataset(SummaryDataset):
    def __init__(
        self, 
        prompt: str = XSUM_PROMPT, 
        max_tokens: int=512,
    ) -> None:
        super().__init__()
        self.dataset = load_dataset('EdinburghNLP/xsum')
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.data_ptr = 0
    
    def sample(
        self,
        label: str,
        n: Optional[int],
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[SummaryQuestion]:
        assert label in ['train', 'validation', 'test']
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
        for index in indices:
            item = dataset[index]
            question = SummaryQuestion(
                prompt=self.prompt + '\"' + item['document'] + '\"',
                truth=item['summary'],
                rouge=self.rouge,
                max_tokens=self.max_tokens,
            )
            questions.append(question)
        return questions