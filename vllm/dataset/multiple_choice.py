from typing import Optional, Dict, Tuple, List
from datasets import load_dataset, Dataset
import re

from vllm.dataset.base_dataset import LLMDatasetType, LLMDataset
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup
from vllm import SamplingParams


# question classes
class MultipleChoiceQuestion:
    def __init__(
        self, 
        prompt: str,
        choices: Dict[str, str], 
        true_label: str,
        max_tokens: int,
    ) -> None:
        ''' Args
        choices: Dict[label, full_text]
        sep: puctuation to separate prompt & text
        '''  
        assert true_label in choices
        self.prompt = prompt
        self.choices = choices
        self.true_label = true_label    # ground truth label
        self.max_tokens = max_tokens
        # request mappings
        self.request_to_label: Dict[str, str] = {} # request_id to labels
        self.label_to_seq_group: Dict[str, SequenceGroup] = {}   # labels to seq_group
        self.logprobs = {}  # label to logprobs
    
    def make_requests(self) -> Dict[str, Tuple[str, SamplingParams]]:
        reqs = {}
        for label in self.choices:
            reqs[label] = (self.prompt, 
                           SamplingParams(
                               model_seq=True, truth=self.prompt + self.choices[label],
                               temperature=0.0, prompt_logprobs=0,
                               max_tokens=self.max_tokens))
        return reqs
    
    def update_request_map(self, seq_groups: Dict[str, SequenceGroup]) -> None:
        for label in seq_groups:
            # choice to seq_group
            assert label in self.choices
            assert label not in self.label_to_seq_group
            seq_group = seq_groups[label]
            self.label_to_seq_group[label] = seq_group
            # request_id to choice
            request_id = seq_group.request_id
            assert request_id not in self.request_to_label
            self.request_to_label[request_id] = label
    
    def is_complete(self) -> bool:
        return len(self.logprobs) == len(self.choices)
    
    def get_answer(self) -> str:
        assert len(self.logprobs) == len(self.choices)
        answer = max(self.logprobs, key= lambda x: self.logprobs[x])
        return answer
    
    def accurate(self) -> bool:
        answer = self.get_answer()
        print(f'prompt: {self.prompt}; '
              f'choices = {self.choices}; '
              f'logprobs: {self.logprobs}, '
              f'answer: {answer}, '
              f'truth: {self.true_label}')
        return answer == self.true_label
         
    def update_request_output(self, req_output: RequestOutput) -> Optional[bool]:
        request_id = req_output.request_id
        assert request_id in self.request_to_label
        label = self.request_to_label[request_id]
        assert label not in self.logprobs
        seq_group = self.label_to_seq_group[label]
        assert seq_group.is_finished()
        seqs = seq_group.get_seqs()
        assert len(seqs) == 1
        # NOTE (yanqi): now we only compute the perplexity of generated tokens
        # logprobs of prompts (question stem) are ignored 
        self.logprobs[label] = seqs[0].data.cumulative_logprob / seqs[0].get_output_len()
        # check if estimates of all choices are ready
        if len(self.logprobs) == len(self.choices):
            return self.accurate()
        else:
            return None
    
    def __repr__(self) -> str:
        return (f'prompt = {self.prompt}; '
                f'choices = {self.choices}; '
                f'true_lable = {self.true_label}; '
                f'log_probs = {self.logprobs}')


class MultipleChoiceDataset(LLMDataset):
    ''' Dataset for multiple choice questions, 
    including COPA, OpenBookQA, PIQA
    '''
    def __init__(self):
        super().__init__(LLMDatasetType.MULTIPLE_CHOICE)
        # mapping between questions and requests
        self.questions = []
        self.request_to_question: Dict[
            str, MultipleChoiceQuestion] = {}  # request_id to question
        
        # dataset specific metrics
        self.num_completed = 0
        self.num_accurate = 0
    
    def register_request(
        self, 
        question: MultipleChoiceQuestion, 
        request_ids: List[str],
    ) -> None:
        self.questions.append(question)
        for request_id in request_ids:
            self.request_to_question[request_id] = question

    def complete_request(self, output: RequestOutput) -> None:
        ''' receive a request output and update dataset metrics '''
        request_id = output.request_id
        assert request_id in self.request_to_question
        question = self.request_to_question[request_id]
        accurate = question.update_request_output(output)
        if accurate is not None:
            self.num_completed += 1
            self.num_accurate += accurate

    def accuracy(self) -> float:
        assert self.num_completed > 0
        return self.num_accurate / self.num_completed


# specific datasets
class OpenBookQADataset(MultipleChoiceDataset):
    def __init__(self, max_tokens: int = 1000):
        ''' Common sense reasoning '''
        super().__init__()
        self.dataset = load_dataset('openbookqa')
        self.max_tokens = max_tokens
        self.data_ptr = 0
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
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
        # make up questions
        questions = []   
        for index in indices:
            item = dataset[index]
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt = item['question_stem']
            choices = {}
            for label, text in zip(
                item['choices']['label'], item['choices']['text']):
                choices[label] = f' {text}'
            questions.append(MultipleChoiceQuestion(
                prompt, choices, item['answerKey'], self.max_tokens))
        return questions


class PIQADataset(MultipleChoiceDataset):
    def __init__(self, max_tokens: int = 1000):
        ''' Common sense reasoning '''
        super().__init__()
        self.dataset = load_dataset('piqa')
        self.max_tokens = max_tokens
        self.data_ptr = 0
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
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
        # make up questions
        questions = []   
        for index in indices:
            item = dataset[index]
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt =  'Question: ' + item['goal'] + '\nAnswer:'
            choices = {}
            for label in ['sol1', 'sol2']:
                choices[label] = item[label]
            true_label = 'sol' + str(item['label'] + 1)
            assert true_label in ['sol1', 'sol2']
            questions.append(MultipleChoiceQuestion(
                prompt, choices, true_label, self.max_tokens))
        return questions


class ARCDataset(MultipleChoiceDataset):
    def __init__(self, config: str, max_tokens: int = 1000):
        ''' Common sense reasoning '''
        super().__init__()
        self.config = config
        assert self.config in ['ARC-Challenge', 'ARC-Easy']
        self.dataset = load_dataset('allenai/ai2_arc', self.config)
        self.max_tokens = max_tokens
        self.data_ptr = 0
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
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
            
        # make up questions
        questions = []   
        for index in indices:
            item = dataset[index]
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt =  'Question: ' + item['question'] + '\nAnswer:'
            choices = {}
            for label, text in zip(
                item['choices']['label'], item['choices']['text']):
                choices[label] = f' {text}'
            questions.append(MultipleChoiceQuestion(
                prompt, choices, item['answerKey'], self.max_tokens))
        return questions
    
    
class SIQADataset(MultipleChoiceDataset):
    def __init__(self, max_tokens: int = 1000):
        ''' Common sense reasoning '''
        super().__init__()
        self.dataset = load_dataset('social_i_qa')
        self.max_tokens = max_tokens
        self.labels = ['answerA', 'answerB', 'answerC']
        self.data_ptr = 0
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        assert label in ['train', 'validation']
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
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt =  'Question: ' + item['context'] + ' ' + item['question'] + '\nAnswer:'            
            choices = {}
            for label in self.labels:
                choices[label] = f' {item[label]}'
            answer_label = self.labels[int(item['label']) - 1] 
            questions.append(MultipleChoiceQuestion(
                prompt, choices, answer_label, self.max_tokens))
        return questions
    

class WinoGrandeDataset(MultipleChoiceDataset):
    def __init__(self, max_tokens: int = 1000):
        ''' Common sense reasoning '''
        super().__init__()
        self.dataset = load_dataset('winogrande', 'winogrande_debiased')
        self.max_tokens = max_tokens
        self.labels = ['option1', 'option2']
        self.data_ptr = 0
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        assert label in ['train', 'test', 'validation']
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
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            cut_index = item['sentence'].index('_')
            if cut_index > 0 and item['sentence'][cut_index - 1] == ' ':
                cut_index -= 1
            prompt = item['sentence'][:cut_index]
            q = item['sentence'][cut_index:]
            choices = {}
            for label in self.labels:
                choices[label] = q.replace('_', item[label])
            answer_label = self.labels[int(item['answer']) - 1] 
            questions.append(MultipleChoiceQuestion(
                prompt, choices, answer_label, self.max_tokens))
        return questions
    
    
class MathQADataset(MultipleChoiceDataset):
    def __init__(self, max_tokens: int = 1000):
        ''' math reasoning '''
        super().__init__()
        self.dataset = load_dataset('math_qa')
        self.max_tokens = max_tokens
        self.labels = ['a', 'b', 'c', 'd', 'e']
        self.data_ptr = 0
    
    def doc_to_choice(cls, doc: str):
        choices = [
            c[4:].rstrip(" ,")
            for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
        ]
        return choices
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        assert label in ['train', 'test', 'validation']
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
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt =  'Question: ' + item['Problem'] + '\nAnswer:'            
            choices = {}
            for label, choice in zip(self.labels, self.doc_to_choice(item)):
                choices[label] = f' {choice}'
            questions.append(MultipleChoiceQuestion(
                prompt, choices, item['correct'], self.max_tokens))
        return questions


class RaceDataset(MultipleChoiceDataset):
    def __init__(self, max_tokens: int = 1000):
        ''' reading comprehension '''
        super().__init__()
        self.dataset = load_dataset('ehovy/race', 'all')
        self.max_tokens = max_tokens
        self.labels = ['A', 'B', 'C', 'D']
        self.data_ptr = 0
    
    def reformat(self, item: dict) -> Tuple[str, List[str]]:
        ''' reformat a raw doc item to prompt & choices '''
        prompt = "Article: " + item['article'] + "\n\n"
        choices = {}
        if '_' in item['question']:
            # blank filling
            cut_index = item['question'].index('_')
            shared = item['question'][:cut_index].rstrip()
            base = item['question'][cut_index:]
            
            prompt = prompt + shared
            for label, choice in zip(self.labels, item['options']):
                choices[label] = ' ' + base.replace('_', choice)
        else:
            # qa
            prompt = prompt + 'Question: ' + item['question'] + '\nAnswer:'
            for label, choice in zip(self.labels, item['options']):
                choices[label] = f' {choice}'
                
        return prompt, choices
    
    def sample(
        self, 
        label: str, 
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MultipleChoiceQuestion]:
        ''' sample the rows and convert them into multiple choice questions '''
        assert label in ['train', 'test', 'validation']
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
        
        # indices = [0, 3]
        # make up questions
        questions = []   
        for index in indices:
            item = dataset[index]
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            prompt, choices = self.reformat(item)
            assert item['answer'] in self.labels
            questions.append(MultipleChoiceQuestion(
                prompt, choices, item['answer'], self.max_tokens))
        return questions