import re
from typing import Optional, Dict, Tuple, List, Union
from datasets import load_dataset, Dataset

from vllm.dataset.base_dataset import LLMDatasetType, LLMDataset
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup
from vllm import SamplingParams
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

LLMRequest = Tuple[List[int], SamplingParams]


class ModelingDataset(LLMDataset):
    def __init__(self):
        super().__init__(LLMDatasetType.SEQUENCE_MODELING)
        self.requests: Dict[str, SequenceGroup] = {}  # indexed by req_id
        # metrics
        self.num_tokens = 0
        self.cum_logprobs = 0
    
    def register_request(self, seq_group: SequenceGroup):
        assert len(seq_group.get_seqs()) == 1
        request_id = seq_group.request_id
        assert request_id not in self.requests
        self.requests[request_id] = seq_group
    
    def complete_request(self, output: RequestOutput):
        request_id = output.request_id
        assert request_id in self.requests
        seq_group = self.requests[request_id]
        assert seq_group.is_finished()
        seq = seq_group.get_seqs()[0]
        # update metric
        self.num_tokens += seq.get_output_len()
        # print(f'seq output_len = {seq.get_output_len()}')
        self.cum_logprobs += seq.data.cumulative_logprob
        
    def perplexity(self) -> float:
        return -self.cum_logprobs / self.num_tokens


# specific datasets
class WikiDataset(ModelingDataset):
    def __init__(
        self, 
        prompt_tokens: int,
        max_tokens: int,
        min_tokens: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        super().__init__()
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1')
        assert min_tokens > prompt_tokens
        assert max_tokens > min_tokens
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokenizer = tokenizer
        
        # dataset specific metrics
        self.wiki_heading = ' = [^=^;]*[^=^;] = \n'
        
        # sample stats
        self.line_ptr = 0
        
    def sample(
        self,
        label: str,
        start_line: Optional[int] = None,
        num_lines: Optional[int] = None,
        n: Optional[int] = None,
        is_random: bool = False,
        continued: bool = False,
    ) -> List[LLMRequest]:
        ''' Args
        n: number of articles to sample
        continued: True if sampling from the line that previous sample ends
        '''
        assert label in ['train', 'validation', 'test']
        dataset = self._make_articles(self.dataset[label]['text'], num_lines, start_line, continued)
        if not dataset:
            return []
        indices = self.index_sampler.sample(n, is_random, len(dataset))
        
        # make up questions
        requests = []
        for index in indices:
            article = dataset[index]
            requests.extend(self._make_requests(article))
        return requests
            
                
    # convert wiki-text strings to articles
    def _make_articles(
        self, 
        data: List[str], 
        num_lines: Optional[int], 
        start_line: Optional[int],
        continued: bool,
    ) -> List[List[str]]:
        ''' convert wiki-text strings to articles
        return: each List[str] is an article, and str being a line
        '''
        start = -1
        init_line = 0
        if continued:
            assert self.line_ptr >= 0
            init_line = self.line_ptr
        if start_line is not None:
            assert start_line >= 0
            assert not continued
            self.line_ptr = start_line
            init_line = start_line
            
        articles = []
        assert init_line >= 0
        # print(f'init_line = {init_line}')
        # for i, text in enumerate(data[init_line:]):
        for i in range(init_line, len(data)):
            text = data[i]
            if re.match(self.wiki_heading, text):
                if start >= 0:
                    # yield(''.join(WIKI_DATA[start:i]))
                    d = data[start:i]
                    while d[-1] in ['\n', '']:
                        d = d[:-1]
                    if d:
                        articles.append(d)
                    start = i
                else:
                    start = i
                if num_lines is not None and i >= num_lines + init_line:
                    break
        # last piece in the dataset
        if i == len(data) - 1 and start < len(data) - 1:
            assert start > 0
            articles.append(data[start:])
        # update pointer for continued sampling
        self.line_ptr = start
        # print(f'line_ptr updated to {self.line_ptr}')
        return articles


    def _make_requests(self, article: List[str]) -> List[LLMRequest]:
        ''' convert an article to requests. Number of requests depends on max_tokens
        return: List[(prompt_token_ids, sampling_params)]
        '''
        line_token_ids = []
        for line in article:
            line_token_ids.append(self.tokenizer.encode(line))
        
        # NOTE: we need to split an article into multiple requests
        # if it is longer than max_tokens
        ptr = 0
        requests = []
        token_ids = []
        while ptr < len(line_token_ids):
            if len(token_ids) + len(line_token_ids[ptr]) > self.max_tokens:
                # Create the request
                prompt_token_ids = token_ids[:self.prompt_tokens]
                truth_token_ids = token_ids
                requests.append((prompt_token_ids, SamplingParams(
                    model_seq=True, truth_token_ids=truth_token_ids,
                    temperature=0.0, prompt_logprobs=0,
                    max_tokens=self.max_tokens)))
                # start a new request
                token_ids = []
                token_ids.extend(line_token_ids[ptr])
                ptr += 1
            else:
                token_ids.extend(line_token_ids[ptr])
                ptr += 1
        
        # add the one last remaining part
        if len(token_ids) >= self.min_tokens:
            # Create the sequences
            prompt_token_ids = token_ids[:self.prompt_tokens]
            truth_token_ids = token_ids
            requests.append((prompt_token_ids, SamplingParams(
                model_seq=True, truth_token_ids=truth_token_ids,
                temperature=0.0, prompt_logprobs=0,
                max_tokens=self.max_tokens)))
        
        return requests
        