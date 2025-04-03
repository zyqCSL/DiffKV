import enum
import numpy as np
from typing import Optional, List
import time

from vllm.outputs import RequestOutput


# base dataset classes
class LLMDatasetType(enum.Enum):
    MULTIPLE_CHOICE = enum.auto()
    TRIVIA = enum.auto()
    SUMMARY = enum.auto()
    SEQUENCE_MODELING = enum.auto()
    SQUAD = enum.auto()
    CODE_GENERATION = enum.auto()
    COT_QA = enum.auto()


class LLMDataset:
    def __init__(self, dtype: LLMDatasetType):
        self.dtype = dtype
        # sampling util
        self.index_sampler = IndexSampler()
    
    def complete_request(self, output: RequestOutput):
        raise NotImplementedError   


class IndexSampler:
    def __init__(self):
        np.random.seed(int(time.time()))
    
    def sample(
        self, 
        n: Optional[int], 
        is_random: bool, 
        size: int,
        start_id: int = 0,
    ) -> List[int]:
        if n is not None:
            assert n + start_id <= size, (
                f'Dataset has {size - start_id} samples left, '
                f'starting at {start_id}, but {n} required')
        if is_random:
            assert n is not None
            indices = np.random.choice(size, n, replace=False).tolist()
        elif n is None:
            indices = list(range(size))
        else:
            indices = list(range(n))
        return [x + start_id for x in indices]