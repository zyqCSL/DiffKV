from typing import Optional, Tuple, List, Dict, Union
from datasets import load_dataset
import re
import string
import collections

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

class GSMEmulateQuestion:
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
                           stop=['Q:', 
                                 '</s>', 
                                 '<|im_end|>', 
                                 '<|start_header_id|>user<|end_header_id|>',
                                 '<|eot_id|>']),
        )

    def get_scores(self) -> bool:
        return self.result
    
    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        
        # self.result = output.outputs[0].text
        
        # assert len(output.outputs[0].emulated_text) > 0
        self.result = output.outputs[0].emulated_text
        
        return _is_correct(self.result, self.answer, zero_shot=False)
    
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


GSM8K_PROMPT = (
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n"
    "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n"
    "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n"
    "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\n"
    "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\n"
    "Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\n"
    "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n"
    "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n"
)

class GSM8kEmulateDataset(LLMDataset):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 prefill_tokens: int = 64, 
                 max_tokens: int = 512,
    ):
        super().__init__(LLMDatasetType.COT_QA)
        self.request_to_question: Dict[str, GSMEmulateQuestion] = {}
        self.dataset = load_dataset("gsm8k", "main")
        self.max_tokens = max_tokens
        self.prompt = GSM8K_PROMPT + "Q: {}\nA:"  
        
        # emulation config
        self.tokenizer = tokenizer
        self.prefill_tokens = prefill_tokens

        # runtime stats
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
        # total number of attended tokens across all steps of all seqs
        self.num_attended_tokens = 0

    def register_request(
        self,
        question: GSMEmulateQuestion,
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
    ):
        assert label in ["train", "test"]
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

        questions = []
        for idx in indices:
            row = dataset[idx]
            prompt = self.prompt.format(row["question"])
            answer = row["answer"]
            # ground truth token ids for prompt
            truth_token_ids = self.tokenizer.encode(prompt)
            prefill_token_ids = truth_token_ids[:self.prefill_tokens]
            
            questions.append(GSMEmulateQuestion(
                prompt, answer, 
                prefill_token_ids, truth_token_ids,
                self.max_tokens))
            
        return questions


_GSM8K_COT_RE = re.compile("The answer is (\\-?[0-9\\.\\,]*[0-9]+)")
_GSM8K_COT_RE_ALT = re.compile("The answer is \\$(\\-?[0-9\\.\\,]*[0-9]+)")
_TRUTH_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE_RE = re.compile("(-?[$0-9.,]{2,})|(-?[0-9]+)")
INVALID_ANS = "[invalid]"

def _find_match(regex: re.Pattern, resp: str, select_index: int = 0) -> str:
    match = regex.findall(resp)
    if match:
        match = match[select_index]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        match = match.strip()
        return match
    else:
        return INVALID_ANS

def extract(completion):
    """
    Extract the answer in the end of the completion.
    """
    match = _GSM8K_COT_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "").replace('$', '')
        return match_str
    else:
        return INVALID_ANS


def _is_correct(model_completion, gt_example, zero_shot: bool):
    gt_answer = _find_match(_TRUTH_RE, gt_example)
    assert gt_answer != INVALID_ANS
    result = _find_match(_GSM8K_COT_RE, model_completion)
    if result == INVALID_ANS:
        result = _find_match(_GSM8K_COT_RE_ALT, model_completion)
    if result == INVALID_ANS and zero_shot:
        result = _find_match(_FLEXIBLE_RE, model_completion, -1)
    result = result.replace(",", "").replace('$', '')
    # print(f"*********** gt_answer:{gt_answer} | result:{result} ***********")
    return result == gt_answer