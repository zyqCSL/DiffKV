from typing import Optional, Tuple, List, Dict
from datasets import load_dataset
import re
import random

from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams


class GPQACoTQuestion:
    def __init__(
        self,
        prompt: str,
        answer: str,
        max_tokens: int,
    ) -> None:
        self.prompt = prompt
        self.answer = answer
        self.max_tokens = max_tokens
        self.result = None
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        sampling_params = SamplingParams(
                n=1, temperature=0.6, top_p=0.95, min_p=0.0, top_k=40, 
                max_tokens=self.max_tokens,
                stop=["</s>", "<\think>"])
        # sampling_params = SamplingParams(
        #         n=1, temperature=0.0, max_tokens=self.max_tokens, 
        #         stop=["</s>", "<\think>"])
        return (
            self.prompt,
            sampling_params,
        )

    def get_scores(self) -> bool:
        return self.result
    
    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        self.result = output.outputs[0].text
        return _is_correct(self.result, self.answer)
    
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
            
SUBSETS = [
    'gpqa_diamond',
    'gpqa_extended',
    'gpqa_main',
]

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_doc(doc):
    # choices = [
    #     preprocess(doc["Incorrect Answer 1"]),
    #     preprocess(doc["Incorrect Answer 2"]),
    #     preprocess(doc["Incorrect Answer 3"]),
    #     preprocess(doc["Correct Answer"]),
    # ]
    # random.shuffle(choices)
    # correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))
    
    choices = [
        doc["Incorrect Answer 1"],
        doc["Incorrect Answer 2"],
        doc["Incorrect Answer 3"],
        doc["Correct Answer"],
    ]

    random.shuffle(choices)
    correct_answer_index = choices.index(doc["Correct Answer"])

    out_doc = {
        "Question": doc["Question"],
        "choice1": choices[0],
        "choice2": choices[1],
        "choice3": choices[2],
        "choice4": choices[3],
        "choices": [choices[0], choices[1], choices[2], choices[3]],
        "answer": f"{chr(65 + correct_answer_index)}",
    }
    return out_doc


GPQA_COT_PROMPT = "What is the correct answer to this question:{Question}\nChoices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}\nLet's think step by step: "
QWQ_COT_PROMPT = "What is the correct answer to this question:{Question}\nChoices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}\n" + \
                 "Please reason step by step." + "<think>\n"

class GPQACoTDataset(LLMDataset):
    def __init__(
        self, 
        max_tokens: int = 4096,
        model = '',
    ):
        super().__init__(LLMDatasetType.COT_QA)
        self.dataset = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')['train']
        self.max_tokens = max_tokens
        
        self.model = model
        assert model in ['', 'qwq']
        
        self.request_to_question: Dict[str, GPQACoTQuestion] = {}
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
    
    def register_request(
        self,
        question: GPQACoTQuestion,
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
    
    def get_scores_str(self) -> str:
        return f"Accuracy: {round(self.num_correct / self.num_total * 100, 2)}%, {self.num_correct}/{self.num_total}"

    def get_scores(self) -> Tuple[int, int]:
        return self.num_correct, self.num_total
    
    def sample(
        self,  
        n: Optional[int], 
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[GPQACoTQuestion]:
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
            item = process_doc(dataset[index])
            if self.model == '':
                prompt = GPQA_COT_PROMPT.format(**item)
            else:
                prompt = QWQ_COT_PROMPT.format(**item)
            answer = item['answer']            
            # print(prompt)
            questions.append(GPQACoTQuestion(prompt, answer, self.max_tokens))
        return questions


_GPQA_COT_RE = re.compile("(?<=The answer is )(.*)(?=.)")
# _GPQA_COT_FALLBACK_RE = re.compile("(\\([A-Z]\\))")
_GPQA_COT_FALLBACK_RE = re.compile("(A|B|C|D)")

# NOTE: we take last match here to adapt to QwQ's generation style
def _find_match(regex: re.Pattern, resp: str, select_index: int = -1) -> str:
    match = regex.findall(resp)
    if match:
        match = match[select_index]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        match = match.replace('(', '').replace(')', '')
        match = match.strip()
    return match


def _is_correct(model_completion: str, answer: str):
    match = _find_match(_GPQA_COT_RE, model_completion)
    if not match or len(match) > 1:
        match = _find_match(_GPQA_COT_FALLBACK_RE, model_completion)
    # print(f"*********** gt_answer:{answer} | result:{match} ***********")
    return match == answer