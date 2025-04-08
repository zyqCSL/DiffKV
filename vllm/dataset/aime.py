import re
import random

from typing import Optional, Tuple, List, Dict
from datasets import load_dataset, concatenate_datasets
from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams

from .math_grader import math_equal
from .math_parser import extract_answer, parse_ground_truth, strip_string

# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py

def qwq_doc_to_text(doc: dict) -> str:
    return doc["question"] + r" Please reason step by step, and put your final answer within \boxed{}." + "<think>\n"

def doc_to_text(doc: dict) -> str:
    return doc["question"] + r" Please reason step by step, and put your final answer within \boxed{}."         

def process_results(ground_truth, results: List[str]) -> bool:
    candidates = results[0]

    try:
        answer = extract_answer(candidates, data_name="aime24")
        answer = strip_string(answer, skip_unit=True)
        answer = normalize_final_answer(answer)
        # math_verify
        mathval = math_equal(ground_truth, answer, timeout=True)
    except Exception as e:
        print(f"Exception: {e}")
        mathval = False
        
    # print(f"*********** gt_answer:{ground_truth} | result:{answer}, correct = {mathval} ***********")

    return mathval


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


# def get_unnormalized_answer(text: str) -> str:
#     INVALID_ANSWER = "[invalidanswer]"
#     end_seq = "I hope it is correct."
#     text += end_seq
#     match = re.search(
#         r"Final Answer: The final answer is(.*?). I hope it is correct.",
#         text,
#     )
#     if match:
#         return match.group(1).strip()
#     else:
#         return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer

#----------------------- vLLM Interface -----------------------#

class AIMEQuestion:
    def __init__(
        self,
        prompt: str,
        answer: str,
        max_tokens: int,
        model: str,
    ) -> None:
        self.prompt = prompt
        self.answer = answer
        self.max_tokens = max_tokens
        self.result = None
        assert model in ["", "qwq"]
        self.model = model
    
    def make_request(self) -> Tuple[str, SamplingParams]:
        if self.model == "qwq":
            sampling_params = SamplingParams(
                n=1, temperature=0.6, top_p=0.95, min_p=0.0, top_k=40, 
                max_tokens=self.max_tokens,
                stop=["</s>", "<\think>"])
            # sampling_params = SamplingParams(
            #     n=1, temperature=0.0, max_tokens=self.max_tokens, 
            #     stop=["</s>", "<\think>"])
        else:
            sampling_params = SamplingParams(
                n=1, temperature=0.0, max_tokens=self.max_tokens, 
                stop=["Problem:", "Problem:\n"])
        return (
            self.prompt,
            sampling_params
            # SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens, 
            #                stop=["Problem:", "Problem:\n"]),
        )

    def get_scores(self) -> bool:
        return self.result
    
    def update_request_output(self, output: RequestOutput) -> bool:
        assert output.finished
        assert len(output.outputs) == 1
        self.result = output.outputs[0].text
        return process_results(self.answer, [self.result])
    
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


class AIMEDataset(LLMDataset):
    def __init__(self,
                 year_after: int = 2018,
                 max_tokens: int = 1024 * 16,
                 model: str = ''):
        super().__init__(LLMDatasetType.COT_QA)
        
        self.year_after = year_after
        # dataset = load_dataset('di-zhang-fdu/AIME_1983_2024')['train']
        # self.dataset = dataset.filter(
        #     lambda example: int(example['Year'].replace(',', '')) >= self.year_after)
        # def _process_doc_1983_2024(doc: dict) -> dict:
        #     out_doc = {
        #         "question": doc["Question"],
        #         "answer": doc["Answer"],
        #         "year": doc["Year"],
        #         "id": doc["ID"],
        #     }
        #     return out_doc
        # self.dataset = self.dataset.map(_process_doc_1983_2024)
        
        # NOTE: Using data after 2021 is to avoid potential overlap with the MATH training set
        # https://huggingface.co/datasets/AI-MO/aimo-validation-aime
        # self.dataset = load_dataset('AI-MO/aimo-validation-aime')['train']
        
        self.dataset = load_dataset('HuggingFaceH4/aime_2024')['train']
        
        def _process_doc_2022_2024(doc: dict) -> dict:
            out_doc = {
                "question": doc["problem"],
                "answer": doc["answer"],
                "id": doc["id"],
            }
            return out_doc
        self.dataset = self.dataset.map(_process_doc_2022_2024)
        
        self.model = model
        
        self.max_tokens = max_tokens
        self.request_to_question: Dict[str, AIMEQuestion] = {}
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
    
    def register_request(
        self,
        question: AIMEQuestion,
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
            
        #     print('CORRECT!!!!')
        # else:
        #     print('WRONG!!!!')
        # print('***************')
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
        repeats: int = 1,
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[AIMEQuestion]:
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
        
        indices = indices * repeats
        random.shuffle(indices)
        
        # make up questions
        questions = []   
        for index in indices:
            item = dataset[index]
            
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            if self.model == "qwq":
                prompt = qwq_doc_to_text(item)
            else:
                prompt = doc_to_text(item)
            # sol = remove_boxed(last_boxed_only_string(item["solution"]))
            # answer = extract_answer(sol, data_name="minerva_math")
            
            _, answer = parse_ground_truth(item, "aime24")
            
            # answer = remove_boxed(last_boxed_only_string(item["solution"]))
            # print(prompt)
            questions.append(AIMEQuestion(prompt, answer, self.max_tokens, model=self.model))
        return questions