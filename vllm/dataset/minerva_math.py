import re

from typing import Optional, Tuple, List, Dict
from datasets import load_dataset, concatenate_datasets
from vllm.dataset.base_dataset import LLMDataset, LLMDatasetType
from vllm.outputs import RequestOutput
from vllm import SamplingParams

from .math_grader import math_equal
from .math_parser import extract_answer, parse_ground_truth, strip_string

# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"

def qwq_doc_to_text(doc: dict) -> str:
    return doc["problem"] + r" Please reason step by step, and put your final answer within \boxed{}." + "<think>\n"
            
def doc_to_fewshot(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:" + "\n" + doc["solution"] + "\n\n"

def list_fewshot_samples() -> List[Dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def get_fewshot_prompt():
    prompt = ''
    for sample in list_fewshot_samples():
        prompt += doc_to_fewshot(sample)
    return prompt


def process_results(ground_truth, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    answer = extract_answer(candidates, data_name="minerva_math")
    answer = strip_string(answer, skip_unit=True)
    answer = normalize_final_answer(answer)
    # math_verify
    mathval = math_equal(ground_truth, answer, timeout=True)
    
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

class MinervaMathQuestion:
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
                n=1, temperature=0.6, top_p=0.95, min_p=0.0, 
                max_tokens=self.max_tokens)
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

SUBSETS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus',
]

class MinervaMathDataset(LLMDataset):
    def __init__(self,
                 label: str,
                 sample_percent: int = 100,
                 subset: str = 'all',
                 max_tokens: int = 1024,
                 model: str = ''):
        super().__init__(LLMDatasetType.COT_QA)
        
        assert label in ['train', 'test']
        self.label = label
        
        if subset == 'all':
            self.subsets = SUBSETS
        else:
            assert subset in SUBSETS
            self.subsets = [subset]
        self.sample_percent = sample_percent
    
        # make up the dataset
        all_datasets = []
        for subset in self.subsets:
            _dataset = load_dataset('EleutherAI/hendrycks_math', subset)[self.label]
            if self.sample_percent < 100:
                _dataset = _dataset.select(
                    range(0, len(_dataset), 100 // self.sample_percent))
            all_datasets.append(_dataset)
        self.dataset = concatenate_datasets(all_datasets)
        
        assert model in ["", "qwq"]
        self.model = model
        # 4-shot prompt
        if self.model == "qwq":
            self.few_shot_prompt = ""
        else:
            self.few_shot_prompt = get_fewshot_prompt()
        
        self.max_tokens = max_tokens
        self.request_to_question: Dict[str, MinervaMathQuestion] = {}
        self.num_correct = 0
        self.num_total = 0
        self.data_ptr = 0
    
    def register_request(
        self,
        question: MinervaMathQuestion,
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
        is_random: bool = False,
        continued: bool = False,
        indices: Optional[List[int]] = None,
    ) -> List[MinervaMathQuestion]:
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
            item = dataset[index]
            
            # separate prompt & choices w. space
            # NOTE: the space between stem & choice should be added to the choice
            # otherwise the prompt tokens and truth tokens mismatch
            if self.model == "qwq":
                prompt = qwq_doc_to_text(item)
            else:
                prompt = self.few_shot_prompt + doc_to_text(item)
            # sol = remove_boxed(last_boxed_only_string(item["solution"]))
            # answer = extract_answer(sol, data_name="minerva_math")
            
            _, answer = parse_ground_truth(item, "minerva_math")
            
            # answer = remove_boxed(last_boxed_only_string(item["solution"]))
            # print(prompt)
            questions.append(MinervaMathQuestion(prompt, answer, self.max_tokens, model=self.model))
        return questions