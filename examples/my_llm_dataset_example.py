import argparse
from typing import List, Tuple, Optional, Union
import numpy as np
import time

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

from vllm.dataset import (
    MultipleChoiceQuestion,
    MultipleChoiceDataset,
    OpenBookQADataset,
    PIQADataset,
    ARCDataset,
    SIQADataset,
    WinoGrandeDataset,
    MathQADataset,
    RaceDataset,
    TriviaQuestion,
    TriviaDataset,
    TriviaQADataset,
    SummaryQuestion,
    SummaryDataset,
    CNNDailyMailDataset,
    XSumDataset,
    ModelingDataset,
    WikiDataset,
    SquadQuestion,
    SquadDataset,
    SquadDatasetV1,
    SquadDatasetV2,
    CodeGenerationQuestion,
    CodeGenerationDataset,
    HumanEvalDataset,
    MbppDataset,
    Llama2MbppDataset,
    MathQAPythonDataset,
    GSMQuestion,
    GSMDataset,
    GSM8kDataset,
    MMLUCoTQuestion,
    MMLUCoTDataset,
)

np.random.seed(int(time.time()))

REQUEST_ID = 0


QUANT_CONFIGS=[8, 8]
COMPRESS_CONFIGS = [0.0, 0.0]

# ------------ sequence modeling tasks ------------#
def add_modeling_requests(
    engine: LLMEngine,
    dataset: ModelingDataset,
    requests: List[Tuple[List[int], SamplingParams]],
) -> None:
    global REQUEST_ID
    for prompt_token_ids, sampling_params in requests:
        seq_group = engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            quant_configs=QUANT_CONFIGS,
            compress_configs=COMPRESS_CONFIGS,
        )
        dataset.register_request(seq_group)
        REQUEST_ID += 1


def test_wiki_dataset(
    engine: LLMEngine,
    num_lines: Optional[int] = None,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    # dataset = WikiDataset(
    #     prompt_tokens=256,
    #     max_tokens=4096,
    #     min_tokens=1024,
    #     tokenizer=engine.tokenizer)
    dataset = WikiDataset(
        prompt_tokens=256, max_tokens=4096, min_tokens=1024, tokenizer=engine.tokenizer
    )
    requests = dataset.sample("train", num_lines, n, is_random)
    add_modeling_requests(engine, dataset, requests)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


# ------------ multiple choice tasks ------------#
def add_multiple_choice_requests(
    engine: LLMEngine,
    dataset: MultipleChoiceDataset,
    questions: List[MultipleChoiceQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        print(question)
        reqs = question.make_requests()
        seq_groups = {}
        request_ids = []
        for label in reqs:
            prompt, sampling_params = reqs[label]
            print(prompt)
            print(sampling_params)
            print("*********")
            seq_group = engine.add_request(
                request_id=str(REQUEST_ID),
                prompt=prompt,
                sampling_params=sampling_params,
                quant_configs=QUANT_CONFIGS,
                compress_configs=COMPRESS_CONFIGS,
            )
            seq_groups[label] = seq_group
            request_ids.append(str(REQUEST_ID))
            REQUEST_ID += 1
        question.update_request_map(seq_groups)
        dataset.register_request(question, request_ids)


def test_openbookqa(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = OpenBookQADataset()
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


def test_piqa(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = PIQADataset()
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


def test_arc(
    engine: LLMEngine,
    easy: bool,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    if easy:
        dataset = ARCDataset("ARC-Easy")
    else:
        dataset = ARCDataset("ARC-Challenge")
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


def test_siqa(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = SIQADataset()
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


def test_winogrande(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = WinoGrandeDataset()
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


def test_mathqa(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = MathQADataset()
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


def test_race(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = RaceDataset()
    questions = dataset.sample("train", n, is_random)
    add_multiple_choice_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                # print(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")


# ------------ summary tasks ------------#
def add_trivia_requests(
    engine: LLMEngine,
    dataset: TriviaDataset,
    questions: List[TriviaQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        # print(question)
        prompt, sampling_params = question.make_request()
        print(prompt)
        print(sampling_params)
        print("*********")
        engine.add_request(
            request_id=str(REQUEST_ID), 
            prompt=prompt, 
            sampling_params=sampling_params,
            quant_configs=QUANT_CONFIGS,
            compress_configs=COMPRESS_CONFIGS,
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1


def test_trivia_qa(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = TriviaQADataset()
    questions = dataset.sample("train", n, is_random)
    add_summary_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")

    print(dataset.get_accuracy())


# ------------ summary tasks ------------#
def add_summary_requests(
    engine: LLMEngine,
    dataset: SummaryDataset,
    questions: List[SummaryQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        # print(question)
        prompt, sampling_params = question.make_request()
        engine.add_request(
            request_id=str(REQUEST_ID), 
            prompt=prompt, 
            sampling_params=sampling_params,
            quant_configs=QUANT_CONFIGS,
            compress_configs=COMPRESS_CONFIGS,
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1


def test_cnn_dailymail(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = CNNDailyMailDataset()
    questions = dataset.sample("train", n, is_random)
    for question in questions:
        print(question)
        print("----------")
    add_summary_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")

    print(dataset.get_scores())


def test_xsum(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = XSumDataset()
    questions = dataset.sample("train", n, is_random)
    for question in questions:
        print(question)
        print("----------")
    add_summary_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")

    print(dataset.get_scores())


def add_squad_requests(
    engine: LLMEngine,
    dataset: SquadDataset,
    questions: List[SquadQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        # print(question)
        prompt, sampling_params = question.make_request()
        engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=prompt,
            sampling_params=sampling_params,
            quant_configs=QUANT_CONFIGS,
            compress_configs=COMPRESS_CONFIGS,
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1


def test_squad(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
    use_v2: bool = True,
) -> None:
    dataset = SquadDatasetV2() if use_v2 else SquadDatasetV1()
    questions = dataset.sample("train", n, is_random)
    add_squad_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")

    print(dataset.get_scores())


def add_code_generation_requests(
    engine: LLMEngine,
    dataset: CodeGenerationDataset,
    questions: List[CodeGenerationQuestion],
    sampling_n: int = 1,
) -> None:
    global REQUEST_ID
    for question in questions:
        # print(question)
        prompt, sampling_params = question.make_request()
        for _i in range(sampling_n):
            engine.add_request(
                request_id=str(REQUEST_ID), 
                prompt=prompt, 
                sampling_params=sampling_params,
                quant_configs=QUANT_CONFIGS,
                compress_configs=COMPRESS_CONFIGS,
            )
            dataset.register_request(question, str(REQUEST_ID))
            REQUEST_ID += 1


def test_humaneval(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
    sampling_n: int = 1,
    temperature: float = 0.0,
) -> None:
    dataset = HumanEvalDataset(sampling_n=sampling_n)
    questions = dataset.sample("test", n, is_random)
    add_code_generation_requests(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")
    print(dataset.get_scores())


def test_mbpp(
    engine: LLMEngine,
    is_llama2: bool = False,
    n: Optional[int] = None,
    is_random: bool = False,
    sampling_n: int = 1,
    temperature: float = 0.0,
) -> None:
    if not is_llama2:
        dataset = MbppDataset(sampling_n=sampling_n)
    else:
        dataset = Llama2MbppDataset(sampling_n=sampling_n)
    questions = dataset.sample("test", n, is_random)
    add_code_generation_requests(engine, dataset, questions, sampling_n)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")
    print(dataset.get_scores())


def test_mathqa_python(
    engine: LLMEngine,
    file_path: str,
    n: Optional[int] = None,
    is_random: bool = False,
    sampling_n: int = 1,
    temperature: float = 0.0,
)-> None:
    dataset = MathQAPythonDataset(file_path, sampling_n=sampling_n)
    questions = dataset.sample(n, is_random)
    add_code_generation_requests(engine, dataset, questions, sampling_n, temperature)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")
    print(dataset.get_scores())


def add_gsm_questions(
    engine: LLMEngine,
    dataset: GSMDataset,
    questions: List[GSMQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        prompt, SamplingParams = question.make_request()
        engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=prompt,
            sampling_params=SamplingParams,
            quant_configs=QUANT_CONFIGS,
            compress_configs=COMPRESS_CONFIGS,
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1


def test_gsm8k(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = GSM8kDataset()
    questions = dataset.sample("test", n, is_random)
    add_gsm_questions(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")
    print(dataset.get_scores_str())


def add_mmlu_cot_questions(
    engine: LLMEngine,
    dataset: MMLUCoTDataset,
    questions: List[MMLUCoTQuestion],
) -> None:
    global REQUEST_ID
    for question in questions:
        prompt, SamplingParams = question.make_request()
        engine.add_request(
            request_id=str(REQUEST_ID),
            prompt=prompt,
            sampling_params=SamplingParams,
            quant_configs=QUANT_CONFIGS,
            compress_configs=COMPRESS_CONFIGS,
        )
        dataset.register_request(question, str(REQUEST_ID))
        REQUEST_ID += 1


def test_mmlu_cot(
    engine: LLMEngine,
    n: Optional[int] = None,
    is_random: bool = False,
) -> None:
    dataset = MMLUCoTDataset()
    questions = dataset.sample("test", n, is_random)
    add_mmlu_cot_questions(engine, dataset, questions)

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                dataset.complete_request(request_output)
                print(f"request {request_output.request_id} finished")
                print(" ************ ")
    print(dataset.get_scores_str())



def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)

    # test_openbookqa(engine, 4, False)
    # test_piqa(engine, 4, True)
    # test_wiki_dataset(engine, 4096, 2, False)
    # test_cnn_dailymail(engine, 5, False)
    # test_xsum(engine, 2, True)
    # test_arc(engine, True, 2, True) # easy
    # test_arc(engine, False, 2, True) # challenge
    # test_siqa(engine, 3, True)
    # test_winogrande(engine, 3, True)
    # test_trivia_qa(engine, 2, True)
    # test_mathqa(engine, 2, True)
    # test_race(engine, 2, True)
    test_squad(engine, 50, False)
    # test_humaneval(engine, 164, False)
    # test_mbpp(engine, n=50, is_random=False, sampling_n=1, temperature=0.0)
    # test_mbpp(engine, is_llama2=True, n=50, is_random=False, sampling_n=1, temperature=0.0)
    # test_mathqa_python(engine, "/home/zhaorunyuan/mathqa/mathqa-python.json", 20, True, sampling_n=1, temperature=0.5)
    # test_gsm8k(engine, 5, False)
    # test_mmlu_cot(engine, 5, False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
