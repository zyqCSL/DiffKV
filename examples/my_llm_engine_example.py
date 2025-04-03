import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    # return [
    #     ("A robot may not injure a human being",
    #      SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1,
    #                     max_tokens=4000)),
    #     ("To be or not to be,",
    #      SamplingParams(temperature=0.8, top_k=1, presence_penalty=0.2)),
    #     ("What is the meaning of life?",
    #      SamplingParams(n=1, temperature=0.2, top_p=0.95,
    #                     frequency_penalty=0.1)),
    #     ("It is only with the heart that one can see rightly",
    #      SamplingParams(n=1, best_of=1, use_beam_search=False,
    #                     temperature=0.0)),
    # ]
    
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1,
                        max_tokens=4000)),
        ("To be or not to be,",
         SamplingParams(temperature=0.0, top_k=1, max_tokens=4000)),
        ("What is the meaning of life?",
         SamplingParams(n=1, temperature=0.0, max_tokens=4000)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=1, temperature=0.0, max_tokens=4000)),
        ("Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
         SamplingParams(n=1, temperature=0.0, max_tokens=500))
    ]

def create_seq_model_tasks() -> List[Tuple[str, SamplingParams]]:
    prompt = "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , " \
        "commonly referred to as Valkyria Chronicles III outside Japan , " \
        "is a tactical role playing video game developed by Sega and Media.Vision for the PlayStation Portable ."
    generated = " Released in January 2011 in Japan , it is the third game in the Valkyria series . " \
            "Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , " \
            "the story runs parallel to the first game and follows the \" Nameless \" , " \
            "a penal military unit serving the nation of Gallia during the Second Europan War " \
            "who perform secret black operations and are pitted against the Imperial unit \" Calamaty Raven \" ." 
    return [
        (prompt,
         SamplingParams(
            #  model_seq=True, truth=prompt + generated,
                        temperature=0.0, logprobs=1, prompt_logprobs=1,
                        max_tokens=4096)),
    ]

def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    
    quant_configs = [8, 8, 8, 8]
    compress_configs = [0.0, 0.0]

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params,
                               quant_configs, compress_configs)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    engine.log_stats = False
    
    tasks = []
    # text generation
    test_prompts = create_test_prompts()
    # tasks += test_prompts[:1]
    # tasks += test_prompts[:2]
    tasks += test_prompts
    
    # seq modeling
    tasks += create_seq_model_tasks()
    
    
    process_requests(engine, tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
