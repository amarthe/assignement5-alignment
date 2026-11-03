import csv
from vllm import LLM, SamplingParams
from typing import Any, Callable
import json

LLM_MODEL = "meta-llama/Llama-3.1-8B"

MMLU_HS_BIOLOGY = "data/mmlu/test/high_school_biology_test.csv"

GSM8K_TEST = "data/gsm8k/test.jsonl"
GSM8K_TRAIN = "data/gsm8k/train.jsonl"

ALPACA = "data/alpaca_eval/alpaca_eval_gpt4_turbo.json"

SST = "data/simple_safety_tests/simple_safety_tests.csv"

SYSTEM_PROMPT_FILE = "cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt"
MMLU_PROMPT_FILE = "cs336_alignment/prompts_safety/mmlu_zero_shot.prompt"
GSM8K_PROMPT_FILE = "cs336_alignment/prompts_safety/gsm8k_zero_shot.prompt"
ALPACA_PROMPT_FILE = "cs336_alignment/prompts_safety/alpaca_eval_zero_shot.prompt"
SST_PROMPT_FILE = "cs336_alignment/prompts_safety/simple_safety_tests_zero_shot.prompt"

########################
# PARSING MODEL ANSWER #
########################

def parse_mmlu(model_answer: str) -> str | None:

    formatted_answer = "correct answer is "
 
    #Check if correctly formatted
    idx = model_answer.find(formatted_answer)
    if idx == -1:
        return None

    #Extract the letter
    answer = model_answer[idx + len(formatted_answer)]
    if answer not in {"A", "B", "C", "D"}:
        return None
    else:
        return answer

def parse_gsm8k(model_answer: str) -> str | None:

    words = model_answer.split()
    for word in words[::-1]:
        if word.isdigit():
            return word

    return None

def parse_alpaca(model_answer: str, example: dict[str, str]) -> dict[str, str]:

    return {"instruction": example["instruction"],
              "output": model_answer,
              "generator": LLM_MODEL.split("/")[1],
              "dataset": example["dataset"]}

def parse_sst(model_answer: str, example: dict[str, Any]) -> dict[str, str]:

    return {"prompts_final": example["instruction"],
            "output": model_answer}

#######################
#  LOADING DATA FILES #
#######################

def load_mmlu_data(data_file: str, subject: str) -> list[dict[str, Any]]:   
 
    data = []

    with open(data_file, "r") as f:
        unformatted_data = csv.reader(f,delimiter=",")
        for row in unformatted_data:
            data.append({"subject": subject,
                         "question": row[0],
                         "options": [row[1], row[2], row[3], row[4]],
                         "answer": row[5]})

    return data


def load_gsm8k_data(data_file: str) -> list[dict[str, str]]:

    #Read file
    data = []
    with open(data_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    #Structure data
    parsed_data = []
    for sample in data:
        question = sample["question"]
        answer = sample["answer"].split("####")
        reasoning = answer[0].strip()
        final_answer = answer[1].strip()
        parsed_data.append({"question": question, "reasoning": reasoning, "answer": final_answer})

    return parsed_data

def load_alpaca_data(data_file: str) -> list[dict[str, str]]:

    #Read file
    with open('data.json', 'r') as f:
        data = json.load(f)

    return data

def load_sst_data(data_file: str) -> list[dict[str, Any]]:

    data = []
    with open(data_file, "r") as f:
        data = csv.DictReader(f,delimiter=",")
    for dic in data:
        dic["instruction"] = dic.pop("prompts_final")

    return data

####################
# HELPER FUNCTIONS #
####################

# TODO peut être changer pour ne lire le fichier qu'une seule fois.
def generate_prompt(system_prompt_file: str, instruction_prompt_file: str, example: dict) -> str:

    #load the prompts
    with open(system_prompt_file, "r") as fs:
        system_prompt = fs.read()
    with open(instruction_prompt_file, "r") as fi:
        instruction_prompt = fi.read()

    #format prompts
    formatted_instruction_prompt = instruction_prompt.format(**example)
    final_prompt = system_prompt.format(instruction=formatted_instruction_prompt)

    return final_prompt

# TODO peut être changer pour ne lire le fichier qu'une seule fois.
def generate_prompt(system_prompt_file: str, instruction_prompt_file: str, example: dict) -> str:

    #load the prompts
    with open(system_prompt_file, "r") as fs:
        system_prompt = fs.read()
    with open(instruction_prompt_file, "r") as fi:
        instruction_prompt = fi.read()

    #format prompts
    formatted_instruction_prompt = instruction_prompt.format(**example)
    final_prompt = system_prompt.format(instruction=formatted_instruction_prompt)

    return final_prompt

def generate_prompts(system_prompt_file: str, instruction_prompt_file: str, examples: list[dict[str, Any]]) -> list[str]:
    prompts = []

    #load the prompts
    with open(system_prompt_file, "r") as fs:
        system_prompt = fs.read()
    with open(instruction_prompt_file, "r") as fi:
        instruction_prompt = fi.read()

    for example in examples:
        formatted_instruction_prompt = instruction_prompt.format(**example)
        final_prompt = system_prompt.format(instruction=formatted_instruction_prompt)
        prompts.append(final_prompt)

    return prompts

def evaluate_answers(model_outputs: list[str], examples: list[dict[str, Any]]) -> tuple[int, int]:

    format_reward, answer_reward = [], []
    sorted_answers = {"correct": [], "uncorrect": [], "unparsed":[]}

    for output, example in zip(model_outputs, examples):
        output_answer = parse_mmlu(output)

        if output_answer == None:
            format_reward.append(0)
            answer_reward.append(0)
            sorted_answers["unparsed"].append(output)
        else:
            format_reward.append(1)
            if output_answer == example["answer"]:
                sorted_answers["correct"].append(output)
            else:
                sorted_answers["uncorrect"].append(output)
            answer_reward.append(1 if output_answer == example["answer"] else 0)

    nb_formatted_answer = sum(format_reward)
    nb_correct_answer = sum(answer_reward)

    return nb_formatted_answer, nb_correct_answer, sorted_answers


def generate_answers(model_server: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    outputs = model_server.generate(prompts, sampling_params)
    model_outputs = []

    for output in outputs:
        model_outputs.append(output.outputs[0].text)

    return model_outputs

def evaluate_answers(model_outputs: list[str], examples: list[dict[str, Any]], answer_parser: Callable) -> tuple[int, int, dict[str, list]]:

    format_reward, answer_reward = [], []
    sorted_answers = {"correct": [], "uncorrect": [], "unparsed":[]}

    for output, example in zip(model_outputs, examples):
        output_answer = answer_parser(output)

        if output_answer == None:
            format_reward.append(0)
            answer_reward.append(0)
            sorted_answers["unparsed"].append(output)
        else:
            format_reward.append(1)
            if output_answer == example["answer"]:
                sorted_answers["correct"].append(output)
            else:
                sorted_answers["uncorrect"].append(output)
            answer_reward.append(1 if output_answer == example["answer"] else 0)

    nb_formatted_answer = sum(format_reward)
    nb_correct_answer = sum(answer_reward)

    return nb_formatted_answer, nb_correct_answer, sorted_answers

