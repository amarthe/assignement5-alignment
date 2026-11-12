import gzip
import json
import os
from typing import Any
import torch
from transformers import PreTrainedTokenizerBase

DATA_FOLDER = "data/hh"
DATA_FILES = ["data/hh/harmless-base.jsonl.gz", "data/hh/helpful-base.jsonl.gz", "data/hh/helpful-online.jsonl.gz", "data/hh/helpful-rejection-sampled.jsonl.gz"]
PROMPT_FILE = "cs336_alignment/prompts_safety/alpaca_sft.prompt"

def format_prompt(instruction: str, answer: str) -> str:
    with open(PROMPT_FILE) as f:
        prompt = f.read()

    return prompt.format(instruction=instruction, response=answer)

def load_data(filepath: str | os.PathLike) -> list[dict]:

    data = []

    with gzip.open(filepath) as f:
        for line in f:
            data.append(json.loads(line))

    return data

def filter_multiturn_data(data: list[dict[str, str]]) -> list[dict]:

    filtered_data = []

    for sample in data:
            human_count = sample["chosen"].count("Human: ")
            assistant_count = sample["chosen"].count("Assistant: ")
            if assistant_count == 1 and human_count == 1:
                 filtered_data.append(sample)

    return filtered_data

def format_data(data: list[dict[str, Any]], origin_file: str) -> list[dict[str, Any]]:

    formatted_data = []

    for sample in data:
        formatted_sample = {}

        instruction_offset = len("\n\nHuman: ")
        div_idx = sample["chosen"].find("\n\nAssistant: ")
        answer_offset = div_idx + len("\n\nAssistant: ")

        formatted_sample["instruction"] = sample["chosen"][instruction_offset:div_idx]
        formatted_sample["chosen"] = sample["chosen"][answer_offset:]  
        formatted_sample["rejected"] = sample["rejected"][answer_offset:]
        formatted_sample["file"]=  origin_file

        formatted_data.append(formatted_sample)

    return formatted_data

def preprocess(datafiles: list[str]=DATA_FILES) -> list[dict]:

    data = []

    for file in datafiles:
        filename = file.split("/")[-1].split(".")[0]
        samples = load_data(file)
        samples = filter_multiturn_data(samples)
        samples = format_data(samples, filename)
        data.extend(samples)

    return data


def get_full_log_prob(lm: torch.nn.Module, token_sequence: torch.Tensor, mask: torch.Tensor):
    input_ids = token_sequence[...,:-1]
    labels = token_sequence[...,1:]

    logits = lm(input_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    labels_log_probs = torch.gather(log_probs, -1 ,labels.unsqueeze(-1)).squeeze(-1)
    return torch.sum(labels_log_probs * mask, dim=-1)

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:

    eos_token = tokenizer.get_vocab()["<eos>"]

    prompt_chosen = format_prompt(prompt, response_chosen)
    prompt_rejected = format_prompt(prompt, response_rejected)
    tokenized_chosen = tokenizer(prompt_chosen)["input_ids"] + [eos_token]
    tokenized_rejected = tokenizer(prompt_rejected)["input_ids"] + [eos_token]

    tokenized_answer_chosen = tokenizer(response_chosen)["input_ids"] + [eos_token]
    tokenized_answer_rejected = tokenizer(response_rejected)["input_ids"] + [eos_token]
    len_answer_chosen = len(tokenized_answer_chosen)
    len_answer_rejected = len(tokenized_answer_rejected)
    len_prompt = len(tokenized_chosen) - len_answer_chosen

    chosen_mask = ([0] * (len_prompt - 1)) + ([1] * (len_answer_chosen))
    rejected_mask = ([0] * (len_prompt - 1)) + ([1] * (len_answer_rejected))

    y_c = torch.tensor(tokenized_chosen).unsqueeze(0)
    y_r = torch.tensor(tokenized_rejected).unsqueeze(0)
    mask_c = torch.tensor(chosen_mask)
    mask_r = torch.tensor(rejected_mask)

    log_prob_train_yc = get_full_log_prob(lm, y_c, mask_c)
    log_prob_ref_yc = get_full_log_prob(lm_ref, y_c, mask_c)
    reward_c = (log_prob_train_yc - log_prob_ref_yc)*beta
    print(reward_c)

    log_prob_train_yr = get_full_log_prob(lm, y_r, mask_r)
    log_prob_ref_yr = get_full_log_prob(lm_ref, y_r, mask_r)
    reward_r = (log_prob_train_yr - log_prob_ref_yr)*beta
    print(reward_r)

    return -torch.log(torch.special.expit(reward_c - reward_r))


        