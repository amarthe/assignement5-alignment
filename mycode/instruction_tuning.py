import torch
from torch.utils.data import Dataset
import random
import json

import os
from transformers import PreTrainedTokenizerBase


PROMPT_FILE = "cs336_alignment/prompts_safety/alpaca_sft.prompt"

class CustomDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str | os.PathLike, seq_length: int, shuffle: bool, end_token: int=128001):
        self.seq_length = seq_length
        self._dataset = []

        with open(PROMPT_FILE, 'r') as f:
            prompt_template = f.read()

        tokens = []
        tokenized_prompts = []
        with open(dataset_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                prompt = prompt_template.format(instruction=sample["prompt"], response=sample["response"])
                tokenized_prompt = tokenizer(prompt)["input_ids"]
                tokenized_prompt.append(end_token)
                tokenized_prompts.append(tokenized_prompt)

        if shuffle:
            random.shuffle(tokenized_prompts)

        for tokens in tokenized_prompts:
            self._dataset.extend(tokens)

        self._dataset = torch.tensor(self._dataset)

    def __len__(self) -> int:
        return self._dataset.size(dim=0)//self.seq_length

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        start, end = i*self.seq_length, (i+1)*(self.seq_length)
        return {"input_ids": self._dataset[start:end], "labels": self._dataset[start+1:end+1]}

def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    idxs = list(range(len(dataset)))
    if shuffle:
        random.shuffle(idxs)

    for i in range(0, len(dataset), batch_size):
        out = {"input_ids": [], "labels": []}
        for j in range(i, min(i+batch_size,len(dataset))):
            out["input_ids"].append(dataset[idxs[j]]["input_ids"].unsqueeze(dim=0))
            out["labels"].append(dataset[idxs[j]]["labels"].unsqueeze(dim=0))
        yield {"input_ids": torch.cat(out["input_ids"],dim=0),
               "labels":    torch.cat(out["labels"],dim=0)}