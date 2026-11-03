from mycode.instruction_tuning_evaluation import parse_mmlu, parse_gsm8k
from mycode.instruction_tuning import CustomDataset, iterate_batches
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

model_name = "meta-llama/Llama-3.1-8B"
dataset_path = ...

seq_length = 512
batch_size = 32
grad_acc_step = 4
minibatch_size = batch_size // grad_acc_step

lr = 2e-5
linear_warmup_ratio = 0.03
wd = 0.1
betas = (0.9, 0.998)
gradient_clip = 1.0

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

dataset = CustomDataset(tokenizer, dataset_path, seq_length, shuffle=True)
total_step = (len(dataset) // batch_size)

optimizer = AdamW(model.parameter(), lr=lr, betas=betas, weight_decay=wd)
scheduler = CosineAnnealingLR(optimizer, T_max=total_step)

for batch in iterate_batches(dataset, batch_size, shuffle=True):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    for step in range(grad_acc_step):
        input_ids_minibatch = input_ids[step*minibatch_size:(step+1)*minibatch_size]
        labels_minibatch = labels[step*minibatch_size:(step+1)*minibatch_size]

        logits = model(input_ids_minibatch).logits
        loss = F.cross_entropy(logits, labels_minibatch) / grad_acc_step
        loss.backward()

    clip_grad_norm_(model.parameters())
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    

