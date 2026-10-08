from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
from typing import Callable, Literal
from einops import rearrange
from torch.optim import Optimizer
import numpy as np
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PADDING_TOKEN = 0

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

@dataclass
class TrainingArgs():

    # Model
    model_name: str = "allenai/OLMo-2-0425-1B"

    # Datasets
    train_dataset: str = "/data/gsm8k/train.jsonl"
    test_dataset: str = "/data/gsm8k/test.jsonl"

    # 
    reward_fn = r1_zero_reward_fn
    prompt_type = "few_shot"

    # Number of data
    n_train_examples: int = 6400
    num_rollout_steps: int = 200

    # Batch parameters
    rollout_batch_size: int = 256
    train_batch_size: int = 256
    group_size: int = 8

    # Eval Parameters
    n_eval_examples: int = 1024
    eval_step: int = 40

    # Inference parameters
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 512

    # Optimization parameters
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 32
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0

    # Misc
    log_file: str = None
    save_directory: str = None
    device = "cuda:0"

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    tokenized_prompts = tokenizer(prompt_strs)["input_ids"]
    tokenized_outputs = tokenizer(output_strs)["input_ids"]

    concatenated_tokens = [tp + to for (tp, to) in zip(tokenized_prompts, tokenized_outputs)]
    prompt_and_output_lens = [len(t) for t in concatenated_tokens]
    pad_length = max(prompt_and_output_lens)-1

    input_ids = torch.tensor([(t + [PADDING_TOKEN]*(pad_length - len(t) + 1))[:-1] for t in concatenated_tokens]) #Convoluted because we need to keep the last token of output except when it the max size"
    labels = torch.tensor([t[1:] + [PADDING_TOKEN]*(pad_length - len(t) + 1) for t in concatenated_tokens])
    response_mask = torch.tensor([[0]*(len(tp)-1) + [1]*(len(to)) + [0]*(pad_length-len(tp)-len(to)+1) for (tp, to) in zip(tokenized_prompts, tokenized_outputs)])

    return {"input_ids": input_ids, "labels": labels, "response_mask":response_mask}

def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool=False) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = logits_log_softmax.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        token_entropy = -(torch.softmax(logits, dim=-1) * logits_log_softmax).sum(-1)
        return {"log_probs": log_probs, "token_entropy":token_entropy}
    else:
        return {"log_probs": log_probs}

def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str]
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        rewards.append(reward_fn(response, gt))
    raw_rewards = torch.tensor([r["reward"] for r in rewards])

    #Logs
    nb_rollout = len(rollout_responses)
    mean_reward = raw_rewards.mean().item()
    mean_format_reward = torch.mean(torch.tensor([r["format_reward"] for r in rewards])).item()
    logs = {"nb_rollout": nb_rollout, "mean_reward": mean_reward, "mean_format_reward": mean_format_reward}
    
    return raw_rewards, logs

def compute_group_normalized_rewards(
        raw_rewards: torch.Tensor,
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std"
) -> tuple[torch.Tensor, dict[str, float]]:
    
    raw_rewards = rearrange(raw_rewards, "(n_prompts group_size) -> n_prompts group_size", group_size=group_size)

    if baseline == "mean":
        baseline_values = torch.mean(raw_rewards, dim=-1, keepdim=True)
    else:
        raise NotImplementedError

    baselined_rewards = raw_rewards - baseline_values

    if advantage_normalizer == "std":
        normalizer_value = torch.std(raw_rewards, dim=-1, keepdim=True) + advantage_eps
    else:
        raise NotImplementedError

    advantages = torch.flatten(baselined_rewards/normalizer_value)

    #TODO later
    logs = {}

    return advantages, logs

def compute_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        response_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if importance_reweighting_method != "none":
        raise NotImplementedError

    raw_rewards_or_advantages = raw_rewards_or_advantages.view(raw_rewards_or_advantages.size()[0], 1)

    policy_gradient_loss = -policy_log_probs * raw_rewards_or_advantages

    #TODO plus tard
    logs = {}

    return policy_gradient_loss, logs

def aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss: torch.Tensor,
        mask: torch.Tensor,
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None
) -> torch.Tensor:

    batch_size = per_token_policy_gradient_loss.size(0)
    if loss_normalization == "sequence":
        normalization = mask.sum(dim=-1, keepdim=True) * batch_size
    elif loss_normalization == "constant":
        assert(normalization_constant is not None)
        normalization = normalization_constant
    else:
        raise NotImplementedError

    loss = torch.sum(per_token_policy_gradient_loss * mask / normalization)
    return loss

def grpo_train_step_standard_on_policy(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optimizer,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        reward_fn: Callable[[str, str], dict[str, float]],
        repeated_prompts: list[str],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        # Reward normalization
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        # Importance reweighting and clipping
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        # Loss normalization
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    total_loss = torch.tensor(0., requires_grad=False)

    # Advantage computation
    rewards, reward_logs = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantage_logs = compute_group_normalized_rewards(rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    # Tokenize prompt and responses
    tokenized_data = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = tokenized_data["input_ids"], tokenized_data["labels"], tokenized_data["response_mask"]

    # Gradient accumulation
    for i in range(0, len(repeated_prompts), microbatch_size):

        # Extract microbatch
        inputs_microbatch = input_ids[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]
        advantages_microbatch = advantages[i:i+microbatch_size]
        mask_microbatch = response_mask[i:i+microbatch_size]

        # Policy log probs computation
        policy_log_probs_and_entropy = get_response_log_probs(model, inputs_microbatch, labels_microbatch, return_token_entropy=True)
        policy_log_probs, entropy = policy_log_probs_and_entropy["log_probs"], policy_log_probs_and_entropy["token_entropy"]

        #Loss computation
        per_token_loss, loss_logs = compute_policy_gradient_loss(advantages_microbatch, policy_log_probs, importance_reweighting_method, old_log_probs, cliprange, mask_microbatch)
        loss = aggregate_loss_across_microbatch(per_token_loss, mask_microbatch, loss_normalization, normalization_constant) * (microbatch_size / len(repeated_prompts))

        # Accumulate gradients
        loss.backward()
        total_loss += loss

    # Optimizer Step
    grad_norm = torch.nn.utils.get_total_norm(model.parameters())
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    # Logs
    logs = {
        "loss": total_loss,
        "grad_norm": grad_norm,
        "token_entropy": entropy,
        "mean_reward": reward_logs["mean_reward"],
        "mean_format_reward": reward_logs["mean_format_reward"],
        "mean_response_length": np.mean([len(r) for r in rollout_responses]),
        "mean_tokenized_response_length": response_mask.sum(dim=-1).mean(dtype=torch.float)
    }

    return total_loss, logs



