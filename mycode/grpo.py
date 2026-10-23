from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass, field
from typing import Callable, Literal
from einops import rearrange
from torch.optim import Optimizer
import numpy as np
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable

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
    train_dataset: str = "data/gsm8k/train.jsonl"
    test_dataset: str = "data/gsm8k/test.jsonl"

    # Prompt type
    reward_fn: Callable = field(default_factory=lambda: r1_zero_reward_fn)
    prompt_type: str = "few_shot"
    stop_tokens: list[str] = field(default_factory=lambda: ["</answer>"])

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
    inference_gpu: int = 1

    # Optimization parameters
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 64
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0

    # Loggingg
    log_folder: str = None
    save_directory: str = None
    use_wandb: bool = False
    wandb_project: str = "LLM-RL-Experiments"
    run_name: str = "defaul_run"

    # Misc
    seed: int = 42
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
    logits = model(input_ids.to(model.device)).logits
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
    elif baseline == "none":
        baseline_values = torch.zeros(raw_rewards.size())
    else:
        raise NotImplementedError()

    baselined_rewards = raw_rewards - baseline_values

    if advantage_normalizer == "std":
        normalizer_value = torch.std(raw_rewards, dim=-1, keepdim=True) + advantage_eps
    elif advantage_normalizer == "none":
        normalizer_value = torch.ones(raw_rewards.size())
    elif advantage_normalizer == "mean":
        normalizer_value = torch.mean(raw_rewards, dim=-1, keepdim=True) + advantage_eps
    else:
        raise NotImplementedError

    advantages = torch.flatten(baselined_rewards/normalizer_value)

    #TODO later
    logs = {}

    return advantages, logs

def compute_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        importance_reweighting_method: Literal["none", "noclip", "grpo", "cispo", "gspo"] = "none",
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        response_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    raw_rewards_or_advantages = raw_rewards_or_advantages.view(raw_rewards_or_advantages.size()[0], 1)

    if importance_reweighting_method == "none":
        policy_gradient_loss = -policy_log_probs * raw_rewards_or_advantages
    
    elif importance_reweighting_method == "noclip":
        assert(old_log_probs is not None)

        importance_weight = torch.exp(policy_log_probs - old_log_probs)
        policy_gradient_loss = -raw_rewards_or_advantages * importance_weight

    elif importance_reweighting_method == "grpo":
        assert(old_log_probs is not None)
        assert(cliprange is not None)

        importance_weight = torch.exp(policy_log_probs - old_log_probs)
        clipped_weights = torch.clip(importance_weight,1-cliprange, 1+cliprange)

        weighted_advantages = raw_rewards_or_advantages * importance_weight
        clipped_weighted_advantages = raw_rewards_or_advantages * clipped_weights

        policy_gradient_loss = -torch.minimum(clipped_weighted_advantages, weighted_advantages)

    elif importance_reweighting_method == "cispo":
        raise NotImplementedError

    elif importance_reweighting_method == "gspo":
        assert(old_log_probs is not None)
        assert(cliprange is not None)
        assert(response_mask is not None)

        response_lengths = response_mask.sum(dim=-1,keepdim=True)
        averaged_importance_weight = torch.exp(torch.sum((policy_log_probs - old_log_probs)*response_mask, dim=-1, keepdim=True) / response_lengths)
        assert(averaged_importance_weight.size() == raw_rewards_or_advantages.size())

        clipped_averaged_importance_weight = averaged_importance_weight.clip(1-cliprange, 1+cliprange)
        assert(clipped_averaged_importance_weight.size() == raw_rewards_or_advantages.size())

        policy_gradient_loss = - torch.minimum(averaged_importance_weight * raw_rewards_or_advantages, clipped_averaged_importance_weight * raw_rewards_or_advantages).tile([1, policy_log_probs.size()[1]])

    else:
        raise NotImplementedError

    #TODO add the importance weights
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

def grpo_train_step(
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
        #» Reward normalization
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
    total_loss = torch.tensor(0., requires_grad=False, device=model.device)

    # Advantage computation
    rewards, reward_logs = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, advantage_logs = compute_group_normalized_rewards(rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    # Tokenize prompt and responses
    tokenized_data = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = tokenized_data["input_ids"], tokenized_data["labels"], tokenized_data["response_mask"]

    # Gradient accumulation
    for i in range(0, len(repeated_prompts), microbatch_size):

        # Extract microbatch
        inputs_microbatch = input_ids[i:i+microbatch_size].to(model.device)
        labels_microbatch = labels[i:i+microbatch_size].to(model.device)
        advantages_microbatch = advantages[i:i+microbatch_size].to(model.device)
        mask_microbatch = response_mask[i:i+microbatch_size].to(model.device)
        old_log_probs_microbatch = old_log_probs[i:i+microbatch_size].to(model.device) if old_log_probs is not None else None

        # Not tested yet
        # if torch.max(advantages_microbatch.abs()).value[0] == 0.:
        #     continue

        # Policy log probs computation
        policy_log_probs_and_entropy = get_response_log_probs(model, inputs_microbatch, labels_microbatch, return_token_entropy=True)
        policy_log_probs, entropy = policy_log_probs_and_entropy["log_probs"], policy_log_probs_and_entropy["token_entropy"]

        #Loss computation
        per_token_loss, loss_logs = compute_policy_gradient_loss(advantages_microbatch, policy_log_probs, importance_reweighting_method, old_log_probs_microbatch, cliprange, mask_microbatch)
        loss = aggregate_loss_across_microbatch(per_token_loss, mask_microbatch, loss_normalization, normalization_constant) 
        if loss_normalization == "sequence": # For GRPO, also average accross batch
            loss = loss * (microbatch_size / len(repeated_prompts))

        # Accumulate gradients
        loss.backward()
        total_loss += loss.to(model.device)

    # Optimizer Step
    grad_norm = torch.sqrt(sum([torch.norm(p.grad).square() for p in model.parameters()]))
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    # Logs
    logs = {
        "loss": total_loss,
        "grad_norm": grad_norm,
        "mean_token_entropy": entropy.mean(),
        "mean_reward": reward_logs["mean_reward"],
        "mean_format_reward": reward_logs["mean_format_reward"],
        "mean_response_length": np.mean([len(r) for r in rollout_responses]),
        "mean_tokenized_response_length": response_mask.sum(dim=-1).mean(dtype=torch.float)
    }

    return total_loss, logs



