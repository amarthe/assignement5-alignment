"""RLVR training with one Hugging Face model for rollouts and optimization."""

from mycode.grpo import get_model_and_tokenizer, compute_rollout_rewards, grpo_train_step
from mycode.inference import load_data, extract_gt_answers, sample_data, generate_prompts, generate_model_answers_hf
from tqdm.auto import tqdm
import logging
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from dataclasses import dataclass, field
import sys
import random
from pathlib import Path
from typing import Callable

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
    inference_batch_size: int = 2

    # Optimization parameters
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 128
    gradient_checkpointing: bool = True
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
    device: str = "cuda:0"

class RLTrainer():

    def  __init__(self, args: TrainingArgs):

        self.args = args

    def init(self):

        args = self.args

        if args.group_size < 2 or args.rollout_batch_size <= 0 or args.rollout_batch_size % args.group_size:
            raise ValueError("rollout_batch_size must be positive and divisible by group_size (at least 2).")
        if args.gradient_accumulation_steps <= 0 or args.rollout_batch_size % args.gradient_accumulation_steps:
            raise ValueError("rollout_batch_size must be divisible by positive gradient_accumulation_steps.")
        if args.inference_batch_size <= 0 or args.eval_step <= 0:
            raise ValueError("inference_batch_size and eval_step must be positive.")
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Initialize some variables
        self.num_prompts_per_batch = args.rollout_batch_size // args.group_size
        self.sampling_params = {
            "temperature": args.sampling_temperature,
            "max_tokens": args.sampling_max_tokens,
            "seed": args.seed,
            "n":args.group_size,
            "stop": args.stop_tokens,
            "include_stop_str_in_output": True,
        }
        self.eval_sampling_params = {
            "temperature": args.sampling_temperature,
            "max_tokens": args.sampling_max_tokens,
            "seed": args.seed,
            "n": 1,
            "stop": args.stop_tokens,
            "include_stop_str_in_output": True,
        }

        # One model handles both generation and optimization. The HF generation
        # helper enables caching for rollouts and restores the training mode.
        self.model, self.tokenizer = get_model_and_tokenizer(args.model_name, args.device)
        self.model.config.use_cache = False
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.optimizer = args.optimizer_class(params=self.model.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)

        # Load datasets
        self.train_dataset = load_data(args.train_dataset)
        self.test_dataset = load_data(args.test_dataset)

        # Init logging
        self.wandb_run, self.logger = self._init_logging()

    def _init_logging(self):

        # Init Wandb
        if self.args.use_wandb:
            import wandb

            wandb.login()
            #wandb_config = {...} #TODO
            wandb_run = wandb.init(project=self.args.wandb_project, name=self.args.run_name)
        else:
            wandb_run = None

        # Init Logging
        logger = logging.getLogger("RL Trainer")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if self.args.log_folder is not None:
            Path(self.args.log_folder).mkdir(parents=True, exist_ok=True)
            filename = f"{self.args.run_name}.log"
            self.log_file_path = f"{self.args.log_folder}/{filename}"

            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return wandb_run, logger

    def train(self):

        args = self.args

        progress_bar = tqdm(range(args.num_rollout_steps))
        for step in progress_bar:

            # Fetch questions
            questions = sample_data(self.train_dataset, n=self.num_prompts_per_batch)
            repeated_questions = []
            for question in questions:
                repeated_questions += [question] * args.group_size

            # Generate prompts, ground truths and rollouts
            prompts = generate_prompts(questions, args.prompt_type)
            repeated_prompts = generate_prompts(repeated_questions, args.prompt_type)
            rollout_responses = generate_model_answers_hf(
                self.model, self.tokenizer, prompts,
                {**self.sampling_params, "seed": args.seed + step},
                batch_size=args.inference_batch_size,
            )
            repeated_ground_truths = extract_gt_answers(repeated_questions)

            if len(rollout_responses) != len(repeated_prompts):
                raise ValueError("Expected group_size rollouts per prompt.")

            self.model.train()

            loss, log = grpo_train_step(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=args.reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size
                #Other parameters are defaults for now
            )

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            # Do not retain training graphs or GPU tensors during the next rollout.
            log = {key: value.detach().item() if isinstance(value, torch.Tensor) else value
                   for key, value in log.items()}
            del loss
            if args.use_wandb:
                self.wandb_run.log(log)

            #log and evaluate
            if (step + 1) % args.eval_step == 0:

                # Fetch questions
                questions = sample_data(self.test_dataset, n=args.n_eval_examples)
                prompts = generate_prompts(questions, args.prompt_type)

                # Evaluate model
                answers = generate_model_answers_hf(
                    self.model, self.tokenizer, prompts, self.eval_sampling_params,
                    batch_size=args.inference_batch_size,
                )
                ground_truths = extract_gt_answers(questions)
                _, eval_reward_log = compute_rollout_rewards(args.reward_fn, answers, ground_truths)
                mean_reward, mean_format_reward = eval_reward_log["mean_reward"], eval_reward_log["mean_format_reward"]
                self.logger.info("Step %s: reward: %.4f\t format_reward: %.4f",
                                 step + 1, mean_reward, mean_format_reward)

                if args.use_wandb:
                    self.wandb_run.log({
                        "eval_mean_reward": mean_reward,
                        "eval_mean_format_reward": mean_format_reward
                    })
