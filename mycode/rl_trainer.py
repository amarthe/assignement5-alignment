from grpo import get_model_and_tokenizer, tokenize_prompt_and_output, get_response_log_probs, compute_rollout_rewards, compute_group_normalized_rewards, compute_policy_gradient_loss, aggregate_loss_across_microbatch, grpo_train_step, TrainingArgs
from mycode.inference import load_data, extract_gt_answers, sample_data, answer_questions, generate_prompts, generate_model_answers
from cs336_alignment.vllm_utils import VLLMServer
from tqdm.auto import tqdm
import logging
import wandb
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from dataclasses import dataclass, field
import sys
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

class RLTrainer():

    def  __init__(self, args: TrainingArgs):

        self.args = args
        #TODO mettre tous les éléments comme attributs de la classes

    def init(self):

        args = self.args

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

        # Initialize Inference setup
        self.vllm_server = VLLMServer(args.model_name, gpu=args.inference_gpu)
        self.vllm_server.start()
        self.vllm_server.init_weight_sync(args.device)

        # Initialize Training Objects
        self.model, self.tokenizer = get_model_and_tokenizer(args.model_name, args.device)
        self.optimizer = args.optimizer_class(params=self.model.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)

        # Load datasets
        self.train_dataset = load_data(args.train_dataset)
        self.test_dataset = load_data(args.test_dataset)

        # Init logging
        self.wandb_run, self.logger = self._init_logging()

    def _init_logging(self):

        # Init Wandb
        if self.args.use_wandb:
            wandb.login()
            #wandb_config = {...} #TODO
            wandb_run = wandb.init(project=self.args.wandb_projecrunt, name=self.args.wandb_run_name)
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
            filename = f"{self.args.run_name}.log"
            self.log_file_path = f"{self.args.log_folder}/{filename}"

            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return wandb_run, logger

    def train(self):

        progress_bar = tqdm(range(self.args.num_rollout_steps))
        for step in progress_bar:

            # Fetch questions
            questions = sample_data(self.train_dataset, n=self.num_prompts_per_batch)
            repeated_questions = []
            for question in questions:
                repeated_questions += [question] * self.args.group_size

            # Generate prompts, ground truths and rollouts
            prompts = generate_prompts(questions, self.args.prompt_type)
            repeated_prompts = generate_prompts(repeated_questions, self.args.prompt_type)
            rollout_responses = generate_model_answers(self.vllm_server, prompts, self.sampling_params)
            repeated_ground_truths = extract_gt_answers(repeated_questions)
                    
            loss, log = grpo_train_step(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                max_grad_norm=self.args.max_grad_norm,
                reward_fn=self.args.reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=self.args.group_size
                #Other parameters are defaults for now
            )

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            if self.args.use_wandb:
                self.wandb_run.log(log)
            self.vllm_server.sync_policy_weights(self.model)

            #log and evaluate
            if step % self.args.eval_step:

                # Fetch questions
                questions = sample_data(self.test_dataset, n=self.args.n_eval_examples)
                prompts = generate_prompts(questions, self.args.prompt_type)

                # Evaluate model
                answers = generate_model_answers(self.vllm_server, prompts, self.eval_sampling_params) 
                ground_truths = extract_gt_answers(questions)
                _, eval_reward_log = compute_rollout_rewards(self.args.reward_fn, answers, ground_truths)
                mean_reward, mean_format_reward = eval_reward_log["mean_reward"], eval_reward_log["mean_format_reward"]
                print(f"Step {step}: reward: {mean_reward}\t format_reward: {mean_format_reward}")

                if self.args.use_wandb:
                    self.wandb_run.log({
                        "eval_mean_reward": mean_reward,
                        "eval_mean_format_reward": mean_format_reward
                    })
