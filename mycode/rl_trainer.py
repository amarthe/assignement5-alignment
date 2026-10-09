from grpo import get_model_and_tokenizer, tokenize_prompt_and_output, get_response_log_probs, compute_rollout_rewards, compute_group_normalized_rewards, compute_policy_gradient_loss, aggregate_loss_across_microbatch, grpo_train_step_standard_on_policy, TrainingArgs
from mycode.inference import load_data, extract_gt_answers, sample_data, answer_questions, generate_prompts, generate_model_answers
from cs336_alignment.vllm_utils import VLLMServer
from tqdm.auto import tqdm
import logging
import wandb
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from dataclasses import dataclass
import sys


@dataclass
class TrainingArgs():

    # Model
    model_name: str = "allenai/OLMo-2-0425-1B"

    # Datasets
    train_dataset: str = "data/gsm8k/train.jsonl"
    test_dataset: str = "data/gsm8k/test.jsonl"

    # Prompt type
    reward_fn = r1_zero_reward_fn
    prompt_type: str = "few_shot"

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
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0

    # Loggingg
    log_folder: str = None
    save_directory: str = None
    use_wandb: bool = False
    wandb_project: str = "LLM-RL-Experiments"
    run_name: str = "defaul_run"

    device: str = "cuda:0"

class RLTrainer():

    def  __init__(self, args: TrainingArgs):

        self.args = args

    def init(self):

        # Initialize Inference setup
        self.vllm_server = VLLMServer(self.args.model_name, gpu=1)
        self.vllm_server.start()
        self.vllm_server.init_weight_sync(self.args.device)

        # Initialize Training Objects
        self.model, self.tokenizer = get_model_and_tokenizer(self.args.model_name, self.args.device)
        self.optimizer = self.args.optimizer_class(params=self.model.parameters(), lr=self.args.learning_rate, betas=self.args.betas, weight_decay=self.args.weight_decay)

        # Load datasets
        self.train_dataset = load_data(self.args.train_dataset)
        self.test_dataset = load_data(self.args.test_dataset)

        # Init logging
        self.wandb_run, self.logger = self._init_logging()

    def _init_logging(self):

        # Init Wandb
        if self.args.use_wandb:
            wandb.login()
            #wandb_config = {...}
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
            pass



