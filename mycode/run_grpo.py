from grpo import get_model_and_tokenizer, tokenize_prompt_and_output, get_response_log_probs, compute_rollout_rewards, compute_group_normalized_rewards, compute_policy_gradient_loss, aggregate_loss_across_microbatch, grpo_train_step_standard_on_policy, TrainingArgs
from mycode.inference import load_data, extract_gt_answers, sample_data, answer_questions, generate_prompts, generate_model_answers
from cs336_alignment.vllm_utils import VLLMServer
from tqdm.auto import tqdm

args = TrainingArgs()
num_prompts_per_batch = args.rollout_batch_size // args.group_size

# Initialize Inference setup
vllm_server = VLLMServer(args.model_name, gpu=1)
vllm_server.start()
vllm_server.init_weight_sync(args.device)

# Initialize Training setup
model, tokenizer = get_model_and_tokenizer(args.model_name, args.device)
optimizer = args.optimizer_class(params=model.parameters(), lr=args.learning_rate, betas=args.be
res_question_only = eval_model(vllm_server, data, "question_only", sampling_params)
res_zero_shot = eval_model(vllm_server, data, "zero_shot", sampling_params)
res_few_shot = eval_model(vllm_server, data, "few_shot", sampling_params)

tas, weight_decay=args.weight_decay)

# Load datasets
train_dataset = load_data(args.train_dataset)
test_dataset = load_data(args.test_dataset)

# Initialize logging
# TODO log file + wandb. TODO when we are the sure everything else is working

progress_bar = tqdm(range(args.num_rollout_steps))
for step in progress_bar:

    # Fetch questions
    questions = sample_data(train_dataset, n=num_prompts_per_batch)
    repeated_questions = []
    for question in questions:
        repeated_questions += [question] * args.group_size
    assert(len(repeated_questions) == args.rollout_batch_size)

    # Generate prompts, ground truths and rollouts
    repeated_prompts = generate_prompts(repeated_questions, args.prompt_type)
    rollout_responses = generate_model_answers(repeated_prompts)
    repeated_ground_truths = extract_gt_answers(repeated_questions)

    # Perform 1 gradient step
    loss, log = grpo_train_step_standard_on_policy(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        reward_fn=args.reward_fn,
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=args.group_size
        #Other parameters are defaults for now
    )
    vllm_server.sync_policy_weights(model)
    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    #log and evaluate
    if step % args.eval_step:

        questions = sample_data(test_dataset, n=args.n_eval_examples)
        prompts = generate_prompts(questions, args.prompt_type)

        answers = answer_questions(prompts, ...)
        ground_truths = extract_gt_answers(questions)
        _, eval_reward_log = compute_rollout_rewards(args.reward_fn, answers, ground_truths)
        mean_reward, mean_format_reward = eval_reward_log["mean_reward"], eval_reward_log["mean_format_reward"]
        print(f"Step {step}: reward: {mean_reward}\t format_reward: {mean_format_reward}")
