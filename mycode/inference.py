import json
import random
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn, extract_answer

GSM8K_TEST_PATH = "data/gsm8k/test.jsonl"
GSM8K_TRAIN_PATH = "data/gsm8k/train.jsonl"

BATCH_SIZE = 32

def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def parse_gsm8k_data(samples):
    parsed_data = []
    for sample in samples:
        question = sample["question"]
        answer = sample["answer"].split("####")
        reasoning = answer[0].strip()
        final_answer = answer[1].strip()
        parsed_data.append({"question": question, "reasoning": reasoning, "answer": final_answer})
    return parsed_data

def extract_gt_answers(samples):
    gt_answers = []
    for sample in samples:
        answer = sample["answer"].split("####")
        gt_answers.append(answer[1].strip())
    return gt_answers
 

def sample_data(data, n=1):
    return random.sample(data, n)

def generate_question_only_prompt(sample):
    return f"{sample['question']} Please put your final answer within \\boxed{{}}."

def generate_zero_shot_prompt(sample):
    return f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {sample['question']}\nAssistant: <think>"

def generate_few_shot_prompt(sample):
    return f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAssistant: <think> There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6. </think> <answer> 6 </answer>\nUser: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAssistant: <think> There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5. </think> <answer> 5 </answer>\nUser: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAssistant: <think> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39. </think> <answer> 39 </answer>\nUser: {sample['question']}\nAssistant: <think>"

def generate_prompts(samples, prompt_type):
    if prompt_type == "question_only":
        return [generate_question_only_prompt(sample) for sample in samples]
    elif prompt_type == "zero_shot":
        return [generate_zero_shot_prompt(sample) for sample in samples]
    elif prompt_type == "few_shot":
        return [generate_few_shot_prompt(sample) for sample in samples]
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

def generate_model_answers(model, prompts, sampling_params, batch_size=BATCH_SIZE):
    answers = model.generate_completions(prompts, sampling_params, batch_size=batch_size)
    return [answer.text for answer in answers]

def answer_questions(model, samples, prompt_type, sampling_params):
    prompts = generate_prompts(samples, prompt_type)
    return generate_model_answers(model, prompts, sampling_params)

def eval_answers(model_answers, gt_answers, prompt_type):
    if prompt_type == "question_only":
        return [question_only_reward_fn(model_answer, gt_answer) for model_answer, gt_answer in zip(model_answers, gt_answers)]
    if prompt_type in ["zero_shot", "few_shot"]:
        return [r1_zero_reward_fn(model_answer, gt_answer) for model_answer, gt_answer in zip(model_answers, gt_answers)]
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

def eval_model(model, samples, prompt_type, sampling_params):
    results = []
    model_answers = answer_questions(model, samples, prompt_type, sampling_params)
    gt_answers = [d["answer"] for d in parse_gsm8k_data(samples)]
    #print(f"Model answer: {model_answer}\nGround truth answer: {gt_answer}")
    results = eval_answers(model_answers, gt_answers, prompt_type=prompt_type)
        
    return results

def extract_model_answer(response):
    """
    stolen from : https://github.com/thevivekpandey/stanford-cs336-assignment-5/blob/main/cs336_alignment/prompting_baselines.py

    Display only -- mirrors how the reward fns locate the answer.

    Scoring comes from reward_fn, never from this. r1_zero puts the answer in
    <answer>...</answer>; question_only puts it in \\boxed{...}.
    """
    if "<answer>" in response:
        response = response.split("<answer>")[-1].replace("</answer>", "").strip()
        if not response:
            return None
        if "\\boxed" not in response:
            return response
    return extract_answer(response)

# def init():
