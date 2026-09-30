from prompting import parse_gsm8k_data, load_data, generate_model_answers, eval_model, gsm8k_test_path
import numpy as np

data = load_data(gsm8k_test_path)

res_question_only = eval_model(data, "question_only")
res_zero_shot = eval_model(data, "zero_shot")
res_few_shot = eval_model(data, "few_shot")

ress = [res_question_only, res_zero_shot, res_few_shot]

# Computing metrics
def compute_sum(dic_array, key):
    sum([d[key] for d in dic_array])

prompt_titles = ["Question Only", "Zero Shot", "Few shot"]
keys = ["format_reward", "answer_reward", "reward"]
keys_titles = ["Format", "Answer", "Reward"]

for i in range(3):
    print(f"{prompt_titles[i]} Prompt:")
    for j in range(3):
        sum = compute_sum(ress[i], keys[j])
        print(f"{keys_titles[j]}: {sum}")
print("\n")