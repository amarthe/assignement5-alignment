from prompting import load_data, eval_model, GSM8K_TEST_PATH
import numpy as np

data = load_data(GSM8K_TEST_PATH)

res_question_only = eval_model(data, "question_only")
res_zero_shot = eval_model(data, "zero_shot")
res_few_shot = eval_model(data, "few_shot")

ress = [res_question_only, res_zero_shot, res_few_shot]

# Computing metrics
def compute_sum(dic_array, key):
    return np.sum([d[key] for d in dic_array])

prompt_titles = ["Question Only", "Zero Shot", "Few shot"]
keys = ["format_reward", "answer_reward", "reward"]
keys_titles = ["Format", "Answer", "Reward"]

for i in range(3):
    print(f"{prompt_titles[i]} Prompt:")
    for j in range(3):
        sum = compute_sum(ress[i], keys[j])
        print(f"{keys_titles[j]}: {sum}")
print("\n")