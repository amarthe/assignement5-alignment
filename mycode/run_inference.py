from inference import load_data, eval_model, GSM8K_TEST_PATH
import numpy as np
from cs336_alignment.vllm_utils import VLLMServer

data = load_data(GSM8K_TEST_PATH)

sampling_params = {
    "temperature": 1,
    "max_tokens": 512,
    "seed": 42,
    "n":1,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}

vllm_server = VLLMServer("allenai/OLMo-2-0425-1B", gpu=0)
vllm_server.start()

res_question_only = eval_model(vllm_server, data, "question_only", sampling_params)
res_zero_shot = eval_model(vllm_server, data, "zero_shot", sampling_params)
res_few_shot = eval_model(vllm_server, data, "few_shot", sampling_params)

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