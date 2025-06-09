from datasets import load_dataset
# from custom_trl_trainer import GRPOTrainer
from trl.trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from prompts import SYSTEM_PROMPT, PROMPT_TEMPLATE
from utils.countdown_rewards import compute_score

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    nums = kwargs["nums"]
    target = kwargs["target"]
    ground_truths = [
        {
            "numbers": [int(n) for n in nums],
            "target": int(target)
        } for nums, target in zip(nums, target)
    ]
    rewards = [
        compute_score(completion, ground_truth)
        for completion, ground_truth in zip(completions, ground_truths)
    ]
    return rewards

# Load the Jiayi-Pan/Countdown-Tasks-3to4 dataset
def load_and_preprocess_data():
    countdown_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    def make_prompt(example):
        target = example.get("target", "")
        numbers = example.get("nums", "")
        # Format numbers list into string
        if isinstance(numbers, list):
            numbers_str = ", ".join(str(n) for n in numbers)
        else:
            numbers_str = str(numbers)
        # Use the PROMPT_TEMPLATE defined above
        prompt = PROMPT_TEMPLATE.format(
            numbers=numbers_str,
            target=target
        )
        return {"prompt": prompt}

    countdown_dataset = countdown_dataset.map(make_prompt)
    return countdown_dataset

if __name__ == "__main__":
    countdown_dataset = load_and_preprocess_data()
    training_args = GRPOConfig(
        output_dir="Qwen3-4B", 
        logging_steps=10, 
        per_device_train_batch_size=8,
        max_completion_length=5000,
        use_vllm=True,
        vllm_server_port=8005
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B-Base",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=countdown_dataset,
    )
    trainer.train()
