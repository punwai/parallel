import asyncio
from datasets import load_dataset
from parallel.custom_trl_trainer import GRPOTrainer
from parallel.parallel_utils import build_parallel_passes
from parallel.train_grpo import load_and_preprocess_data, reward_len
from parallel.utils.fork_helpers import format_child_answers, try_extract_forks, try_extract_joins
from trl.trainer.grpo_config import GRPOConfig

from parallel.prompts import SYSTEM_PROMPT, PROMPT_TEMPLATE
from parallel.utils.countdown_rewards import compute_score
from trl.data_utils import maybe_apply_chat_template
from dataclasses import dataclass
from typing import List, Union
import asyncio

if __name__ == "__main__":
    countdown_dataset = load_and_preprocess_data()
    training_args = GRPOConfig(
        output_dir="Qwen3-4B", 
        logging_steps=10, 
        use_vllm=True,
        vllm_server_port=8005
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B-Base",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=countdown_dataset
    )
    prompts = [next(iter(countdown_dataset))] * 16
    formatted_prompts = [{"prompt": prompt["prompt"]} for prompt in prompts]

    trainer._generate_and_score_completions(formatted_prompts)
    print("Done")
    
    # parallel_passes = build_parallel_passes(trainer, prompts * 4)