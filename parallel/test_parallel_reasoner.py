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
    import torch
    import os

    if not torch.distributed.is_initialized():
        # Set default values if not already set
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        # Default to single-node, single-process if not set
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
        )

    countdown_dataset = load_and_preprocess_data()
    training_args = GRPOConfig(
        output_dir="Qwen3-1.7B", 
        logging_steps=10, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        max_completion_length=2048,
        use_vllm=True,
        vllm_server_port=8000,
        bf16=True,
        report_to="wandb",
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B-Base",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=countdown_dataset
    )
    prompts = [next(iter(countdown_dataset))]
    formatted_prompts = [{"prompt": prompt["prompt"]} for prompt in prompts]

    # trainer._generate_and_score_completions(formatted_prompts)
    print("Done")
    
    parallel_passes = build_parallel_passes(trainer, prompts * 4)
    print(parallel_passes.continued_reasoning_prompts)
    print(parallel_passes.continued_reasoning)