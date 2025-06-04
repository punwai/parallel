from parallel.utils.fork_helpers import format_child_answers, try_extract_forks, try_extract_joins
from trl.trainer.grpo_config import GRPOConfig

from parallel.prompts import SYSTEM_PROMPT, PROMPT_TEMPLATE
from parallel.utils.countdown_rewards import compute_score
from trl.data_utils import maybe_apply_chat_template


from dataclasses import dataclass
from typing import List, Union
import asyncio

def apply_chat_template_and_force_assistant_continue(messages, tokenizer):
    """Apply chat template, and force the assistant to continue"""
    completions_text = maybe_apply_chat_template(messages, tokenizer)["text"]
    # Split on <|im_end|> and take first part, in case there are multiple
    chunks = completions_text.split("<|im_end|>")
    if len(chunks) > 2:
        completions_text = chunks[-2]
    return completions_text
    
def get_child_prompt_given_task(task, previous_text, user_text, budget, tokenizer):
    print("previous_text", previous_text)
    print("user_text", user_text)
    assistant_text = apply_chat_template_and_force_assistant_continue({
        "messages": [
            {
                "role": "user",
                "content": user_text
            },
            {
                "role": "assistant",
                "content": f"{previous_text}<child budget={budget}>{task}</child>"
            },
        ]
    }, tokenizer)
    prefix_ids = tokenizer.encode(f"<child budget={budget}>{task}</child>")
    return assistant_text, prefix_ids

def get_parent_prompt_given_answer_json(answer_json, previous_text, user_text, tokenizer):
    assistant_text = apply_chat_template_and_force_assistant_continue({
        "messages": [
            {
                "role": "user",
                "content": user_text
            },
            {
                "role": "assistant",
                "content": f"{previous_text}{answer_json}"
            },
        ]
    }, tokenizer)
    return assistant_text

# ── 0. dataclass ───────────────────────────────────────────────────────────────
@dataclass
class UnprocessedParallelPass:
    # string repr
    initial_reasoning: str
    child_reasoning_traces: List[str]
    continued_reasoning: str
    # list repr
    initial_reasoning_ids: List[int]
    child_reasoning_prefix_ids: List[List[int]]

    child_reasoning_ids: List[List[int]]
    continued_reasoning_prefix_ids: List[List[int]]

    continued_reasoning_ids: List[int]

# ── 1. main driver ─────────────────────────────────────────────────────────────
def build_parallel_passes(
    trainer,
    raw_prompts: List[Union[str, dict]],
    accelerator,
    *,
    parent_max_tokens: int = 1500
) -> List[UnprocessedParallelPass]:
    """
    Runs one full fork-and-join episode per prompt and returns a list of
    UnprocessedParallelPass objects (one per input prompt).
    """
    # --- 1A. normalize user prompts ------------------------------------------------
    user_prompts = [
        p["prompt"] if isinstance(p, dict) else p
        for p in raw_prompts
    ]

    # --- 1B. phase-1: parent ↦ <fork> ---------------------------------------------
    parent_input_texts = [
        maybe_apply_chat_template(
            {"messages": [{"role": "user", "content": up}]},
            tokenizer=trainer.processing_class
        )["text"]
        for up in user_prompts
    ]

    parent_ids = trainer.vllm_client.generate(
        prompts              = parent_input_texts,
        n                    = 1,
        repetition_penalty   = 1.0,
        temperature          = 1.0,
        top_p                = 1.0,
        top_k                = -1,
        min_p                = 0.0,
        max_tokens           = parent_max_tokens,
        guided_decoding_regex= None,
        stop                 = ["</fork>"],
    )
    parent_texts = trainer.processing_class.batch_decode(
        parent_ids, skip_special_tokens=True
    )
    print("parent_texts", parent_texts)
    child_task_batch = [try_extract_forks(txt) for txt in parent_texts]

    # --- 1C. phase-2: spawn children ----------------------------------------------
    all_child_prompts = []
    child_batch_prefix_ids =[]
    for task_info, p_txt, user_txt in zip(
            child_task_batch, parent_texts, user_prompts):
        child_prompts = [
            get_child_prompt_given_task(
                task, p_txt, user_txt, task_info["budget"], trainer.processing_class
            )
            for task in task_info["tasks"]
        ]
        child_batch_prefix_ids.append([prefix_ids for _, prefix_ids in child_prompts])
        child_prompts = [prompt for prompt, _ in child_prompts]

        all_child_prompts.append(
            {"prompts": child_prompts, "budget": task_info["budget"]}
        )

    print(all_child_prompts)

    child_futures = [
        trainer.vllm_client.agenerate(
            prompts              = cp["prompts"],
            n                    = 1,
            repetition_penalty   = 1.0,
            temperature          = 1.0,
            top_p                = 1.0,
            top_k                = -1,
            min_p                = 0.0,
            max_tokens           = cp["budget"],
            guided_decoding_regex= None,
            stop                 = ["</join>"],
        )
        for cp in all_child_prompts
    ]
    loop = asyncio.get_event_loop()
    child_ids_batch = loop.run_until_complete(asyncio.gather(*child_futures))
    child_texts_batch = [
        trainer.processing_class.batch_decode(ids, skip_special_tokens=True)
        for ids in child_ids_batch
    ]

    # --- 1D. phase-3: parent consumes <join> --------------------------------------
    final_parent_inputs = []
    final_parent_prefix_ids = []

    for task_info, child_texts, p_txt, user_txt in zip(
            child_task_batch, child_texts_batch, parent_texts, user_prompts):
        joins = [try_extract_joins(ct) for ct in child_texts]
        answer_json = format_child_answers([
            {"task": t, "join": j}
            for t, j in zip(task_info["tasks"], joins)
        ])
        final_parent_inputs.append(
            get_parent_prompt_given_answer_json(
                answer_json, p_txt, user_txt, trainer.processing_class
            )
        )
        final_parent_prefix_ids.append(trainer.processing_class.encode(answer_json))

    final_parent_ids = trainer.vllm_client.generate(
        prompts              = final_parent_inputs,
        n                    = 1,
        repetition_penalty   = 1.0,
        temperature          = 1.0,
        top_p                = 1.0,
        top_k                = -1,
        min_p                = 0.0,
        max_tokens           = parent_max_tokens,
        guided_decoding_regex= None,
        stop                 = ["</fork>"],
    )
    final_parent_texts = trainer.processing_class.batch_decode(
        final_parent_ids, skip_special_tokens=True
    )

    # --- 1E. pack everything -------------------------------------------------------
    return [
        UnprocessedParallelPass(
            initial_reasoning      = init,
            child_reasoning_traces = child_batch,
            continued_reasoning    = cont,
            # step 1
            initial_reasoning_ids  = parent_ids,
            # step 2
            child_reasoning_prefix_ids = child_batch_prefix_ids,
            child_reasoning_ids = child_ids_batch,
            # step 3
            continued_reasoning_prefix_ids = final_parent_prefix_ids,
            continued_reasoning_ids = final_parent_ids,
        )
        for init, child_batch, cont in zip(
            parent_texts, child_texts_batch, final_parent_texts
        )
    ]
# ────────────────────────────────────────────────────────────────────────────────
