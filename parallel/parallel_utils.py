from parallel.utils.fork_helpers import format_child_answers, try_extract_forks, try_extract_joins
from trl.trainer.grpo_config import GRPOConfig

from parallel.prompts import SYSTEM_PROMPT, PROMPT_TEMPLATE
from parallel.utils.countdown_rewards import compute_score
from trl.data_utils import maybe_apply_chat_template


from dataclasses import dataclass
from typing import List, Union
import asyncio

# clip the text after the end_token
def clip_after(text, end_token):
    return text.split(end_token)[0] + end_token

def apply_chat_template_and_force_assistant_continue(messages, tokenizer):
    """Apply chat template, and force the assistant to continue"""
    completions_text = maybe_apply_chat_template(messages, tokenizer)["text"]
    # Split on <|im_end|> and take first part, in case there are multiple
    last_idx = completions_text.rfind("<|im_end|>")
    if last_idx != -1:
        completions_text = completions_text[:last_idx]
    return completions_text
    
def get_child_prompt_given_task(task, previous_text, user_text, budget, tokenizer):
    assistant_text = apply_chat_template_and_force_assistant_continue({
        "messages": [
            {
                "role": "user",
                "content": user_text
            },
            {
                "role": "assistant",
                "content": f"{previous_text} <child budget={budget}>{task}</child>"
                # "content": f"Hmmm.... Let's think step by step."
            },
        ]
    }, tokenizer)
    return assistant_text

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
    child_reasoning_prompts: List[List[str]]

    child_reasoning_ids: List[List[int]]
    continued_reasoning_prompts: List[List[str]]

    continued_reasoning_ids: List[int]

# ── 1. main driver ─────────────────────────────────────────────────────────────
def build_parallel_passes(
    trainer,
    raw_prompts: List[Union[str, dict]],
    *,
    total_max_tokens: int = 1500
) -> UnprocessedParallelPass:
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
        max_tokens           = total_max_tokens,
        guided_decoding_regex= None,
        stop                 = "</fork>",
    )
    parent_inference_tokens_used = [len(ids) for ids in parent_ids]
    tokens_remaining = [max(1, total_max_tokens - p) for p in parent_inference_tokens_used]

    parent_texts = trainer.processing_class.batch_decode(
        parent_ids, skip_special_tokens=True
    )
    parent_texts = [clip_after(txt, "</fork>") for txt in parent_texts]
    child_task_batch = [try_extract_forks(txt) for txt in parent_texts]

    # --- 1C. phase-2: spawn children ----------------------------------------------
    child_prompt_indices = []
    present_child_indices = set()
    all_child_prompts = []
    child_batch_prompts = []
    for ix, (task_info, p_txt, user_txt, tokens_remaining_intrabatch) in enumerate(zip(child_task_batch, parent_texts, user_prompts, tokens_remaining)):
        if task_info is not None:
            child_prompts = [
                get_child_prompt_given_task(
                    task, p_txt, user_txt, task_info["budget"], trainer.processing_class
                )
                for task in task_info["tasks"]
            ]
            child_prompts = [prompt for prompt in child_prompts]

            # we always deal with flattened child prompts
            child_prompt_indices.extend([ix] * len(task_info["tasks"]))
            child_batch_prompts.extend(child_prompts)
            # 
            all_child_prompts.append(
                {"prompts": child_prompts, "budget": task_info["budget"], "tokens_remaining": tokens_remaining_intrabatch}
            )
            present_child_indices.add(ix)

    child_futures = [
        trainer.vllm_client.agenerate(
            prompts              = cp["prompts"],
            n                    = 1,
            repetition_penalty   = 1.0,
            temperature          = 1.0,
            top_p                = 1.0,
            top_k                = -1,
            min_p                = 0.0,
            max_tokens           = min(cp["tokens_remaining"], cp["budget"]),
            guided_decoding_regex= None,
            stop                 = "</join>",
        )
        for cp in all_child_prompts
    ]

    loop = asyncio.get_event_loop()
    child_ids_batch = loop.run_until_complete(asyncio.gather(*child_futures))

    # get the per-child max length. However, this is undetermined.
    child_tokens_used = [max([len(child_id_batch) for child_id_batch in children_id_batch]) for children_id_batch in child_ids_batch]
    child_texts_batch = [
        trainer.processing_class.batch_decode(ids, skip_special_tokens=True)
        for ids in child_ids_batch
    ]
    filtered_tokens_remaining = [t for ix, t in enumerate(tokens_remaining) if ix in present_child_indices]
    tokens_remaining = [max(1, f - fc) for f, fc in zip(filtered_tokens_remaining, child_tokens_used)]

    # --- 1D. phase-3: parent consumes <join> --------------------------------------
    final_parent_inputs = []

    present_child_tasks = [c for c in child_task_batch if c is not None]
    for task_info, child_texts, p_txt, user_txt, tokens_remaining_intrabatch in zip(
            present_child_tasks, child_texts_batch, parent_texts, user_prompts, tokens_remaining
        ):
        # some of these can be None. How to we handle this?
        joins = [try_extract_joins(ct) for ct in child_texts]
        answer_json = format_child_answers([
            {"task": t, "join": j}
            for t, j in zip(task_info["tasks"], joins)
        ])
        final_parent_input_id_text = get_parent_prompt_given_answer_json(
            answer_json, p_txt, user_txt, trainer.processing_class
        )
        final_parent_inputs.append(final_parent_input_id_text)

    # Call vllm_client.agenerate separately for each input, asynchronously, since tokens_remaining may differ
    async def generate_final_parent_ids():
        tasks = [
            trainer.vllm_client.agenerate(
                prompts              = [parent_input],
                n                    = 1,
                repetition_penalty   = 1.0,
                temperature          = 1.0,
                top_p                = 1.0,
                top_k                = -1,
                min_p                = 0.0,
                max_tokens           = tokens_rem,
                guided_decoding_regex= None,
                stop                 = ["</fork>"],
            )
            for parent_input, tokens_rem in zip(final_parent_inputs, tokens_remaining)
        ]
        results = await asyncio.gather(*tasks)
        # Each result is a batch of 1, so extract the first element
        return [ids[0] for ids in results]

    loop = asyncio.get_event_loop()
    final_parent_ids = loop.run_until_complete(generate_final_parent_ids())
    final_parent_texts = trainer.processing_class.batch_decode(
        final_parent_ids, skip_special_tokens=True
    )

    print("LEN CHILD IDS", len(child_ids_batch))
    print("LEN CHILD TEXTS", [len(id) for id in child_ids_batch])

    child_ids_batch_flattened = [item for sublist in child_ids_batch for item in sublist]
    print("LEN CHILD IDS FLATTENED", len(child_ids_batch_flattened))

    # can you unflatten the child_reasoning_prompts and child_ids_batch_flattened
    # you can use the child_prompt_indices to do this.

    # Unflatten child_reasoning_prompts and child_ids_batch_flattened using child_prompt_indices
    # child_prompt_indices is a list of lists of indices, one per parent prompt, each containing indices of children

    # Unflatten child_reasoning_prompts
    unflattened_child_reasoning_prompts = [[] for _ in range(len(raw_prompts))]
    unflattened_child_ids_batch = [[] for _ in range(len(raw_prompts))]

    for prompt_idx, ids, prompts in zip(child_prompt_indices, child_ids_batch_flattened, child_batch_prompts):
        unflattened_child_reasoning_prompts[prompt_idx].append(prompts)
        unflattened_child_ids_batch[prompt_idx].append(ids)
    print("UNFLATTENED CHILD REASONING PROMPTS", len(unflattened_child_reasoning_prompts))
    print("UNFLATTENED CHILD REASONING PROMPTS", [len(ids) for ids in unflattened_child_reasoning_prompts])

    # --- 1E. pack everything -------------------------------------------------------
    return UnprocessedParallelPass(
        initial_reasoning      = parent_texts,
        child_reasoning_traces = child_texts_batch,
        continued_reasoning    = final_parent_texts,
        # reasoning_ids
        initial_reasoning_ids  = parent_ids,
        child_reasoning_ids = unflattened_child_ids_batch,
        continued_reasoning_ids = final_parent_ids,
        # Array[Array[]] one for each child in the prompt -- note that we enforce the
        # models to reason, or else they do not get any rewards at all.
        child_reasoning_prompts = unflattened_child_reasoning_prompts,
        continued_reasoning_prompts = final_parent_inputs,
    )
# ────────────────────────────────────────────────────────────────────────────────
