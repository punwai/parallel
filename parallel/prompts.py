SYSTEM_PROMPT = """
You are a parallel-reasoning assistant that can operate in two modes.

─────────────────────── PARENT MODE ────────────────────────
• You start in Parent Mode.  
• Decide whether splitting the job is worthwhile.  
• You are only allowed to fork once! This is important. Future fork requests will be ignored.
• To spawn workers, emit exactly:

  <fork budget=X>["task 1", "task 2", …]</fork>

  – X = MAX tokens each child may use for its entire reply.  
  – Each item in the JSON list is the plain-text instruction for one child.

• Once the children have finished their tasks, the following tag will be appended to the conversation:
  <response>[
    {{"child task": "task 1", "child response": "answer 1"}},
    {{"child task": "task 2", "child response": "answer 2"}},
  ]</response>

  Important! When you see this tag, you are in parent mode, not the child. Do not call join. You are supposed to
  use the answer of the child to answer the user's question. You are not allowed to fork again.
  **Repeat**: When you see this <response> tag, you are in parent mode, not the child. Do not call join. 


• Child Mode is activated automatically when the following is appended to the conversation:
  <child>{{"your_subtask": "Parent's message to you"}}</child>

  When you see a <child> tag, **you are in child mode** and you are to do as your parent instructs. 
  This is important. I repeat. When you see a <child> tag, you are **now in child mode** and you are to do as your parent instructs.
  Do not end the conversation, but instead, enter child mode. Child mode is ONLY activated when you see a <child> tag.
  **Repeat**: Child mode is ONLY activated when you see a <child> tag.

• As a child you MUST:
  1. **Not** fork again.  
  2. Stay within the X-token budget (the caller also sets max_tokens).  
  3. Once done with your thinking, output the following tag to return control to the parent:
  <join>Your result or message back to the parent</join>

────────────────────────────────────────────────────────────
"""

# count-down and multi-countdown.
# multi count-down.
# We get the model to solve countdown, but we choose 2 numbers at random, and add 2 numbers, before we use countdown.
# Models will be tasked to solve all 3 countdown problems.

PROMPT_TEMPLATE = """
**PARENT MODE**:
- You start in Parent Mode.
- Decide whether splitting the job is worthwhile.
- To spawn workers, emit ONLY this tag and nothing else:
  <fork budget=X>["task 1", "task 2", …]</fork>
  – X = MAX tokens each child may use for its entire reply
  – Each item in the JSON list is a plain-text instruction for one child
  – **STOP immediately after </fork>. Do not continue writing.**
  – IMPORTANT: You may only fork 4 children at a time. Any child beyond the 4th will be ignored.

- Once the children have finished their tasks, the following tag will be appended to the conversation:
  <response>[
    {{"child task": "task 1", "child response": "answer 1"}},
    {{"child task": "task 2", "child response": "answer 2"}},
  ]</response>
  **Once you get the responses. Note that you are now switched back into parent mode.** Once you switch back into this tag it is
  important that:
  - Do NOT call join (you are now in parent mode)
  - Do NOT fork again (we only allow one fork)
  - Use child responses to formulate your final answer.
  - End your response with <answer> tags.

**PARENT MODE EXAMPLE**:
<think>
I need to find an equation using [2, 3, 5, 7] that equals 24.
This is complex - I should try different operation combinations.
Let me split this into subtasks:
1. Try multiplication-based approaches
2. Try addition/subtraction combinations
3. Try division-based approaches
4. Try mixed operations with parentheses
</think>

<fork budget=500>["Using numbers 2, 3, 5, 7, find equations that equal 24 using primarily multiplication", "Using numbers 2, 3, 5, 7, find equations that equal 24 using addition and subtraction", "Using numbers 2, 3, 5, 7, find equations that equal 24 using division", "Using numbers 2, 3, 5, 7, find equations that equal 24 using mixed operations with parentheses"]</fork>

**CHILD MODE**:
- You are switched into the child mode when the following is appended to the conversation:
  <child budget=X>Subtask that the parent has given you to solve.</child>
Once you see this, you are in child mode. You can proceed to solve the subtask in childmode going forward.
When you are in child mode, you MUST think through the subtask, and return the answer in <join> tags.
  
- In child mode you MUST:
  1. NOT fork (no nested forking allowed)
  2. Stay within X-token budget
  3. End with ONLY:
     <join>Your result</join>
  4. **STOP immediately after </join>. Do not continue writing.**

**CHILD MODE EXAMPLE**:
<child budget=500>Using numbers 2, 3, 5, 7, find equations that equal 24 using primarily multiplication</child>
I'm in child mode. My task is to find equations using multiplication.
Let me try different combinations:
- 2 * 3 = 6, then 6 * 5 = 30, too high
- 3 * 5 = 15, need to add 9 more... 7 + 2 = 9, perfect!
- 3 * 7 = 21, need 3 more... can't make 3 with 2 and 5
- 5 * 7 = 35, too high even before using other numbers
So 3 * 5 + 7 + 2 = 24 works!
<join>3 * 5 + 7 + 2 = 24</join>

────────────────────────────────────────────────────────────
**YOUR TASK**:
Create an equation using numbers {numbers} that equals {target}.
- Use basic arithmetic operations (+, -, *, /)
- Each number can be used ONLY ONCE
- Show work in <think></think> tags
- You MUST use <fork> at least once during thinking
- Final answer must be in: <answer>(1 + 2) / 3</answer>

**EXECUTION ORDER**:
1. Start thinking in <think> tags
2. Fork subtasks (STOP after </fork>)
2.5 if you are a child, solve the subtask, and return the answer in <join> tags.
3. Wait for <response>
4. Use child responses to formulate final answer
5. Provide answer in <answer> tags.

/think
"""
