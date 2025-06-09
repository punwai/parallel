import json
import re
from typing import List, Dict, Optional
import json
from typing import List, Dict

def format_child_answers(
    pairs: List[Dict[str, str]],
    *,
    pretty: bool = False
) -> str:
    """
    Convert a list like
        [{"task": "task 1", "join": "answer 1"},
         {"task": "task 2", "join": "answer 2"}]
    into the canonical string

        <child_answer>{"task 1": "answer 1", "task 2": "answer 2"}</child_answer>

    Parameters
    ----------
    pairs   : list of dicts with 'task' and 'join' keys.
    pretty  : if True, produce multi-line indented JSON (default False).

    Returns
    -------
    str
    """
    # 1. collapse into {task: answer} mapping
    mapping = [{"child task": item["task"], "child response": item["join"]} for item in pairs]

    # 2. JSON serialise
    json_str = json.dumps(
        mapping,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=False,
    )

    # 3. wrap in the required tag
    return f"<response>{json_str}</response>"

# ------------ pre-compiled regexes ------------
_FORK_RE = re.compile(
    r"<fork\s+budget=(\d+)>\s*(\[.*?])\s*</fork>",  # captures budget + JSON list
    re.DOTALL,
)
_JOIN_RE = re.compile(
    r"<join>\s*(.*?)\s*</join>",                    # captures join payload
    re.DOTALL,
)

# ------------ public helpers ------------------
def try_extract_forks(text: str) -> Optional[List[Dict[str, object]]]:
    """
    Parse all <fork budget=X>[...]</fork> blocks.

    Returns
    -------
    A list of dicts like:
        [{"budget": 16, "tasks": ["task A", "task B"]}, ...]
    or None if no fork tags appear.

    Raises
    ------
    ValueError on malformed JSON or non-list payloads.
    """
    results: List[Dict[str, object]] = []

    for m in _FORK_RE.finditer(text):
        budget = int(m.group(1))
        try:
            tasks = json.loads(m.group(2))
            if not isinstance(tasks, list):
                return None
            # Only keep the first 4 tasks
            tasks = list(map(str, tasks))[:4]
            results.append({"budget": budget, "tasks": tasks})
        except json.JSONDecodeError as exc:
            return None
        
    return results[0] if results and len(results) else None


def try_extract_joins(text: str) -> Optional[str]:
    """
    Return the first payload found inside a <join>...</join> tag,
    stripped of leading/trailing whitespace.  If none, return None.
    """
    m = _JOIN_RE.search(text)
    return m.group(1).strip() if m else None

