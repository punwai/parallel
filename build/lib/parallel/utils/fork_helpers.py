import json
import re
from typing import List, Dict, Optional

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
                raise ValueError("Fork payload must be a JSON list.")
            results.append({"budget": budget, "tasks": list(map(str, tasks))})
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON inside <fork>: {exc}") from None

    return results or None


def try_extract_joins(text: str) -> Optional[str]:
    """
    Return the first payload found inside a <join>...</join> tag,
    stripped of leading/trailing whitespace.  If none, return None.
    """
    m = _JOIN_RE.search(text)
    return m.group(1).strip() if m else None
