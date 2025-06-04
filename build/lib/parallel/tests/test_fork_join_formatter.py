# tests/test_fork_helpers.py
import pytest
from parallel.utils.fork_helpers import try_extract_forks, try_extract_joins

# ---------- try_extract_forks -------------------------------------------------
def test_no_fork_returns_none():
    assert try_extract_forks("plain text") is None

def test_single_fork_parsed_correctly():
    text = '<fork budget=12>["a","b"]</fork>'
    expected = [{"budget": 12, "tasks": ["a", "b"]}]
    assert try_extract_forks(text) == expected

def test_multiple_forks_combined():
    text = (
        '<fork budget=5>["x"]</fork> middle '
        '<fork budget=3>["y","z"]</fork>'
    )
    expected = [
        {"budget": 5, "tasks": ["x"]},
        {"budget": 3, "tasks": ["y", "z"]},
    ]
    assert try_extract_forks(text) == expected

def test_malformed_json_raises():
    bad = '<fork budget=1>["missing quote]</fork>'
    with pytest.raises(ValueError):
        try_extract_forks(bad)

# ---------- try_extract_joins -------------------------------------------------
def test_no_join_returns_none():
    assert try_extract_joins("nothing to join") is None

def test_first_join_returned():
    text = "<join>foo</join><join>bar</join>"
    assert try_extract_joins(text) == "foo"

def test_join_whitespace_stripped():
    text = "<join>\n  result  \t</join>"
    assert try_extract_joins(text) == "result"
