"""
Codex-Style Agent & Coding Accuracy Benchmark Suite.
Tests:
1. Canonical Coding & Unit Tests (HumanEval/MBPP style)
2. Code Editing & Refactoring
3. Structured JSON Schema Output
4. Tool-Call Generation Correctness
5. Multi-Step Constraint Adherence
6. Error Traceback Recovery
"""

import json
import re
import traceback
from typing import Dict, Any, List
from mlx_lm import generate

CODING_TESTS = [
    {
        "id": "humaneval_fibonacci",
        "name": "Fibonacci with Dynamic Programming",
        "prompt": "Write a Python function `fib(n: int) -> int` that calculates the n-th Fibonacci number in O(n) time and O(1) auxiliary space. Output ONLY the Python code inside ```python ```.",
        "test_code": """
assert fib(0) == 0
assert fib(1) == 1
assert fib(10) == 55
assert fib(20) == 6765
"""
    },
    {
        "id": "humaneval_is_palindrome",
        "name": "Case-Insensitive Palindrome",
        "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring non-alphanumeric characters and case. Output ONLY the Python code inside ```python ```.",
        "test_code": """
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("") == True
assert is_palindrome("ab_a") == True
"""
    },
    {
        "id": "humaneval_flatten",
        "name": "Nested List Flattener",
        "prompt": "Write a Python function `flatten(nested: list) -> list` that recursively flattens a deeply nested list of arbitrary depth into a 1D list. Output ONLY the Python code inside ```python ```.",
        "test_code": """
assert flatten([1, [2, [3, 4], 5], [6]]) == [1, 2, 3, 4, 5, 6]
assert flatten([]) == []
assert flatten([[[[1]]]]) == [1]
"""
    },
    {
        "id": "codex_code_edit",
        "name": "Code Editing / Feature Extension",
        "prompt": """Below is an existing function:
```python
def process_user(data):
    return {"name": data["name"], "age": data["age"]}
```
Modify `process_user` so that if "email" is in `data`, it includes `email` in lowercase; otherwise defaults `email` to `None`.
Output ONLY the updated Python function inside ```python ```.""",
        "test_code": """
assert process_user({"name": "Alice", "age": 30, "email": "ALICE@EXAMPLE.COM"}) == {"name": "Alice", "age": 30, "email": "alice@example.com"}
assert process_user({"name": "Bob", "age": 25}) == {"name": "Bob", "age": 25, "email": None}
"""
    },
    {
        "id": "codex_structured_json",
        "name": "Structured JSON Extraction",
        "prompt": """Extract the user profile information from this text:
'John Doe is a 28-year-old Backend Engineer living in San Francisco with skills in Python and Rust.'
Return STRICTLY a JSON object matching this schema:
{
  "name": string,
  "age": number,
  "role": string,
  "city": string,
  "skills": list of strings
}
Do NOT include any commentary. Return ONLY the raw JSON object.""",
        "validator": lambda text: validate_json_schema(text, ["name", "age", "role", "city", "skills"])
    },
    {
        "id": "codex_tool_call",
        "name": "Tool Call Generation",
        "prompt": """You are an agent with access to a tool `search_database(table: str, query: str, limit: int)`.
The user wants to search for 'quantization' in the 'research_papers' table and get at most 5 results.
Generate the exact tool call in JSON format with fields "name" and "arguments". Return ONLY the JSON object.""",
        "validator": lambda text: validate_tool_call(text, "search_database", {"table": "research_papers", "query": "quantization", "limit": 5})
    },
    {
        "id": "codex_error_recovery",
        "name": "Error Recovery / Bug Fixing",
        "prompt": """The following code threw `ZeroDivisionError`:
```python
def compute_averages(records):
    # records is a list of lists of numbers
    return [sum(r) / len(r) for r in records]
```
Fix the function so that empty sublists return `0.0` instead of crashing.
Output ONLY the corrected Python function inside ```python ```.""",
        "test_code": """
assert compute_averages([[10, 20], [], [5, 5, 5]]) == [15.0, 0.0, 5.0]
assert compute_averages([]) == []
"""
    }
]

def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def validate_json_schema(text: str, required_keys: List[str]) -> bool:
    try:
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        data = json.loads(clean)
        return all(k in data for k in required_keys)
    except Exception:
        return False

def validate_tool_call(text: str, expected_tool: str, expected_args: Dict[str, Any]) -> bool:
    try:
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        data = json.loads(clean)
        if data.get("name") != expected_tool:
            return False
        args = data.get("arguments", {})
        return all(args.get(k) == v for k, v in expected_args.items())
    except Exception:
        return False

def run_codex_eval(model: Any, tokenizer: Any, prompt_cache: Any = None) -> Dict[str, Any]:
    """Runs all Codex & HumanEval-style tests against the model."""
    results = []
    passed_count = 0

    for test in CODING_TESTS:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": test["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        output = generate(model, tokenizer, prompt=formatted, prompt_cache=prompt_cache, max_tokens=256)
        
        passed = False
        error_msg = ""
        
        if "test_code" in test:
            code = extract_python_code(output)
            try:
                exec_globals = {}
                exec(code, exec_globals)
                exec(test["test_code"], exec_globals)
                passed = True
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
        elif "validator" in test:
            passed = test["validator"](output)
            if not passed:
                error_msg = "Validation failed on output schema/content"

        if passed:
            passed_count += 1

        results.append({
            "id": test["id"],
            "name": test["name"],
            "passed": passed,
            "error": error_msg,
            "raw_output": output.strip()[:200] + "..." if len(output) > 200 else output.strip()
        })

    pass_rate = round((passed_count / len(CODING_TESTS)) * 100, 2)
    return {
        "passed_count": passed_count,
        "total_count": len(CODING_TESTS),
        "pass_rate_pct": pass_rate,
        "test_details": results
    }
