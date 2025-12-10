from __future__ import annotations
import json
from operator import truediv
import os
import requests
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import csv
import logging
from uuid import uuid4
import threading
import hashlib
import fnmatch
from tree_sitter import Parser
from tree_sitter_language_pack import get_language
run_id = None
_current_tool_manager = None
_codeparse_util_language_cache = {}
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1500"))
PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
VERSION_COMPATIBILITY_FIX = """
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;
"""
MAX_FIX_TASK_STEPS = 200
LATEST_OBSERVATIONS_TO_KEEP = 15
SUMMARIZE_BATCH_SIZE = 5
MAX_SUMMARY_RANGES = 6
GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
GLM_OLD_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [
    model
    for model in [GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, GLM_MODEL_NAME, QWEN_MODEL_NAME]
    for _ in range(2)
]
REASONING_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]
FORMAT_PROMPT = textwrap.dedent(
    """
**CRITICAL: You can make MULTIPLE tool calls in ONE response for efficiency!**
## Response Formats
### Format 1: Multiple Tool Calls (RECOMMENDED for efficiency)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {valid JSON}
### Format 2: Single Tool Call (Legacy, less efficient)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}
## When to Use Multiple Tool Calls
**ALWAYS batch these operations:**
1. **Edit + Test**: After code edit, MUST test in same response
2. **Multiple Searches**: Batch all search operations together
3. **Multiple File Reads**: Read all needed files at once
4. **Multiple Tests**: Run all test files together
## Examples
✅ **Excellent - Edit and Test Together**:
next_thought: I'll fix the bug and immediately verify with tests
tool_call_1:
    tool_name: apply_code_edit
    tool_args: {"file_path": "abcd.py", "search": "old_code", "replace": "fixed_code"}
tool_call_2:
    tool_name: run_code
    tool_args: {"content": "test_content", "file_path": "file.js"}
✅ **Good - Batch Multiple Searches**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: search_in_all_files_content
    tool_args: {"search_term": "function problematic_func"}
tool_call_2:
    tool_name: search_in_all_files_content
    tool_args: {"search_term": "problematic_func("}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "abcd.js"}
❌ **Bad - One tool per response (too slow)**:
Response 1:
next_thought: Let me edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", ...}
Response 2 (next turn):
next_thought: Now let me test it
next_tool_name: run_code
...  # ← Should have been in previous response!
## Critical Rules
- Use multiple tool_call_N when possible (tool_call_1, tool_call_2, tool_call_3, ...)
- After any edit: MUST include test in same response
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
"""
)
STOP_INSTRUCTION = textwrap.dedent(
    """
# 🎯 RESPONSE REQUIREMENTS
- DO NOT generate `observation:` - it will be provided by the system
- You can make MULTIPLE tool calls in one response using tool_call_1, tool_call_2, tool_call_3, etc.
- For efficiency: Batch related operations together (e.g., edit + test in ONE response)
- Format: next_thought: ... followed by one or more tool_call_N blocks
"""
)
DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
You're not allowed to repeat the same tool call with the same arguments.
Your previous response:
{previous_response}
Try a different approach:
1. If you just searched, try reading the file instead
2. If you just edited, try running tests to verify
3. If tests failed, try a different fix approach
"""
)
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
Now let's start.
```
{problem_statement}
```
"""
)
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.
## Follow these steps to fix the issue:
1. As a first step, find the relevant files in the repo to work on.
2. Localise the code causing the issue.
3. Edit the sourcecode of the repo to resolve the issue.
4. Think about edgecases and make sure the fix handles them as well.
5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase
- Prefer using `get_function_body` to retrieve the complete body of a specific function. This is the most efficient way to get function code when you know the function name.
- Prefer using `get_file_content` to read the code you need to modify.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.
# Tool Usage Guidelines
- Use appropriate tools to gather context before making changes.
- **Choose the right tool for the job**:
  - Use `get_file_skeleton` to retrieve the skeleton of a file, rather than reading the whole file content. Use this tool when you need to know the structure of a file, which includes the functions and classes in a file.
  - Use `get_function_body` to retrieve the complete body of a specific function (including decorators). This is the most efficient way to get function code when you know the function name.
  - Use `get_file_content` for reading arbitrary file sections, entire files, or specific code areas.
  - Use `search_in_all_files_content` to find where a search term (regex pattern) appears and get context around matches.
- If required parameters are missing, infer them from the problem statement and code.
- Use exact values provided by the user (especially in quotes).
- Don't make up values for or ask about optional parameters.
- Use `search_in_all_files_content` to find all occurrences of an issue before fixing.
You have access to the following tools:-
{tools_docs}
Here is the problem statement:
{problem_statement}
{format_prompt}
""")
class EnhancedNetwork:
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        """Extract tool_name and tool_args from a tool_call block"""
        tool_name_match = re.search(r"tool_name\s*:\s*(\S+)", block, re.IGNORECASE)
        if not tool_name_match:
            return None
        tool_name = tool_name_match.group(1).strip().strip('"').strip("'")
        # Find tool_args
        args_match = re.search(r"tool_args\s*:\s*(\{)", block, re.IGNORECASE)
        if not args_match:
            return None
        args_start = args_match.start(1)
        json_str = cls._extract_balanced_braces(block, args_start)
        if json_str:
            try:
                tool_args = json.loads(json_str)
                return {"tool_name": tool_name, "tool_args": tool_args}
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    json_str_fixed = json_str.replace("'", '"')
                    tool_args = json.loads(json_str_fixed)
                    return {"tool_name": tool_name, "tool_args": tool_args}
                except:
                    pass
        return None
    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        """Extract balanced JSON object starting from start_pos"""
        if start_pos >= len(text):
            return None
        brace_count = 0
        in_string = False
        escape_next = False
        start = -1
        for i in range(start_pos, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    if start == -1:
                        start = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        return text[start : i + 1]
        return None
    @classmethod
    def get_cost_usage(cls) -> dict:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/usage?evaluation_run_id={run_id if run_id else str(uuid4())}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            usage_info = response.json()
            if isinstance(usage_info, dict):
                return usage_info
            else:
                return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
        except Exception as e:
            # Return a safe default dictionary instead of a string
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
    @classmethod
    def inference(
        cls,
        messages: List[Dict[str, Any]],
        model: str,
        run_id: str = str(uuid4()),
        temperature: float = 0.0,
    ) -> dict:
        """Prod inference with caching"""
        # Support both single model (str) and list of models
        models = [model] if isinstance(model, str) else model
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")
            if role == "assistant" and not content.strip():
                continue
            cleaned_msgs.append({"role": role, "content": content})
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")
        # Calculate and print input token size
        (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        ) = cls._request_next_action_with_retry(
            cleaned_msgs, models=models, temperature=temperature
        )
        return (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        )
    @classmethod
    def make_request(
        cls,
        messages: list,
        model: str,
        attempt: int = 0,
        temperature: float = 0.0,
        timeout: int = 180,
        tool_mode: str = "none",
        tool_docs: list = [],
    ) -> str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        # Normalise attempts
        attempts = max(1, attempt or 1)
        request_data = {
            "evaluation_run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "model": model,
            "tool_mode": tool_mode,
            "tools": tool_docs,
        }
        headers = {"Content-Type": "application/json"}
        start_time = time.time()
        for i in range(attempts):
            try:
                response = requests.post(
                    url, json=request_data, timeout=(60, timeout), headers=headers
                )  # (connect timeout, read timeout)
                response.raise_for_status()
                try:
                    response_json = response.json()
                except JSONDecodeError as e:
                    # If final attempt, surface as error; otherwise retry
                    if i >= attempts - 1:
                        elapsed = time.time() - start_time
                        raise ValueError(
                            f"HTTP ERROR: Invalid JSON response for model {model} after {attempts} attempts: {e}"
                        )
                    continue
                # Support both OpenAI-style and raw text responses
                try:
                    raw_text = response_json["content"]
                    tool_calls = response_json["tool_calls"]
                except Exception as e:
                    raise RuntimeError(
                        f"HTTP ERROR: Response Parse Error timeout for model {model} after {attempts} attempts"
                    )
                if (tool_mode == "none" and (raw_text is None or raw_text == "")) or (
                    tool_mode != "none" and (tool_calls is None or len(tool_calls) == 0)
                ):
                    raise RuntimeError(
                        f"HTTP ERROR: NO RESPONSE FOUND Tool model {model} after {attempts} attempts"
                    )
                elapsed = time.time() - start_time
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(
                        f"HTTP ERROR: Request timeout for model {model} after {attempts} attempts"
                    )
                time.sleep(5)
                continue
            except requests.exceptions.ConnectionError as e:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(
                        f"HTTP ERROR: Connection error for model {model} after {attempts} attempts: {e}"
                    )
                time.sleep(10)
                continue
            except requests.exceptions.HTTPError as e:
                status_code = (
                    e.response.status_code if e.response is not None else "unknown"
                )
                elapsed = time.time() - start_time
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model}"
                # Check for 504 Gateway Timeout specifically
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(
                            f"HTTP ERROR 504: Gateway Timeout for model {model} after {attempts} attempts: {e}"
                        )
                    time.sleep(10)
                    continue
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(10)
                continue
            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(
                        f"HTTP ERROR: Request failed for model {model} after {attempts} attempts: {e}"
                    )
                time.sleep(10)
                continue
        # Fallback (should not reach here due to raises above)
        raise RuntimeError(
            f"HTTP ERROR: Failed to get response for model {model} after {attempts} attempts"
        )
    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ""
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r",\s*"
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        result_json = {}
        for i in range(len(arguments)):
            value = match.group(i + 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # value=value.replace('"', '\\"')
            value = value.replace("\\n", "\n")
            result_json[arguments[i]] = value
        return result_json
    @classmethod
    def is_http_response(cls, raw_text: str):
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "HTTP ERROR: Request failed for model" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None
    @classmethod
    def _request_next_action_with_retry(
        cls,
        messages: dict,
        models: List[str],
        max_retries: int = 3,
        base_delay: float = 1.0,
        temperature: float = 0.0,
    ) -> str:
        raw_text = "not defined"
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        current_model_idx = 0
        used_model = models[0] if models else None
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                current_model = (
                    models[current_model_idx]
                    if current_model_idx < len(models)
                    else models[-1]
                )
                used_model = current_model
                # index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text, _ = cls.make_request(
                    messages, model=current_model, temperature=temperature
                )
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not (is_valid):
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = (
                    cls.parse_response(raw_text)
                )
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                # Check if it's a 504 error - switch to next model
                is_504_error = (
                    "504" in error_body
                    or "HTTP ERROR 504" in error_body
                    or "Gateway Timeout" in error_body
                )
                if is_504_error and current_model_idx < len(models) - 1:
                    current_model_idx += 1
                    # Don't count this as an attempt for the next model, so continue without incrementing
                    time.sleep(3)
                    continue
                if attempt < max_retries - 1:
                    delay = base_delay
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name] += 1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name] += 1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name] += 1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    if (
                        "HTTP ERROR" not in error_body
                        and "RATE_LIMIT_EXCEEDED" not in error_body
                        and "RESERVED_TOKEN_PRESENT" not in error_body
                        and "EMPTY_RESPONSE" not in error_body
                        and "TIMEOUT" not in error_body
                        and "NETWORK_ERROR" not in error_body
                        and "HTTP ERROR 429" not in raw_text
                        and "INCOMPLETE_RESPONSE" not in error_body
                    ):
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append(
                            {"role": "user", "content": "observation: " + error_body}
                        )
                    time.sleep(3)
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)
        return (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        )
    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        """Sanitize response text by normalizing field names"""
        text_resp = re.sub("['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("['\"]*observation['\"]*:", "observation:", text_resp)
        text_resp = re.sub("['\"]*tool_call_['\"]*", "tool_call_", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            text_resp = "next_thought: " + text_resp
        if (
            "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_tool_name = (
                text_resp.split("next_tool_name:")[1]
                .split("next_tool_args:")[0]
                .strip()
                .strip("\n")
                .strip("'")
                .strip('"')
                .strip()
            )
            text_resp = re.sub(
                f"next_tool_name:['\" ]*{re.escape(next_tool_name)}['\" ]*",
                "next_tool_name: " + next_tool_name,
                text_resp,
            )
        return text_resp
    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        """
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        """
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        error_msg = ""
        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg = f"Invalid JSON: {next_tool_args}"
            try:
                next_tool_args = cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(
                        tool_name, required=True
                    ),
                    next_tool_args,
                )
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args
    @classmethod
    def is_valid_response(cls, raw_text: str) -> bool:
        if (
            type(raw_text) is dict
            and raw_text.get("error", None) is not None
            and raw_text.get("error") != ""
        ):
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        stripped = raw_text.strip()
        has_next_thought = (
            "next_thought" in raw_text.lower() or "<next_thought>" in raw_text.lower()
        )
        has_next_tool_name = (
            "next_tool_name" in raw_text.lower()
            or "<next_tool_name>" in raw_text.lower()
        )
        has_next_tool_args = (
            "next_tool_args" in raw_text.lower()
            or "<next_tool_args>" in raw_text.lower()
        )
        # Valid endings: JSON format or XML-style tags
        valid_ending = (
            stripped.endswith("}")
            or stripped.endswith("}]")
            or stripped.endswith("</next_tool_args>")
            or stripped.endswith(">")
        )
        if (
            has_next_thought
            and has_next_tool_name
            and has_next_tool_args
            and not valid_ending
        ):
            return False, cls.ErrorType.INCOMPLETE_RESPONSE.name
        if len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        return cls.is_http_response(raw_text)
    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}
    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str | None, Any, Any, str | None]:
        """
        Enhanced parser supporting both single and multi-tool call formats.
        Single format:
            next_thought: ...
            next_tool_name: ...
            next_tool_args: {...}
        Multi-tool format:
            next_thought: ...
            tool_call_1:
                tool_name: ...
                tool_args: {...}
            tool_call_2:
                tool_name: ...
                tool_args: {...}
        """
        error_msg = None
        text_resp = text_resp.strip()
        # Remove observation if present
        if "observation:" in text_resp.lower():
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[
                0
            ].strip()
        text_resp = cls.sanitise_text_resp(text_resp)
        if "Infrastructure is at maximum capacity" in text_resp:
            return None, None, None, "HTTP ERROR Maximum Capacity"
        if "No instances available" in text_resp:
            return None, None, None, "HTTP ERROR NO INSTANCES AVAILABLE"
        next_thought = None
        thought_patterns = [
            r"next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))",
            r"next_thought\s*:\s*(.*?)(?=\ntool_call_)",
            r"next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)",
            r"next_thought\s*:\s*(.*)",
        ]
        for pattern in thought_patterns:
            match = re.search(pattern, text_resp, re.DOTALL | re.IGNORECASE)
            if match:
                next_thought = match.group(1).strip()
                if next_thought and len(next_thought) > 2:
                    break
        if not next_thought:
            next_thought = "Processing request"
        # Check for multi-tool call format (tool_call_1, tool_call_2, etc.)
        tool_call_pattern = r"tool_call_(\d+)\s*:"
        tool_call_matches = list(
            re.finditer(tool_call_pattern, text_resp, re.IGNORECASE)
        )
        if tool_call_matches:
            # Multi-tool call format
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = (
                    tool_call_matches[i + 1].start()
                    if i + 1 < len(tool_call_matches)
                    else len(text_resp)
                )
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                error_msg = (
                    "Multi-tool format detected but no valid tool calls extracted"
                )
                return next_thought, None, None, error_msg
            # Return list of tool names and args
            tool_names = [call["tool_name"] for call in tool_calls]
            tool_args_list = [call["tool_args"] for call in tool_calls]
            if len(tool_names) == 1:
                return next_thought, tool_names[0], tool_args_list[0], error_msg
            else:
                return next_thought, tool_names, tool_args_list, error_msg
        # Try single tool call format (legacy)
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp:
            if (
                text_resp.find("next_thought:")
                < text_resp.find("next_tool_name:")
                < text_resp.find("next_tool_args:")
            ):
                next_tool_name_raw = (
                    text_resp.split("next_tool_name:")[1]
                    .split("next_tool_args:")[0]
                    .strip()
                    .strip("\n")
                )
                next_tool_args_raw = (
                    text_resp.split("next_tool_args:")[1]
                    .strip()
                    .split("next_thought:")[0]
                    .strip()
                    .strip("\n")
                )
                try:
                    # Handle array format
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(
                        next_tool_names, next_tool_args_raw
                    )
                    if isinstance(parsed_args, list):
                        next_tool_args_list = parsed_args
                    else:
                        next_tool_args_list = [parsed_args for _ in next_tool_names]
                    if len(next_tool_names) == 1:
                        return (
                            next_thought,
                            next_tool_names[0],
                            next_tool_args_list[0],
                            error_msg,
                        )
                    else:
                        return (
                            next_thought,
                            next_tool_names,
                            next_tool_args_list,
                            error_msg,
                        )
                except (JSONDecodeError, Exception) as e:
                    error_msg = f"Invalid JSON in tool args: {str(e)}"
                    Utils.log_to_failed_messages(text_resp)
                    return next_thought, None, None, error_msg
        # If we get here, parsing failed
        if "next_thought:" not in text_resp:
            error_msg = "Invalid response. next_thought not found"
        elif "next_tool_name:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. No tool calls found (expected next_tool_name: or tool_call_N:)"
        elif "next_tool_args:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. next_tool_args not found"
        else:
            error_msg = "Invalid response format. Could not parse tool calls."
        Utils.log_to_failed_messages(text_resp)
        return next_thought, None, None, error_msg
    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {
                "role": "system",
                "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else.",
            },
            {"role": "user", "content": json_string},
        ]
        selected_model = QWEN_MODEL_NAME
        retry = 0
        while retry < 5:
            try:
                response, _ = cls.make_request(messages, model=selected_model)
                break
            except Exception as e:
                retry += 1
                other_models = [
                    model for model in AGENT_MODELS if model != selected_model
                ]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
        try:
            response = response.replace("```json", "").strip("```")
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            pass
        return None
    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
        INCOMPLETE_RESPONSE = 10
class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        """
        Limit the number of strings to 1000
        """
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return (
                "\n".join(strings_list[:n])
                + "\n..."
                + f"({len(strings_list)-n} more lines)"
            )
        else:
            return strings
    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        import re
        if isinstance(messages, list):
            text = " ".join(
                str(m.get("content", "") if isinstance(m, dict) else m)
                for m in messages
            )
        else:
            text = messages
        # Split into words and non-word tokens (punctuation, operators, etc.)
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for token in tokens:
            if token.isspace():
                continue  # Whitespace is typically absorbed
            elif len(token) == 1:
                count += 1  # Single chars (punctuation, operators)
            else:
                count += max(1, (len(token) + 2) // 3)
        return count
    @classmethod
    def log_to_failed_messages(cls, text_resp: str):
        with open("../failed_messages.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([text_resp])
    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)
class EnhancedCOT:
    def __init__(self, latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        # Store summaries: key is (start_idx, end_idx) tuple, value is summary string
        self.summaries: dict[tuple[int, int], str] = {}
        # Track which indices have been summarized
        self.summarized_ranges: list[tuple[int, int]] = []
    def _get_summary_for_index(self, idx: int) -> Optional[str]:
        """Get the summary for a given message index if it exists."""
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None
    def _check_and_summarize_if_needed(self):
        """Check if we need to summarize older messages and trigger summarization."""
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep
        if cutoff_idx < self.summarize_batch_size:
            return  # Not enough messages to summarize yet
        oldest_unsummarized = 0
        for start, end in sorted(self.summarized_ranges):
            if start <= oldest_unsummarized < end:
                oldest_unsummarized = end
            elif start > oldest_unsummarized:
                break  # Found a gap, use oldest_unsummarized
        # Only summarize if we have a batch ready and it's before the cutoff
        if oldest_unsummarized >= cutoff_idx:
            return  # All messages before cutoff are already summarized or being kept
        # Calculate the range to summarize (don't go beyond cutoff)
        summarize_start = oldest_unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        # Only summarize if we have a full batch (or at least summarize_batch_size messages)
        # This ensures incomplete batches remain unsummarized
        batch_size = summarize_end - summarize_start
        if batch_size >= self.summarize_batch_size:
            # Check if this range is already summarized
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()
    def _summarize_messages_batch(self, start_idx: int, end_idx: int) -> Optional[str]:
        """Summarize a batch of messages using LLM."""
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        # Build the conversation to summarize
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if thought.is_deleted:
                continue
            # Format the thought and tool call
            assistant_part = f"next_thought: {thought.next_thought}\n"
            assistant_part += f"next_tool_name: {thought.next_tool_name}\n"
            assistant_part += f"next_tool_args: {thought.next_tool_args}\n"
            # Format observation (truncate very long observations for summarization)
            if isinstance(thought.observation, (list, tuple)):
                try:
                    obs_render = json.dumps(
                        list(thought.observation), ensure_ascii=False
                    )
                except Exception:
                    obs_render = str(thought.observation)
            else:
                obs_render = str(thought.observation) if thought.observation else ""
            # Truncate very long observations to avoid token limits during summarization
            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append(
                {
                    "assistant": assistant_part,
                    "user": user_part,
                    "is_error": thought.is_error,
                }
            )
        if not conversation_parts:
            return None
        # Build the prompt for summarization
        conversation_text = ""
        for i, part in enumerate(conversation_parts, 1):
            conversation_text += f"\n--- Step {i} ---\n"
            conversation_text += f"Assistant: {part['assistant']}\n"
            # Observation already truncated to 2000 chars, show more context (up to 1500) for summarization
            user_obs = part["user"]
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conversation_text += f"User: {user_obs}\n"
            if part["is_error"]:
                conversation_text += "[Error occurred]\n"
        summarization_prompt = textwrap.dedent(
            f"""
        You are summarizing a conversation history between an AI agent and its environment.
        Summarize the following conversation steps concisely, focusing on:
        1. Key actions taken (tools used, files modified, tests run)
        2. Important findings or errors encountered
        3. Progress made toward solving the problem
        4. Critical decisions or changes in approach
        Keep the summary concise (2-4 sentences per step) but preserve important details.
        Conversation to summarize:
        {conversation_text}
        Provide a concise summary:
        """
        )
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes conversation history concisely.",
                },
                {"role": "user", "content": summarization_prompt},
            ]
            retry = 0
            while retry < 5:
                try:
                    response, _ = EnhancedNetwork.make_request(
                        messages, model=QWEN_MODEL_NAME, temperature=0.0
                    )
                    return response.strip()
                except Exception as e:
                    retry += 1
                    time.sleep(2)
        except Exception as e:
            return None
        return None
    def is_thought_repeated(self) -> bool:
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if (
            last.next_tool_name == prev.next_tool_name
            and last.next_tool_args == prev.next_tool_args
        ):
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False
    def add_action(
        self, action: EnhancedCOT.Action
    ) -> bool:  # don't add if thought is repeated
        self.thoughts.append(action)
        # Check if we need to summarize older messages
        # Only check when we have enough messages to potentially summarize
        total_thoughts = len(self.thoughts)
        if (
            total_thoughts
            >= self.latest_observations_to_keep + self.summarize_batch_size
        ):
            self._check_and_summarize_if_needed()
        return True
    def to_str(self):
        messages = []
        last_summary_range = None
        # Only include summaries for the last N summarized ranges to keep context bounded
        if self.summarized_ranges:
            allowed_summary_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:])
        else:
            allowed_summary_ranges = set()
        for i, thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                summary = self._get_summary_for_index(i)
                if summary:
                    found_range = False
                    for (start, end), summ in self.summaries.items():
                        if start <= i < end:
                            current_range = (start, end)
                            if current_range not in allowed_summary_ranges:
                                found_range = True
                                break
                            # Only add summary once per range
                            if current_range != last_summary_range:
                                messages.append(
                                    {
                                        "role": "system",
                                        "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]",
                                    }
                                )
                                last_summary_range = current_range
                            found_range = True
                            break  # Found the range, break out of inner loop
                    # Skip individual messages in this range - continue outer loop
                    if found_range:
                        continue
                # If no summary available, show the message as-is (full content)
                assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                # Render list observations as JSON array for the model
                if isinstance(thought.observation, (list, tuple)):
                    try:
                        obs_render = json.dumps(
                            list(thought.observation), ensure_ascii=False
                        )
                    except Exception:
                        obs_render = str(thought.observation)
                else:
                    obs_render = str(thought.observation) if thought.observation else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
            else:
                # Latest observations - always show full content
                if thought.is_error is None or i == len(self.thoughts) - 1:
                    assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render = json.dumps(
                                list(thought.observation), ensure_ascii=False
                            )
                        except Exception:
                            obs_render = str(thought.observation)
                    else:
                        obs_render = str(thought.observation)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error == None and thought.is_error != None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str = (
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render = json.dumps(
                                    list(thought.observation), ensure_ascii=False
                                )
                            except Exception:
                                obs_render = str(thought.observation)
                        else:
                            obs_render = str(thought.observation)
                        user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
        return messages
    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: str,
            next_tool_args: dict,
            observation: list | tuple | str,
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = (
                ";".join(observation) if isinstance(observation, list) else observation
            )
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False
class CodeParseUtil:
    """
    Utility class for parsing and extracting code from files and patches(replace ast.parse functionality to work with any languages powered by LLMs)
    """    
    def __init__(self):
        self._parsers = {}  # Cache for language parsers
    
    def _get_parser(self, language: str):
        if Parser is None or get_language is None:
            return None
        
        if language not in self._parsers:
            try:
                lang_obj = get_language(language)
                if lang_obj is None:
                    return None
                parser = Parser(lang_obj)
                self._parsers[language] = parser
            except Exception as e:
                logger.warning(f"Error creating parser for {language}: {e}")
                return None
        
        return self._parsers[language]
        
    def _is_identifier_node(self, node) -> bool:
        """Check if a node represents an identifier (name/variable)."""
        return "identifier" in node.type.lower()
    
    def _classify_node_type(self, node) -> tuple[str, int | None]:
        node_type_str = node.type.lower()
        if "function" in node_type_str or "method" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("function", i)
            return ("function", None)
        elif "class" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("class", i)
            return ("class", None)
        
        # Not a function or class
        return ("other", None)
    def detect_language(self, source: str, file_path: str | None = None) -> str | None:
        global _codeparse_util_language_cache
        if file_path and not os.path.exists(file_path):
            return None
        if not source or not source.strip():
            return None
        if file_path:
            file_path = os.path.abspath(file_path) if file_path else None
            if file_path and file_path in _codeparse_util_language_cache:
                return _codeparse_util_language_cache[file_path]
        stripped_source = source.strip()
        if len(stripped_source) <= 1000:
            sample = stripped_source
        else:
            first_part = stripped_source[:500]
            last_part = stripped_source[-500:]
            sample = f"{first_part}\n\n... [middle content omitted] ...\n\n{last_part}"
        prompt = f"""Detect the programming language of the following code sample.
        Analyze the code and determine which language it is. Return ONLY one of these three options:
        - "python" if the code is Python
        - "javascript" if the code is JavaScript
        - "other" if it's any other language
        Code sample:
        ```
        {sample}
        ```
        Return ONLY the language name (python, javascript, or other), no other text or explanation."""
        
        retry = 0
        messages = [{"role": "user", "content": prompt}]
        models_to_try = [QWEN_MODEL_NAME, GLM_MODEL_NAME]
        while retry < 3:
            try:
                result, _ = EnhancedNetwork.make_request(
                    messages=messages,
                    model=models_to_try[retry % len(models_to_try)],
                    temperature=0.0
                )
                cleaned = result.strip().lower()
                cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
                cleaned = cleaned.strip('"').strip("'").strip()
                if cleaned in ['python', 'javascript', 'other']:
                    detected_language = cleaned if cleaned != 'other' else None
                else:
                    if 'python' in cleaned:
                        detected_language = 'python'
                    elif 'javascript' in cleaned:
                        detected_language = 'javascript'
                    else:
                        retry += 1
                        if retry < 3:
                            messages.append({"role": "assistant", "content": result})
                            messages.append({"role": "user", "content": "Please return ONLY one word: 'python', 'javascript', or 'other'. No other text."})
                            time.sleep(1)
                        continue
                
                if file_path:
                    _codeparse_util_language_cache[file_path] = detected_language
                
                return detected_language
                        
            except Exception as e:
                logger.warning(f"Error detecting language with LLM (attempt {retry + 1}/3): {e}")
                retry += 1
                if retry < 3:
                    time.sleep(1)
                continue
        
        return None
    def _find_specific_function(self, node, source_lines: list[str], target_qualified: str, target_simple: str, class_name: str = "", parent_node = None) -> dict | None:
        if not node.children: return None
        node_type, name_child_index = self._classify_node_type(node)
        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if not name and parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if name:
                new_class_name = f"{class_name}.{name}" if class_name else name
                for child in node.children:
                    result = self._find_specific_function(child, source_lines, target_qualified, target_simple, new_class_name, node)
                    if result is not None: return result
        elif node_type == "function":
            name = None
            internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    internal_name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if not name:
                name = internal_name
            if name:
                qualified_name = f"{class_name}.{name}" if class_name else name
                is_qualified_target = '.' in target_qualified
                is_match = qualified_name == target_qualified or (not is_qualified_target and name == target_simple)
                if is_match:
                    at_start = node.start_point[0]
                    for i in range(at_start - 1, -1, -1):
                        if source_lines[i].strip().startswith('@'): at_start = i
                        elif source_lines[i].strip(): break
                    return {'start_line': at_start + 1, 'end_line': node.end_point[0] + 1}
            for child in node.children:
                result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
                if result is not None: return result
        for child in node.children:
            result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
            if result is not None: return result
        return None
    
    def _get_definition_line(self, node, source_lines: list[str], include_at_lines: bool = False) -> tuple[str, bool]:
        at_start = node.start_point[0]
        starts_with_at = False
        if include_at_lines:
            for i in range(at_start - 1, -1, -1):
                if source_lines[i].strip().startswith('@'): 
                    at_start = i
                    starts_with_at = True
                elif source_lines[i].strip(): break
        if at_start < len(source_lines):
            line = source_lines[at_start].strip()
            if line.startswith('@') and not include_at_lines:
                starts_with_at = True
                for i in range(at_start + 1, len(source_lines)):
                    next_line = source_lines[i].strip()
                    if next_line and not next_line.startswith('@'):
                        return (next_line, False)
            if line.startswith('@'):
                starts_with_at = True
            return (line, starts_with_at)
        return ("", False)
    
    def _extract_skeleton_structure(self, node, source_lines: list[str], classes: list, functions: list, class_name: str = "", current_class_methods: list = None, parent_node = None, processed_nodes = None, processed_classes = None):
        if processed_nodes is None:
            processed_nodes = set()
        if processed_classes is None:
            processed_classes = set()
        node_key = (node.start_point[0], node.start_point[1], node.end_point[0], node.end_point[1])
        if node_key in processed_nodes:
            return
        processed_nodes.add(node_key)
        if not node.children: return
        node_type, name_child_index = self._classify_node_type(node)
        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if not name and parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if name:
                class_line = node.start_point[0] + 1
                class_key = (name, class_line)
                if class_key not in processed_classes:
                    processed_classes.add(class_key)
                    def_line, starts_with_at = self._get_definition_line(node, source_lines)
                    if starts_with_at:
                        return
                    methods = []
                    new_class_name = f"{class_name}.{name}" if class_name else name
                    for child in node.children:
                        self._extract_skeleton_structure(child, source_lines, classes, functions, new_class_name, methods, node, processed_nodes, processed_classes)
                    classes.append({'name': name, 'line': class_line, 'definition': def_line, 'methods': methods})
                return
        elif node_type == "function":
            name = None
            internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    internal_name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
            if parent_node:
                for child in parent_node.children:
                    if self._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = line[name_start[1]:name_end[1]].strip() if name_start[0] == name_end[0] else line[name_start[1]:].strip()
                            if name: break
            if not name:
                name = internal_name
            if name:
                def_line, starts_with_at = self._get_definition_line(node, source_lines)
                if starts_with_at:
                    return
                func_info = {'name': name, 'line': node.start_point[0] + 1, 'definition': def_line}
                if current_class_methods is not None:
                    current_class_methods.append(func_info)
                else:
                    functions.append(func_info)
                return
        for child in node.children:
            self._extract_skeleton_structure(child, source_lines, classes, functions, class_name, current_class_methods, node, processed_nodes, processed_classes)
    
    def get_file_skeleton(self, file_path: str) -> str:
        if not os.path.exists(file_path): return f"File not found: {file_path}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f: source = f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {e}"
        if not source: return f"File {file_path} is empty"
        if Parser is None: return "tree-sitter parser not available"
        try:
            source_bytes, source_lines = bytes(source, 'utf8'), source.splitlines()
            language = self.detect_language(source, file_path=file_path)
            if not language: return f"Could not detect language for {file_path}"
            parser = self._get_parser(language)
            if parser is None: return f"Could not create parser for language: {language}"
            tree = parser.parse(source_bytes)
            classes, standalone_functions = [], []
            self._extract_skeleton_structure(tree.root_node, source_lines, classes, standalone_functions, "", None, None, set(), set())
            result = [f"File: {file_path}\n"]
            for cls in classes:
                result.append(f"\n{cls['line']}| {cls['definition']}")
                for method in cls['methods']:
                    result.append(f"{method['line']}|   {method['definition']}")
            if standalone_functions:
                if classes:
                    result.append("")
            for func in standalone_functions:
                result.append(f"{func['line']}| {func['definition']}")
            if not classes and not standalone_functions:
                result.append("\nNo classes or functions found in this file.")
            return "\n".join(result)
        except Exception as e:
            return f"Error generating skeleton for {file_path}: {e}"
    
    def get_function_body(self, file_path: str, function_name: str, add_line_numbers: bool = False) -> str:
        if not function_name or not os.path.exists(file_path): return ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f: source = f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""
        if not source or Parser is None: return ""
        try:
            source_bytes, source_lines = bytes(source, 'utf8'), source.splitlines()
            language = self.detect_language(source, file_path=file_path)
            if not language: return ""
            parser = self._get_parser(language)
            if parser is None: return ""
            tree = parser.parse(source_bytes)
            target_qualified, target_simple = function_name, function_name.split('.')[-1]
            func_info = self._find_specific_function(tree.root_node, source_lines, target_qualified, target_simple)
            if func_info is None: return ""
            start_idx, end_idx = func_info['start_line'] - 1, func_info['end_line'] - 1
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines):
                body_lines = source_lines[start_idx:end_idx + 1]
                if add_line_numbers:
                    return '\n'.join(f"{start_idx + i + 1}| {line}" for i, line in enumerate(body_lines))
                return '\n'.join(body_lines)
        except Exception as e:
            logger.warning(f"Error finding function {function_name} in {file_path}: {e}")
        return ""
    
class SearchManager:
    def __init__(self):
        pass
    def search_in_all_files(self, grep_search_command: str) -> str:
        """
        Performs grep search across all files in the codebase.
        Arguments:
            grep_search_command: grep search command to execute
        Returns:
            Search results with file paths and line numbers, or error message
        """
        # Validate that the command is a grep command
        cmd_stripped = grep_search_command.strip()
        if not cmd_stripped.startswith("grep"):
            return f"Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        # Execute the grep command
        result = subprocess.run(
            ["bash", "-c", grep_search_command], capture_output=True, text=True
        )
        # Check for command execution errors (return codes other than 0 and 1)
        # Note: grep returns 1 when no matches are found, which is not an error
        if result.returncode > 1:
            error_msg = (
                result.stderr.strip() if result.stderr.strip() else "Unknown error"
            )
            return f"Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout
        lines = [l for l in output.splitlines()]
        if not lines:
            return f"No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return f"Search results are too long. Please refine your search term into more specific terms."
        else:
            return output
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file.
        Arguments:
            file_path: target file for pattern matching
            search_term: text pattern to find
        Returns:
            Matching locations with line numbers and context, or error message
        """
        def extract_matches(
            file_path: str, search_term: str, *, max_output_lines: int = 1000
        ) -> str:
            """
            Return the source code around matches showing ±20 lines of context.
            The final output is truncated with `limit_strings` to avoid excessive verbosity.
            """
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_lines = f.read().splitlines()
            except Exception as e:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                    f"Error reading '{file_path}': {e}",
                )
            # Identify all lines that contain the search term.
            escaped_search_term = re.escape(search_term)
            match_lines = [
                idx + 1
                for idx, line in enumerate(source_lines)
                if escaped_search_term in line
            ]
            if not match_lines:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                    f"'{search_term}' not found in file '{file_path}'",
                )
            # Show ±20 lines around each match
            chunks: list[str] = []
            context_lines = 20
            seen_ranges = set()  # Track ranges to avoid duplicates
            for ln in match_lines:
                start_line = max(1, ln - context_lines)
                end_line = min(len(source_lines), ln + context_lines)
                range_key = (start_line, end_line)
                if range_key not in seen_ranges:
                    seen_ranges.add(range_key)
                    context_src = "\n".join(source_lines[start_line - 1 : end_line])
                    chunks.append(f"(lines {start_line}-{end_line}):\n{context_src}")
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)
        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return f"Search results are too long. Please refine your search term into more specific terms."
        else:
            return output
class TestManager:
    def __init__(
        self,
        runner_hint: str | None = None,
        runner_mode_hint: str | None = None,
        file_ops: "FileOperationsUtil" = None,
    ):
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.file_ops = file_ops
    def run_code(self, content: str, file_path: str, generated_test_files: list) -> str:
        """
        Runs code by saving it to a file and executing it (supports multiple languages).
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in
            generated_test_files: list to track generated test files
        Returns:
            Standard output from code execution or error message
        """
        python_extensions = (".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")
        if file_path.endswith(python_extensions):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
        self.file_ops.save(file_path, content)
        generated_test_files.append(file_path)
        try:
            run_command = llm_select_run_command_for_file(file_path)
            logger.info(f"Running command in run_code: {run_command}")
            result = subprocess.run(
                run_command, capture_output=True, text=True, check=False, timeout=60
            )
        except ValueError as e:
            return f"Error: {e}"
        if result.returncode != 0:
            return f"Error running code: {result.stderr}"
        observation = f"{result.stdout}\n"
        return observation
class FileSystemManager:
    def __init__(self):
        pass
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
        search_in_file_callback=None,
    ) -> str:
        
        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            """Helper method to add line numbers to content."""
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return "\n".join(numbered_lines)
        # If search term is provided, use callback to search in file
        if search_term and search_in_file_callback:
            return search_in_file_callback(file_path, search_term)
        # Read file content
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:
                # Read specific line range
                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, start_idx + 1)
                else:
                    result = content
            else:
                # Read entire file
                content = f.read()
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, 1)
                else:
                    result = content
        # Apply limit if specified
        return Utils.limit_strings(result, n=limit) if limit != -1 else result
    def list_directory_structure(self, file_path: str = ".", depth: int = 1) -> str:
        """
        Lists the directory structure of the repository.
        Arguments:
            file_path: the directory path to list (default: ".")
            depth: maximum depth to traverse (default: 1)
        Returns:
            Formatted directory tree structure
        """
        def tree(path: str, prefix: str = "", current_depth: int = 0) -> list:
            if current_depth > depth:
                return []
            try:
                ignore = {
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    "node_modules",
                    ".tox",
                    ".venv",
                    "venv",
                    ".eggs",
                }
                items = sorted(os.listdir(path))
                dirs = [
                    i
                    for i in items
                    if not i.startswith(".")
                    and i not in ignore
                    and not i.endswith(".egg-info")
                    and os.path.isdir(os.path.join(path, i))
                ]
                files = [
                    i
                    for i in items
                    if not i.startswith(".") and os.path.isfile(os.path.join(path, i))
                ]
                entries = []
                for i, d in enumerate(dirs):
                    is_last = i == len(dirs) - 1 and not files
                    entries.append(f"{prefix}{'└── ' if is_last else '├── '}{d}/")
                    if current_depth < depth:
                        entries.extend(
                            tree(
                                os.path.join(path, d),
                                prefix + ("    " if is_last else "│   "),
                                current_depth + 1,
                            )
                        )
                for i, f in enumerate(files):
                    entries.append(
                        f"{prefix}{'└── ' if i == len(files) - 1 else '├── '}{f}"
                    )
                return entries
            except:
                return [f"{prefix}[Error]"]
        lines = tree(file_path, "", 0)
        return (
            f"Directory structure (depth={depth}):\n{file_path}/\n"
            + "\n".join(lines)
            + f"\n\n{sum(1 for l in lines if l.rstrip().endswith('/'))}-dirs, {sum(1 for l in lines if not l.rstrip().endswith('/') and '[' not in l)}-files"
        )
class FileOperationsUtil:
    def __init__(self, new_files_created: list):
        self.new_files_created = new_files_created
        self.file_system_manager = None
        self.search_manager = None
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
        structural_truncation: bool = False,
    ) -> str:
        """
        Get file content with optional filtering and formatting.
        Arguments:
            file_path: path to the file
            search_start_line: start line for range filtering
            search_end_line: end line for range filtering
            search_term: term to search for
            limit: maximum number of lines to return
            add_line_numbers: whether to add line numbers
        Returns:
            File content as string
        """
        # Create a lambda that calls search_manager
        search_callback = lambda fp, st: self.search_manager.search_in_file(fp, st)
        return self.file_system_manager.get_file_content(
            file_path=file_path,
            search_start_line=search_start_line,
            search_end_line=search_end_line,
            search_term=search_term,
            limit=limit,
            add_line_numbers=add_line_numbers,
            search_in_file_callback=search_callback,
        )
    def save(self, file_path: str, content: str) -> str:
        """
        Save content to file after validating syntax.
        Arguments:
            file_path: path to save the file
            content: content to write
        Returns:
            Success message
        Raises:
            Error if syntax check fails
        """
        with open(file_path, "w") as file:
            file.write(content)
        self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"
    def set_managers(self, file_system_manager, search_manager):
        """Set manager references after initialization to avoid circular dependencies."""
        self.file_system_manager = file_system_manager
        self.search_manager = search_manager
class CodeEditManager:
    def __init__(self, file_ops: "FileOperationsUtil" = None):
        self.file_ops = file_ops
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files with syntax validation.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Returns:
            Operation status - success confirmation with context or detailed error with guidance
        """
        def add_context_to_similar_match(
            original_content: str, formatted_match: str, context_lines: int = 2
        ) -> str:
            """Add context lines around a similar match for better understanding."""
            lines = original_content.split("\n")
            # Extract the actual content from the formatted match (remove the description part)
            match_lines = formatted_match.split("\n")
            if len(match_lines) < 2:
                return formatted_match
            # Skip the description line (e.g., "Lines 45-47: ..." or "Line 23: ...")
            actual_content_lines = match_lines[1:]
            actual_content = "\n".join(actual_content_lines)
            # Find where this content appears in the original file
            best_match_start = -1
            best_similarity = 0
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i : i + len(actual_content_lines)]
                candidate_content = "\n".join(candidate_lines)
                import difflib
                similarity = difflib.SequenceMatcher(
                    None, actual_content.strip(), candidate_content.strip()
                ).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match  
            
            start_line = max(0, best_match_start - context_lines)
            end_line = min(
                len(lines), best_match_start + len(actual_content_lines) + context_lines
            )
            # Build the context with line numbers
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = (
                    ">>> "
                    if best_match_start
                    <= i
                    < best_match_start + len(actual_content_lines)
                    else "    "
                )
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            # Extract original description
            description = (
                match_lines[0]
                if match_lines
                else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            )
            return f"{description}\n" + "\n".join(context_lines_list)
        def find_most_similar_content(
            original_content: str, search_string: str, max_results: int = 3
        ) -> list[tuple[float, str]]:
            """Find the most similar content chunks to the search string."""
            import difflib
            # Split content into meaningful chunks
            lines = original_content.split("\n")
            # Try different chunk sizes to find the best match
            chunks = []
            # Individual lines
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
            # Multi-line chunks (3-5 lines) for better context
            search_lines = search_string.split("\n")
            target_chunk_size = max(3, len(search_lines))
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i : i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append(
                        (f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content)
                    )
            # Calculate similarity scores
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(
                    None, search_string.strip(), chunk_content
                ).ratio()
                if ratio > 0.3:  # Only include reasonably similar content
                    similarities.append((ratio, chunk_desc, chunk_content))
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [
                (ratio, f"{desc}\n{content}")
                for ratio, desc, content in similarities[:max_results]
            ]
        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        original = self.file_ops.get_file_content(file_path, limit=-1)
        match original.count(search):
            case 0:
                # Find most similar content to help LLM correct the search string
                similar_matches = find_most_similar_content(original, search, 1)
                error_msg = f"Error: search string not found in file {file_path}."
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        # Add context lines around the match for better understanding
                        content_with_context = add_context_to_similar_match(
                            original, content, context_lines=2
                        )
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content_with_context}"
                else:
                    error_msg += " No similar content found. Please check the file content and provide the exact code you want to replace."
                return error_msg
            case 1:
                new_content = original.replace(search, replace)
                try:
                    self.file_ops.save(file_path, new_content)
                    replace_pos = new_content.find(replace)
                    if replace_pos != -1:
                        lines = new_content.split("\n")
                        # Find which line number the replacement starts at
                        chars_so_far = 0
                        replace_line_start = 0
                        for i, line in enumerate(lines):
                            if chars_so_far + len(line) >= replace_pos:
                                replace_line_start = i
                                break
                            chars_so_far += len(line) + 1  # +1 for newline
                        # Calculate how many lines the replacement spans
                        replace_lines_count = replace.count("\n") + 1
                        replace_line_end = replace_line_start + replace_lines_count - 1
                        # Extract 20 lines before and after
                        start_line = max(0, replace_line_start - 5)
                        end_line = min(len(lines), replace_line_start + 5)
                        context_lines = []
                        for i in range(start_line, end_line):
                            line_num = i + 1
                            # Mark the edited lines with >>> prefix
                            if replace_line_start <= i <= replace_line_end:
                                prefix = ">>> "
                            else:
                                prefix = "    "
                            context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                        context = "\n".join(context_lines)
                        return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n{context}"
                    else:
                        return "ok, code edit applied successfully"
                except Exception as e:
                    return f"Error: syntax error in file {file_path}. {str(e)}"
            case num_hits:
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
# ============================  ToolManagers   ======================================
class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}
    def __init__(self, **kwargs):
        pass
    def _save(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as file:
            file.write(content)
        # self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"
    def get_tool_docs(self) -> str:
        return "\n\n".join(
            [
                json.dumps(tool_metadata, ensure_ascii=False)
                for _, tool_metadata in self.TOOL_LIST.items()
            ]
        )
    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            ):
                required.append(param.name)
            type_hint = (
                str(param.annotation) if param.annotation != param.empty else "string"
            )
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(
                    f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}"
                )
            # Special handling for list[str] / List[str] annotations so that the
            # generated JSON schema correctly represents an array of strings.
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description,
                }
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description,
            }
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters,
        }
        return tool_schemas
    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method
    def tool(fn):
        def wrapper(self, *args, **kwargs):
            # Use .get() with default 0 to handle methods not in TOOL_LIST
            self.tool_invocations[fn.__name__] = (
                self.tool_invocations.get(fn.__name__, 0) + 1
            )
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                # Initialize tool_failure entry if not present
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {
                        j: 0 for j in self.Error.ErrorType.__members__
                    }
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message
        # Preserve original function metadata
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper
    @classmethod
    def get_tool_args_for_tool(
        self, tool_name: str, required_only: bool = False
    ) -> list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(self.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return self.TOOL_LIST[tool_name]["input_schema"]["required"]
    def get_final_git_patch(self) -> str:
        """
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        """
        try:
            command = f"""
            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore
            git add .
            git diff --cached > .patch.txt
            cat .patch.txt
            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            output = subprocess.run(
                ["bash", "-c", command], timeout=30, capture_output=True, text=True
            )
            # output = output.stdout.decode("utf-8") + '\n' + output.stderr.decode("utf-8")
            return output.stdout
        except Exception as e:
            return f"Error generating git patch: {e}"
    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14
        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message
class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(
        self,
        available_tools: Optional[list[str]] = [],
        runner_hint: str | None = None,
        runner_mode_hint: str | None = None,
        initial_checkpoint=None,
        problem_statement: str = None,
        should_review: bool = True,
    ):
        self.new_files_created = []
        self.available_tools = available_tools
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.generated_test_files = []
        self.initial_checkpoint = initial_checkpoint
        self.observation_dir = ".observation"
        self.problem_statement = problem_statement
        self.code_chunks_cache = {}
        self.repo_dir = "."
        self.saved_observation_counter = 0
        if should_review:
            self.is_reviewed = False
            self.file_by_file_reviewed = False
        else:
            self.is_reviewed = True
            self.file_by_file_reviewed = True
        self.number_of_reviews = 0
        # Initialize hypothesis and strategy tracking
        self.hypothesis_counter = 0
        self.hypotheses = []
        self.strategy_counter = 0
        self.strategies = []
        # Create observation directory if it doesn't exist
        os.makedirs(self.observation_dir, exist_ok=True)
        # Initialize file operations utility
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        # Initialize managers
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.test_manager = TestManager(
            runner_hint=runner_hint,
            runner_mode_hint=runner_mode_hint,
            file_ops=self.file_ops,
        )
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
        # Set manager references in file_ops for cross-manager operations
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.TOOL_LIST = {}
        # Check all classes in the method resolution order (MRO) to include inherited tools
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if (
                        available_tools is not None and name not in available_tools
                    ):  # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {
            k: {j: 0 for j in self.Error.ErrorType.__members__}
            for k in self.TOOL_LIST.keys()
        }
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(
            file_path=file_path, search=search, replace=replace
        )
    @EnhancedToolManager.tool
    def list_directory_structure(self, file_path: str = ".", depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            file_path: the directory path to list (default: ".")
            depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(
            file_path=file_path, depth=depth
        )
    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str) -> str:
        """
        Runs any code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
        """
        return self.test_manager.run_code(
            content=content,
            file_path=file_path,
            generated_test_files=self.generated_test_files,
        )
    @EnhancedToolManager.tool
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
    ) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self.file_ops.get_file_content(
            file_path,
            search_start_line,
            search_end_line,
            search_term,
            add_line_numbers=True,
            limit=1000,
        )
    
    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files, excluding agent files.
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(self, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            # Exclude all files in .observation directory
            observation_dir = getattr(self, "observation_dir", ".observation")
            if os.path.exists(observation_dir):
                try:
                    for root, dirs, files in os.walk(observation_dir):
                        for file in files:
                            file_path = os.path.relpath(os.path.join(root, file))
                            exclude.add(file_path)
                except Exception:
                    pass
            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )
            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    def _save_large_observation(self, observation: str, tool_name: str) -> str:
        """
        Save a large observation to a file in .observation directory and return the file path.
        """
        self.saved_observation_counter += 1
        filename = f"observation_{self.saved_observation_counter}_{tool_name}_{int(time.time())}.txt"
        file_path = os.path.join(self.observation_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(observation)
            return file_path
        except Exception as e:
            logger.error(f"Failed to save observation to {file_path}: {e}")
            return f"Error: Failed to save observation: {e}"
    @EnhancedToolManager.tool
    def get_file_skeleton(self, file_path: str) -> str:
        """
        Retrieves the skeleton of a file
        Arguments:
            file_path: filesystem path to target file.
        Output:
            The skeleton of the file.
        """
        try:
            code_parse_util = CodeParseUtil()
            return code_parse_util.get_file_skeleton(file_path)
        except Exception as e:
            logger.error(f"Failed to get file skeleton for {file_path}: {e}")
            return f"Error: Failed to get file skeleton: {e}"
    @EnhancedToolManager.tool
    def get_function_body(self, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve (supports both qualified names like "ClassName.method_name" and simple names like "method_name").
        Returns:
            The complete function body including decorators, or empty string if function not found.
        """
        code_parse_util = CodeParseUtil()
        return code_parse_util.get_function_body(file_path, function_name, add_line_numbers=True)
    @EnhancedToolManager.tool
    def search_in_all_files_content(self, grep_search_command: str) -> str:
        """
        Performs grep search across all files in the codebase
        Arguments:
            grep_search_command: grep search command to locate (e.g., "grep <your grep command>").
        Output:
            locations where pattern was found with file paths and line numbers
        """
        return self.search_manager.search_in_all_files(grep_search_command)
    @EnhancedToolManager.tool
    def finish_find_files_to_fix(self, files: List[str]):
        """
        Signals completion of the file finding workflow execution
        Arguments:
            files: The list of files to fix.
        """
        self.files_to_fix = files
        return files
    @EnhancedToolManager.tool
    def finish(self):
        '''
        Signals completion of the current workflow execution
        Arguments:
            None
        '''
        return "finish"
# ============================  Create Task Functions   ======================================
def is_all_tests_passed(output: str) -> bool:
    check_all_tests_passed_prompt = """
    Check the test output and tell me if all the tests passed successfully or there is any failure or error.
    This is the output:
    ```
    {output}
    ```
    Return only "true" or "false".
    """
    retry = 1
    while retry < 3:
        try:
            result, _ = EnhancedNetwork.make_request(
                messages=[
                    {
                        "role": "user",
                        "content": check_all_tests_passed_prompt.format(output=output),
                    }
                ],
                model=QWEN_MODEL_NAME,
            )
            if result.lower() == "true":
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"[IS_ALL_TESTS_PASSED] Exception: {e}")
            retry += 1
            time.sleep(2)
    return False
def llm_select_run_command_for_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    retry = 0
    while retry < 3:
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""
                    I'd like you to respond with the command to run this file. Make your command as simple as possible.
                    ```
                    {file_path}
                    {file_content}
                    ```
                    You must respond in JSON format:
                    ```
                    {{
                        "command": ["bbb", "aaa.js", "ccc", "ddd"]
                    }}
                    ```
                    """,
                }
            ]
            raw_text, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            json_result = json.loads(
                raw_text.replace("```json", "").replace("```", "").strip()
            )
            return json_result.get("command")
        except Exception as e:
            time.sleep(2)
            retry += 1
def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    def extract_file_names_using_llm(initial_solution: str) -> list:
        retry = 0
        while retry < 3:
            try:
                file_names_prompt = f"""
                Extract the file names from the initial solution. Return only the file names in a list only.
                This is the initial solution:
                ```
                {initial_solution}
                ```
                Return only the file names in a list.
                Example:
                ```
                ["a.py", "b.js"]
                ```
                """
                result, _ = EnhancedNetwork.make_request(
                    messages=[{"role": "user", "content": file_names_prompt}],
                    model=QWEN_MODEL_NAME,
                )
                return json.loads(
                    result.replace("```json", "").replace("```", "").strip()
                )
            except Exception as e:
                retry += 1
                time.sleep(3)
        return []
    if not initial_solution.strip():
        return []
    file_names = extract_file_names_using_llm(initial_solution)
    created_files = []
    current_file, content = None, []
    def write_file():
        if current_file and content:
            path = os.path.join(base_dir, current_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                file_content = "\n".join(content)
                # Preserve structure, only strip trailing whitespace
                file_content = (
                    file_content.rstrip() + "\n"
                    if file_content.strip()
                    else file_content
                )
                f.write(file_content)
            created_files.append(path)
    filename_set = set(file_names)
    for fname in file_names:
        # Also add just the filename part (in case line has "file.js" but file_names has "src/file.js")
        filename_set.add(fname.split("/")[-1])
    for line in initial_solution.split("\n"):
        stripped = line.strip()
        # Check if this line exactly matches any extracted filename
        if stripped in filename_set:
            write_file()
            # Use the original filename from file_names if available, otherwise use stripped
            current_file = next(
                (
                    f
                    for f in file_names
                    if f == stripped
                    or f.endswith("/" + stripped)
                    or f.split("/")[-1] == stripped
                ),
                stripped,
            )
            current_file, content = current_file, []
        elif current_file:
            content.append(line)
    write_file()
    return created_files
def clean_code_response(response: str) -> str:
    """Clean code response by removing markdown code blocks for any language"""
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response
def process_create_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id
    def get_files_to_modify(problem_statement: str) -> str:
        """Get initial structure from current directory"""
        tool_manager = FixTaskEnhancedToolManager(
            available_tools=[
                "get_file_content",
                "list_directory_structure",
                "finish_find_files_to_fix",
            ]
        )
        FIND_FILES_TO_MODIFY = textwrap.dedent(
            """
            You are a helpful assistant that finds the files to modify related to the problem statement.
            You must check the directory structure using `list_directory_structure` tool and then determine which files are needed for the problem statement.
            You must then use the `finish_find_files_to_fix` tool to signal the completion of the file finding workflow execution.
            You have access to the following tools:-
            {tools_docs}
            {format_prompt}
            """
        ).format(tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT)
        try:
            cot = EnhancedCOT(latest_observations_to_keep=10, summarize_batch_size=10)
            instance_prompt = f"Problem Statement:\n{problem_statement}"
            result = execute_agent_workflow(
                cot,
                tool_manager,
                FIND_FILES_TO_MODIFY,
                instance_prompt,
                20,
                300,
                [GLM_MODEL_NAME, KIMI_MODEL_NAME],
                finish_tool_name="finish_find_files_to_fix",
                log_prefix="FINISH_FIND_FILES_TO_MODIFY",
                reject_observation_token_threshold=50000,
                save_observation_to_file_token_threshold=50000,
            )
            if not result:
                return ""
            if not isinstance(result, list):
                result = [result]
            contents = []
            for file_path in result:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        contents.append(f"{file_path}\n{{\n{f.read()}\n}}")
                except Exception as e:
                    logger.error(f"Failed to open file {file_path}: {e}")
            return "\n\n".join(contents)
        except Exception as e:
            logger.error(f"Error in get files to modify: {e}")
            return ""
    total_timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    problem_statement = input_dict.get("problem_statement", "")
    tool_manager = EnhancedToolManager()
    start_time = time.time()
    initial_structure = get_files_to_modify(problem_statement)
    timeout = total_timeout - (time.time() - start_time) - 60
    success_count = 0
    initial_solutions = []
    s_time = time.time()
    while success_count < 1:
        if time.time() - s_time > timeout:
            break
        if time.time() - s_time > 800 and len(initial_solutions) == 0:
            break
        initial_solution, _ = single_process_create_task(
            problem_statement, initial_structure
        )
        if initial_solution is not None:
            success_count += 1
            os.system("git reset --hard")
            extract_and_write_files(initial_solution)
            patch = tool_manager.get_final_git_patch()
            return patch
        time.sleep(10)
    temperature = 0.0
    initial_solution, _ = basic_approach(
        initial_structure, problem_statement, temperature=temperature
    )
    print(f"Initial solution in process_create_task: {initial_solution}")
    if initial_solution is not None:
        os.system("git reset --hard")
        extract_and_write_files(initial_solution)
        patch = tool_manager.get_final_git_patch()
        return patch
    
    elapsed_time = time.time() - s_time
    return fix_task_solve_workflow(problem_statement, timeout=total_timeout - elapsed_time - 60, run_id_1=run_id, enhancement=enhancement, should_review=False)
def generate_initial_solution(
    problem_statement: str, initial_structure: str, temperature: float = 0.7
) -> str:
    print("[GENERATE_INITIAL_SOLUTION] Starting solution generation")
    GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = (
        textwrap.dedent(
            """
        You are an expert software engineer. Your task is to generate a complete, working solution for the given problem statement.
        Strict Requirements:
        1. Output the full content of files along with their file names. You **MUST** output the **file name** along with file content.
        2. Do not include explanations, comments, or markdown formatting in the main code.
        3. Use only standard libraries and frameworks (no external libraries).
        4. Implement all required classes and functions exactly with the same names as in the initial code stub.
        5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
        6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
        7. The solution must be executable as-is with no placeholders or TODOs.
        8. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
        9. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
        Return only the final code.
        Response Examples:
        ```python
        a.py
        {{content}}
        b.py
        {{content}}
        ```
        """
        )
    )
    INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated code for potential infinite loops and provide a corrected version if issues are found.
        CRITICAL INFINITE LOOP DETECTION:
        1. Check for while True: loops without guaranteed exit conditions
        2. Verify all while loops have clear termination conditions
        3. Ensure recursive functions have proper base cases
        4. Look for loops that depend on external state that might never change
        5. Check for patterns that could lead to infinite iteration.
        If you find potential infinite loops:
        - Provide a corrected version of the code
        - Ensure all loops have finite termination conditions
        - Add reasonable iteration limits or timeout mechanisms where appropriate
        If no infinite loops are detected:
        - Return the original code unchanged
        STRICT REQUIREMENT: Return the final code along with file names. Do not include any explanations, comments, or additional text.
        example:
        ```python
        a.py
        {{content}}
        b.py
        {{content}}
        ```
        """
    )
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT,
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial structure:\n{initial_structure}\nGenerate the complete and correct implementation in files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
        },
    ]
    selected_model = QWEN_MODEL_NAME
    print("[GENERATE_INITIAL_SOLUTION] Requesting code generation from model")
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(
                code_generation_messages, model=selected_model, temperature=temperature
            )
            if isinstance(result, tuple):
                code_response, _ = result
            else:
                code_response = result
            loop_check_messages = [
                {"role": "system", "content": INFINITE_LOOP_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final code.",
                },
            ]
            result2 = EnhancedNetwork.make_request(
                loop_check_messages, model=selected_model
            )
            if isinstance(result2, tuple):
                loop_check_response, _ = result2
            else:
                loop_check_response = result2
            # Clean up the final response (use compat response as it's the final validated version)
            solution = clean_code_response(loop_check_response)
            return solution
        except Exception as e:
            retry += 1
            time.sleep(2)
    if retry >= 10:
        return ""
    return ""
def basic_approach(
    initial_structure: str, problem_statement: str, temperature: float = 0.0
) -> tuple[str, str] | tuple[None, None]:
    initial_solution = generate_initial_solution(
        problem_statement, initial_structure, temperature
    )
    if not initial_solution:
        return (None, None)
    created_files = extract_and_write_files(initial_solution)
    test_cases = generate_single_testset(
        problem_statement, str(created_files), initial_structure, temperature
    )
    if not test_cases:
        return (None, None)
    test_files = extract_and_write_files(test_cases)
    for file in test_files:
        try:
            # Get the appropriate command for the file type
            run_command = llm_select_run_command_for_file(file)
            result = subprocess.run(
                run_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            return (None, None)
        except ValueError as e:
            return (None, None)
        except Exception as e:
            return (None, None)
        if not is_all_tests_passed(result.stdout):
            return (None, None)
    return (initial_solution, test_cases)
def generate_single_testset(
    problem_statement: str,
    files_to_test: str,
    initial_structure: str,
    temperature: float = 0.0,
) -> str:
    """Generate a single test set and return testcode as string"""
    GENERATE_TESTCASES_PROMPT = textwrap.dedent(
        """
        You are an expert testcase developer.
            Important points:-
            - Follow the best practices and conventions of the language of the code skeleton.
            - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
            - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
            - Use the only built-in testing framework for the language of the code skeleton. **MUST** use the built-in testing framework.
                - For python, use `unittest` to write a test.
                - For javascript, **MUST** use **`node:test` and `node:assert`** to write a test.
                - For other languages, use built-in test frameworks as well.
            You must respond directly with the test cases in the following format.
            =========TEST_CASES
            <<test cases>>
            Do not include anything else. For Example (JavaScript):
            =========TEST_CASES
            import { test } from 'node:test';
            import assert from 'node:assert/strict';
            import { main_func } from './main_module.js';
            test('main_func should return expected output', () => {
                assert.strictEqual(main_func(), 'expected_output');
            });
        """
    )
    retry = 0
    test_generation_messages = [
        {"role": "system", "content": GENERATE_TESTCASES_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nInitial structure:\n{initial_structure}\n\nGenerate the complete and correct testcases.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```\n```javascript\ntest_a.js\ncontents of test_a.js\n\ntest_b.js\ncontents of test_b.js\n```",
        },
    ]
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(
                test_generation_messages, model=selected_model, temperature=temperature
            )
            if isinstance(result, tuple):
                testcode_response, _ = result
            else:
                testcode_response = result
            testcases = clean_code_response(testcode_response)
            # Safety check for empty testcases
            if not testcases or not testcases.strip():
                retry += 1
                continue
            lines = testcases.split("\n")
            if not lines or len(lines) == 0:
                retry += 1
                test_generation_messages.append(
                    {"role": "assistant", "content": testcode_response}
                )
                test_generation_messages.append(
                    {
                        "role": "user",
                        "content": f"Include file name in the response. example:\n```python\ntest_a.py\n{{content}}\n\ntest_b.py\n{{content}}\n```\n```javascript\ntest_a.js\n{{content}}\n\ntest_b.js\n{{content}}\n```",
                    }
                )
                continue
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_single_testset: {e}")
            other_models = [model for model in AGENT_MODELS if model != selected_model]
            if other_models:
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return ""
def single_process_create_task(
    problem_statement: str, initial_structure: str
) -> tuple[str, str] | tuple[None, None]:
    BASIC_APPROACH_RETRY = 10
    min_temperature = 0.1
    max_temperature = 1.2
    temperature_schedule = []
    warmup_steps = 2
    plateau_steps = 4
    high_steps = BASIC_APPROACH_RETRY - (warmup_steps + plateau_steps)
    if high_steps < 0:
        warmup_steps = 1
        plateau_steps = BASIC_APPROACH_RETRY // 2
        high_steps = BASIC_APPROACH_RETRY - (warmup_steps + plateau_steps)
    mid_temperature = 0.6
    for i in range(warmup_steps):
        t = min_temperature + (mid_temperature - min_temperature) * (
            i / max(1, warmup_steps - 1)
        )
        temperature_schedule.append(round(t, 3))
    for _ in range(plateau_steps):
        t = mid_temperature + random.uniform(-0.09, 0.09)
        t = max(min_temperature, min(max_temperature, t))
        temperature_schedule.append(round(t, 3))
    for i in range(high_steps):
        if high_steps > 1:
            t = mid_temperature + (max_temperature - mid_temperature) * (
                i / (high_steps - 1)
            )
        else:
            t = max_temperature
        temperature_schedule.append(round(t, 3))
    for attempt, temperature in enumerate(temperature_schedule):
        os.system("git reset --hard")
        initial_solution, test_cases = basic_approach(
            initial_structure, problem_statement, temperature=temperature
        )
        print(f"Initial solution in single_process_create_task: {initial_solution}")
        if initial_solution is not None:
            return (initial_solution, test_cases)
        # Adaptive sleep: sleep more for higher temperature to allow model to "think"
        sleep_time = 1 + 0.5 * (temperature - min_temperature)
        time.sleep(sleep_time)
    print(f"Initial solution in single_process_create_task: None")
    return (None, None)
# ============================  Fix Task Functions   ======================================
def process_fix_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id
    # setting environment to include current working directory and lib directory
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()
    try:
        patch_text = fix_task_solve_workflow(
            problem_text, timeout=timeout - 60, run_id_1=run_id, enhancement=enhancement, should_review=True
        )
        os.system("git reset --hard")
    except Exception as e:
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logs.append(error_info)
    finally:
        os.chdir(cwd)
    return patch_text
def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
) -> tuple[str, List[str], List[str]]:
    global run_id, _current_tool_manager
    run_id = run_id_1
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_function_body",
            "get_file_content",
            "get_file_skeleton",
            "search_in_all_files_content",
            "apply_code_edit",
            "run_code",
            "finish",
        ],
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
    )
    _current_tool_manager = tool_manager
    logger.info(f"Starting main agent execution... Enhancement: {enhancement}")
    logger.info(f"Available tools: {tool_manager.available_tools}")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT,
    )
    enhanced_problem = problem_statement
    if enhancement:
        enhanced_problem = (
            problem_statement
            + "\n\n---\n\n# Enhanced Problem Analysis\n\n"
            + enhancement
        )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
        problem_statement=enhanced_problem
    )
    return execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        [GLM_MODEL_NAME, KIMI_MODEL_NAME],
        log_prefix="FIX_MAIN_AGENT",
    )
def _generate_metacognitive_prompt(
    cot, step: int, interval: int, baseline_test_status, last_metacog_step: int
) -> str:
    """
    Generate a meta-cognitive reflection prompt that forces the agent to evaluate its progress.
    This makes the agent explicitly think about:
    1. Are we closer to the solution than N steps ago?
    2. What assumptions might be wrong?
    3. Should we change strategies?
    COMPLETENESS CHECK:
    1. Have you searched for ALL occurrences of the pattern you modified?
    2. Have you tested ALL operations, not just construction?
    3. Did you verify the fix works in ALL code paths?
    4. List any remaining instances that might need fixing.
    """
    import textwrap
    recent_steps = (
        cot.thoughts[last_metacog_step:] if last_metacog_step > 0 else cot.thoughts
    )
    if not recent_steps:
        return None
    tool_usage = {}
    error_count = 0
    for thought in recent_steps:
        if isinstance(thought.next_tool_name, list):
            for tool in thought.next_tool_name:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        else:
            tool_usage[thought.next_tool_name] = (
                tool_usage.get(thought.next_tool_name, 0) + 1
            )
        if thought.is_error:
            error_count += 1
    if len(tool_usage) == 1:
        pattern_warning = f"\n⚠️  WARNING: You've only used '{list(tool_usage.keys())[0]}' tool in the last {interval} steps. This may indicate stuck behavior."
    elif len(tool_usage) == 2 and sum(tool_usage.values()) > interval * 0.8:
        pattern_warning = f"\n⚠️  WARNING: You've mostly alternated between '{list(tool_usage.keys())[0]}' and '{list(tool_usage.keys())[1]}'. This may indicate circular behavior."
    else:
        pattern_warning = ""
    test_progress_note = ""
    if baseline_test_status == "FAILED":
        test_progress_note = "\n⚠️  IMPORTANT: Tests are still failing. You have not made measurable progress on test success."
    elif baseline_test_status == "PASSED":
        test_progress_note = f"\n✓  PROGRESS: Tests passed at step {step}. Good work!"
    prompt = textwrap.dedent(
        f"""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                    🧠 META-COGNITIVE CHECKPOINT (Step {step})                    ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    You have now completed {interval} steps since the last reflection (steps {last_metacog_step} to {step}).
    📊 ACTIVITY SUMMARY:
    • Tools used: {', '.join(f'{k}({v})' for k, v in tool_usage.items())}
    • Errors encountered: {error_count}/{len(recent_steps)} steps{pattern_warning}{test_progress_note}
    🔍 MANDATORY REFLECTION - Answer these questions in your next_thought:
    1. PROGRESS CHECK:
       → Am I measurably closer to solving this problem than {interval} steps ago?
       → What concrete evidence do I have of progress? (e.g., "tests now pass", "found root cause", "reproduced bug")
       → If NO clear progress: what assumption was WRONG?
    2. STRATEGY EVALUATION:
       → Is my current approach working, or am I stuck in a loop?
       → Have I been doing the same type of actions repeatedly without new insights?
       → Should I try a completely different strategy?
    3. NEXT DECISION:
       → What is the ONE most important thing to do next?
       → Why is this more important than other options?
       → What will I learn from this action that I don't already know?
    ⚡ CRITICAL: If you are NOT making progress, you MUST change your approach.
    Consider: reading different files, using different tools, testing different hypotheses,
    or using rollback_to_checkpoint_tool to return to a known good state.
    Your next next_thought should briefly address these reflection points before taking action.
    """
    ).strip()
    return prompt
# ============================  Common Functions   ======================================
def execute_agent_workflow(
    cot: EnhancedCOT,
    tool_manager: EnhancedToolManager,
    system_prompt: str,
    instance_prompt: str,
    n_max_steps: int,
    timeout: int,
    models: List[str],
    log_prefix: str = "AGENT",
    finish_tool_name="finish",
    reject_observation_token_threshold: int = 50000,
    save_observation_to_file_token_threshold: int = 4000,
) -> str:
    global run_id
    logger.info(f"{log_prefix} Starting agent execution... ")
    start_time = time.time()
    logs: List[str] = []
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought = None
    next_tool_name = None
    next_tool_args = None
    logs.append(f"cwd: {os.getcwd()}")
    modified_files = set()
    files_with_syntax_errors = set()
    metacog_checkpoint_interval = 15  # Reflect every 15 steps
    last_metacog_step = 0
    baseline_test_status = None  # Track if we're making progress on tests
    current_model_index = 0
    for step in range(n_max_steps):
        selected_model = models[current_model_index]
        elapsed_time = time.time() - start_time
        logger.info("=" * 40 + f"[{log_prefix}] Step {step}" + "=" * 40)
        cost_usage = EnhancedNetwork.get_cost_usage()
        logger.info(
            f"[{log_prefix}] Elapsed time: {elapsed_time}, Usage: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
        )
        if cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0):
            logger.warning(
                f"[{log_prefix}] Usage exceeded limit: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
            )
            break
        if time.time() - start_time > timeout:
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        # META-COGNITIVE CHECKPOINT: Periodic self-reflection
        if step > 0 and (step - last_metacog_step) >= metacog_checkpoint_interval:
            metacog_prompt = _generate_metacognitive_prompt(
                cot=cot,
                step=step,
                interval=metacog_checkpoint_interval,
                baseline_test_status=baseline_test_status,
                last_metacog_step=last_metacog_step,
            )
            if metacog_prompt:
                messages.append({"role": "system", "content": metacog_prompt})
                last_metacog_step = step
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        # Adaptive temperature and model selection strategy
        temperature = 0.0
        if cot.is_thought_repeated():
            logger.info(
                f"[ADAPTIVE_STRATEGY] Thought repeated {cot.repeated_thoughts} times"
            )
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
            # Progressive temperature scaling
            if cot.repeated_thoughts == 1:
                temperature = 0.2
            elif cot.repeated_thoughts == 2:
                temperature = 0.4
            elif cot.repeated_thoughts == 3:
                temperature = 0.6
            else:
                temperature = min(0.7 + (cot.repeated_thoughts - 3) * 0.05, 0.9)
            # Model rotation after 2 repeated thoughts
            if cot.repeated_thoughts >= 2:
                # Use deterministic rotation instead of random
                model_idx = (cot.repeated_thoughts - 2) % len(models)
                selected_model = models[model_idx]
        # Also adapt if agent is stuck after many steps
        elif step > 200 and step % 50 == 0:
            temperature = 0.3
        try:
            inference_start_time = time.time()
            models_to_try = [selected_model] + [
                m for m in models if m != selected_model
            ]
            (
                next_thought,
                next_tool_name,
                next_tool_args,
                raw_text,
                total_attempts,
                error_counter,
                messages,
                used_model,
            ) = EnhancedNetwork.inference(
                messages, model=models_to_try, run_id=run_id, temperature=temperature
            )
            selected_model = used_model
            inference_duration = time.time() - inference_start_time
        except Exception as e:
            inference_duration = 0
            pass
        # Handle both single and multiple tool calls
        tool_names_list = (
            next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        )
        tool_args_list = (
            next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]
        )
        
        logger.info(f"[{log_prefix}] Used model: {selected_model}, Inference time: {inference_duration:.2f}s")
        logger.info(f"[{log_prefix}] Next thought: {next_thought}\n\n")
        logger.info(f"[{log_prefix}] About to execute {len(tool_names_list)} tool call(s): {tool_names_list}\n")
        logger.info(f"[{log_prefix}] Tool arguments: {json.dumps(tool_args_list, indent=4)}\n\n")
        # Update tool manager with current step and COT snapshot for checkpoint creation
        tool_manager._current_step = step
        tool_manager._cot_snapshot_cache = [
            {
                "thought": t.next_thought,
                "tool": t.next_tool_name,
                "args": str(t.next_tool_args)[:200],
                "success": not t.is_error,
            }
            for t in cot.thoughts[-10:]
        ]
        all_observations = []
        all_successful = True
        for idx, (tool_name, tool_args) in enumerate(
            zip(tool_names_list, tool_args_list)
        ):
            try:
                if '"' in tool_name or "'" in tool_name:
                    tool_name = tool_name.replace('"', "").replace("'", "")
                observation = (
                    tool_manager.get_tool(tool_name)(**tool_args)
                    if tool_args
                    else tool_manager.get_tool(tool_name)()
                )
                # Track file modifications for multi-file change awareness
                if (
                    tool_name == "apply_code_edit"
                    and tool_args
                    and "file_path" in tool_args
                ):
                    file_path = tool_args["file_path"]
                    if "ok, code edit applied successfully" in str(observation).lower():
                        modified_files.add(file_path)
                    elif "syntax error" in str(observation).lower():
                        files_with_syntax_errors.add(file_path)
                estimated_tokens = Utils.count_tokens(str(observation))
                if estimated_tokens > reject_observation_token_threshold:
                    observation = f"Error: Tool output from '{tool_name}' exceeded token limit ({estimated_tokens} tokens > 50000 tokens limit). The response is too large to process. Please use more specific queries, target smaller file ranges, or break the request into smaller operations."
                elif estimated_tokens > save_observation_to_file_token_threshold:
                    # Save the large observation to a file
                    observation_path = tool_manager._save_large_observation(
                        str(observation), tool_name
                    )
                    observation = f"Tool output from `{tool_name}` exceeded token limit ({estimated_tokens} tokens > 4000 tokens limit). The full output has been saved to: {observation_path}. You can read this file using the get_file_content tool if needed."
                all_observations.append(observation)
            except EnhancedToolManager.Error as e:
                error_msg = f"Tool {idx+1} ({tool_name}) error: {e.message}"
                all_observations.append(error_msg)
                all_successful = False
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                error_msg = (
                    f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                )
                all_observations.append(error_msg)
                all_successful = False
        # Combine observations
        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                [
                    f"Tool {i+1} ({tool_names_list[i]}):\n{obs}"
                    for i, obs in enumerate(all_observations)
                ]
            )
        
        logger.info(f"[{log_prefix}] Combined observation: {combined_observation}\n\n")
        # Store in COT (errors already handled in the loop above)
        cot.add_action(
            EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,  # Keep original format (list or single)
                next_tool_args=next_tool_args,
                observation=combined_observation,
                is_error=not all_successful,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages,
            )
        )
        # Check if finish was called in any of the tool calls
        if finish_tool_name in tool_names_list:
            if finish_tool_name == "finish_find_files_to_fix":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs
            elif finish_tool_name == "finish":
                # INSERT_YOUR_CODE
                # Find and return the observation for the finish_tool_name
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        if obs != "finish":
                            break
                        return tool_manager.get_final_git_patch()
    return tool_manager.get_final_git_patch()
def enhance_problem_statement(problem_statement: str) -> str:
    ENHANCEMENT_PROMPT = textwrap.dedent(
        """
        You are an expert at analyzing problem statements and extracting key information.
        Analyze the given problem statement and extract the following structured information:
        1. **Problem Summary** (1-2 sentences): What needs to be fixed or implemented?
        2. **Current Behavior**: What is happening now? (Include error messages, unexpected outputs, etc.)
        3. **Expected Behavior**: What should happen instead?
        4. **Reproduction Steps** (if applicable): Clear steps to reproduce the issue
        5. **Success Criteria**: How will we know the problem is solved?
            - What tests should pass?
            - What behavior should change?
            - What outputs should be different?
        6. **Key Requirements**:
            - Must-have functionality
            - Constraints to respect (backwards compatibility, performance, etc.)
            - Files/functions likely involved
        7. **Important Notes**:
            - Edge cases to consider
            - Potential pitfalls
            - Related functionality that might be affected
        If any section is not applicable or cannot be determined from the problem statement, write "Not specified" for that section.
        Format your response as markdown with clear section headers.
        Be concise but complete. Extract information that's present, don't invent details.
        """
    )
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 5:
        try:
            messages = [
                {"role": "system", "content": ENHANCEMENT_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n\n{problem_statement}",
                },
            ]
            enhanced, _ = EnhancedNetwork.make_request(
                messages, model=selected_model, temperature=0.0
            )
            return enhanced
        except Exception as e:
            retry += 1
            other_models = [model for model in AGENT_MODELS if model != selected_model]
            selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return ""
def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
    if Path(
        os.getcwd() + "/lib"
    ).exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", work_dir]
            )
            subprocess.run(
                ["git", "config", "--global", "user.email", "agent@sandbox.local"],
                check=True,
            )
            subprocess.run(
                ["git", "config", "--global", "user.name", "sandbox_agent"], check=True
            )
            subprocess.run(["git", "add", "."], check=True)
            result = subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", work_dir]
            )
    except Exception as e:
        logger.error(f"ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)
def get_problem_type(problem_statement: str, enhancement: str) -> str:
    retry = 0
    PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
        """
        You are a helpful Problem Classifier to find a Task Name from PROJECT DESCRIPTION and project structure.
        Classify development tasks as either:
        - FIX: If the PROJECT DESCRIPTION is about fixing a bug, creating a new functionality or improving the existing codebase.
        - CREATE: If the PROJECT DESCRIPTION is about creating a new functionality from scratch.
        Output ONLY: "CREATE" or "FIX"
        """
    )
    selected_model = QWEN_MODEL_NAME
    while retry < 5:
        try:
            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"{problem_statement}\n# Enhanced Problem: \n{enhancement}",
                },
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model)
            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                return response
        except Exception as e:
            retry += 1
            other_models = [model for model in AGENT_MODELS if model != selected_model]
            selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return PROBLEM_TYPE_FIX
def check_problem_type(problem_statement):  # type: ignore
    type_count = {PROBLEM_TYPE_CREATE: 0, PROBLEM_TYPE_FIX: 0}
    enhancement = enhance_problem_statement(problem_statement)
    for _ in range(3):
        problem_type = get_problem_type(problem_statement, enhancement)
        type_count[problem_type] += 1
    if type_count[PROBLEM_TYPE_CREATE] > type_count[PROBLEM_TYPE_FIX]:
        return PROBLEM_TYPE_CREATE, enhancement
    elif type_count[PROBLEM_TYPE_FIX] > type_count[PROBLEM_TYPE_CREATE]:
        return PROBLEM_TYPE_FIX, enhancement
    return PROBLEM_TYPE_FIX, enhancement
def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, run_id, agent_start_time
    agent_start_time = time.time()
    run_id = os.getenv("EVALUATION_RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    set_env_for_agent()
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 100
    result = None
    exception_occurred = None
    task_completed = threading.Event()
    def run_task():
        nonlocal result, exception_occurred
        try:
            global _current_tool_manager
            
            _current_tool_manager = EnhancedToolManager()
            problem_type, enhancement = check_problem_type(
                input_dict.get("problem_statement")
            )
            if problem_type == PROBLEM_TYPE_FIX:
                result = process_fix_task(input_dict, enhancement)
            else:
                result = process_create_task(input_dict, enhancement)
            print(f"Final Patch: {result}")
        except Exception as e:
            exception_occurred = e
            logger.error(f"Error in agent_main: {e}")
            try:
                time.sleep(1)
                result = process_fix_task(input_dict, enhancement)
            except Exception as e2:
                exception_occurred = e2
        finally:
            task_completed.set()
    
    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()
    task_thread.join(timeout=timeout)
    
    timed_out = task_thread.is_alive()
    if timed_out:
        logger.warning(f"Task execution timed out after {timeout} seconds, killing thread")
    
    global _current_tool_manager
    if _current_tool_manager is not None:
        try:
            final_patch = _current_tool_manager.get_final_git_patch()
            if final_patch:
                result = final_patch
        except Exception as e:
            logger.warning(f"Failed to get final patch from tool manager: {e}")
        finally:
            _current_tool_manager = None
    
    try:
        subprocess.Popen(["git", "reset", "--hard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    return result if result else ""