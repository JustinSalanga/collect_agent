from __future__ import annotations
import json
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import difflib
import tempfile
import threading
run_id = None
agent_start_time = None
_current_tool_manager = None
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = 1500
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
AGENT_MODELS = [model for model in [GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, GLM_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]
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
    tool_args: {"content": "test_content", "file_path": "file.js", "run_command": ["node", "file.js"]}
✅ **Good - Batch Multiple Searches**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: search_in_all_files_content
    tool_args: {"grep_search_command": "grep -r 'problematic_func' ."}
tool_call_2:
    tool_name: search_in_all_files_content
    tool_args: {"grep_search_command": "grep -r 'function problematic_func' ."}
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
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
Role: You are a senior bug-fix engineer working on an open-source repository.
You will be tasked to fix an issue from this repository.
Your thinking should be thorough and so it's fine if it's very long. You should think step by step before and after each action you decide to take.
You already have everything you need to solve this problem in the repository, even without internet connection.
Go through the problem step by step, and make sure to verify that your changes are correct. NEVER GIVE UP without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.
THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.
                                         
Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.                   
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
# Workflow
## High-Level Problem Solving Strategy
1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. Debug as needed. Use debugging techniques to isolate and resolve issues.
6. Test frequently. Run tests after each change to verify correctness.
7. Iterate until the root cause is fixed and all tests pass.
8. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.
Refer to the detailed sections below for more information on each step.
                                         
## 1. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.
## 2. Codebase Investigation
- Explore relevant files and directories.
- Search for key functions, classes, or variables related to the issue.
- Read and understand relevant code snippets.
- Identify the root cause of the problem.
- Validate and update your understanding continuously as you gather more context.
                                         
## 3. Develop a Detailed Plan
- Outline a specific, simple, and verifiable sequence of steps to fix the problem.
- Break down the fix into small, incremental changes.
## 4. Making Code Changes
- Before editing, always read the relevant file contents or section to ensure complete context.
- If a patch is not applied correctly, attempt to reapply it.
- Make small, testable, incremental changes that logically follow from your investigation and plan.
## 5. Debugging
- Make code changes only if you have high confidence they can solve the problem
- When debugging, try to determine the root cause rather than addressing symptoms
- Debug for as long as needed to identify the root cause and identify a fix
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening
- To test hypotheses, you can also add test statements or functions
- Revisit your assumptions if unexpected behavior occurs.
## 6. Testing
- Run tests frequently using the available testing tools (for example, by calling the `run_code` tool).
- After each change, verify correctness by running relevant tests via the testing tool rather than invoking shell commands directly.
- If tests fail, analyze failures and revise your patch.
- Write additional tests if needed to capture important behaviors or edge cases.
- Ensure all tests pass before finalizing.
## 7. Final Verification
- Confirm the root cause is fixed.
- Review your solution for logic correctness and robustness.
                                         
## 8. Final Reflection and Additional Testing
- Reflect carefully on the original intent of the user and the problem statement.
- Think about potential edge cases or scenarios that may not be covered by existing tests.
- Write additional tests that would need to pass to fully validate the correctness of your solution.
- Run these new tests and ensure they all pass.
- Be aware that there are additional hidden tests that must also pass for the solution to be successful.
- Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.
# Tool Documentation
You have access to the following tools:-
{tools_docs}
# Tool Usage Guidelines
- Use appropriate tools to gather context before making changes.
- **Choose the right tool for the job**:
  - Use `get_file_content` for reading arbitrary file sections, entire files, or specific code areas.
- If required parameters are missing, infer them from the problem statement and code.
- Use exact values provided by the user (especially in quotes).
- Don't make up values for or ask about optional parameters.
- Use `search_in_all_files_content` to find all occurrences of an issue before fixing.
# Meta-Cognitive Checkpoints
Every 15 steps, you will receive a META-COGNITIVE CHECKPOINT that analyzes your recent activity and progress:
- **Progress Analysis**: Shows what tools you've used and whether you're making measurable progress
- **Pattern Detection**: Alerts you if you're stuck in repetitive behavior (e.g., using same tools repeatedly)
- **Mandatory Reflection**: You MUST address these reflection questions in your next_thought:
  1. Am I measurably closer to solving this problem than 15 steps ago?
  2. Is my current approach working, or am I stuck in a loop?
  3. What is the ONE most important thing to do next?
**How to respond to meta-cognitive prompts:**
- Honestly evaluate your progress with concrete evidence (not assumptions)
- If you haven't made progress, identify which assumption was WRONG
- If stuck in a pattern, CHANGE your approach (different files, different strategy)
- Be specific about what you'll learn from your next action that you don't already know
**Critical**: These checkpoints exist to prevent wasted effort. Take them seriously and be willing to pivot when not making progress.
# Cognitive Tools for Knowledge Persistence
You have access to powerful cognitive tools designed to preserve knowledge and prevent retry loops:
## Hypothesis Tracking
**Purpose**: Track theories about the bug to avoid retesting rejected hypotheses.
**Tools**:
- **create_hypothesis(description, evidence)**: Log a theory when you form one
  - Use when: You have a theory but need to investigate further
  - Example: "Function fails on edge case" based on "test_edge_case fails with unexpected error"
- **test_hypothesis(hypothesis_id, outcome, findings)**: Record test results
  - Use when: You've tested a hypothesis through code changes or investigation
  - Outcomes: "confirmed", "rejected", "inconclusive"
  - Example: After implementing a fix, mark hypothesis #1 as "confirmed" with "fix resolves test failure"
- **list_hypotheses()**: Review all hypotheses and their status
  - Use when: During meta-cognitive checkpoints, or when stuck
  - Shows: Which theories confirmed/rejected/untested
## Strategy Memory
**Purpose**: Remember what approaches you've tried.
**Tools**:
- **log_strategy(approach, reasoning)**: Record planned approach BEFORE implementing
  - Use when: About to make significant code changes
  - Example: "Update function in <file> at line <N>" because "<reference relevant hypothesis for reasoning>"
- **mark_strategy_outcome(strategy_id, success, reason)**: Record whether it worked
  - Use when: After testing the strategy (tests pass/fail)
  - Example: Mark strategy #1 as failed: "Tests passed but broke edge case in rare input scenario"
- **list_attempted_strategies()**: Review all strategies and outcomes
  - Use when: During reflection, or when choosing next approach
  - Shows: Which strategies succeeded/failed/pending
## Large Observation Analysis
**Purpose**: Extract actionable insights from observations too large to fit in context (saved to .observation/ directory).
**Tool**:
- **analyze_saved_observation(query_type)**: Parse most recent saved observation
  - Use when: After test runs with large output, or when observation was saved to file
  - Query types:
    - "errors": Extract error messages and tracebacks
    - "failed_tests": List which specific tests failed
    - "warnings": Extract warnings
    - "summary": Get statistics and first/last lines
**When to Use These Tools**:
1. **Early Investigation** (Steps 1-10):
   - Use `create_hypothesis` when you identify potential root causes
   - Use `analyze_saved_observation` if test output was too large
2. **Before Making Changes** (Before edits):
   - Use `log_strategy` to record your planned approach
   - Reference hypothesis IDs in your reasoning
3. **After Testing** (After running tests):
   - Use `test_hypothesis` to mark hypotheses as confirmed/rejected
   - Use `mark_strategy_outcome` to record whether strategy worked
   - Use `analyze_saved_observation` if test output was saved to file
4. **During Meta-Cognitive Checkpoints** (Every 15 steps):
   - Use `list_hypotheses` to see what you've learned
   - Use `list_attempted_strategies` to avoid retrying failed approaches
**Critical**: These tools create institutional memory. Use them consistently to avoid wasting effort.
# Critical Requirements
- Fix must be backward compatible unless stated otherwise.
- Ensure changes are exhaustive and don't break other functionality.
- Don't edit test files directly - use the dedicated test generation tool when needed.
- Don't create new files unless absolutely necessary.
- Check both expected output in the problem statement AND in relevant test cases.
# Step Efficiency
You have a limited step budget (target: 15 steps, maximum: 30 steps). Prioritize simpler, faster solutions and make forward progress with each step. Test frequently to catch issues early. Don't over-investigate - once you understand the issue, implement the fix.
Here is the problem statement:
{problem_statement}
# Response Format Requirements
{format_prompt}
"""
)
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
        result = subprocess.run(["bash", "-c", grep_search_command], capture_output=True, text=True)
        # Check for command execution errors (return codes other than 0 and 1)
        # Note: grep returns 1 when no matches are found, which is not an error
        if result.returncode > 1:
            error_msg = result.stderr.strip() if result.stderr.strip() else "Unknown error"
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
        def extract_matches(file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
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
            match_lines = [idx + 1 for idx, line in enumerate(source_lines) if escaped_search_term in line]
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
        ) = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
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
        timeout: int = 150,
        tool_mode: str = "none",
        tool_docs: list = [],
    ) -> str:
        global run_id, DEFAULT_TIMEOUT, agent_start_time
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 200
        # Check if agent execution time exceeds DEFAULT_TIMEOUT seconds
        if agent_start_time is not None:
            elapsed_time = time.time() - agent_start_time
            if elapsed_time >= timeout:
                raise RuntimeError(f"HTTP ERROR: Agent execution timeout after {elapsed_time:.2f} seconds (limit: {DEFAULT_TIMEOUT} seconds)")
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
                response = requests.post(url, json=request_data, timeout=(30, timeout), headers=headers)  # (connect timeout, read timeout)
                response.raise_for_status()
                try:
                    response_json = response.json()
                except JSONDecodeError as e:
                    # If final attempt, surface as error; otherwise retry
                    if i >= attempts - 1:
                        elapsed = time.time() - start_time
                        raise ValueError(f"HTTP ERROR: Invalid JSON response for model {model} after {attempts} attempts: {e}")
                    continue
                # Support both OpenAI-style and raw text responses
                try:
                    raw_text = response_json["content"]
                    tool_calls = response_json["tool_calls"]
                except Exception as e:
                    raise RuntimeError(f"HTTP ERROR: Response Parse Error timeout for model {model} after {attempts} attempts")
                if (tool_mode == "none" and (raw_text is None or raw_text == "")) or (
                    tool_mode != "none" and (tool_calls is None or len(tool_calls) == 0)
                ):
                    raise RuntimeError(f"HTTP ERROR: NO RESPONSE FOUND Tool model {model} after {attempts} attempts")
                elapsed = time.time() - start_time
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request timeout for model {model} after {attempts} attempts")
                time.sleep(1)
                continue
            except requests.exceptions.ConnectionError as e:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Connection error for model {model} after {attempts} attempts: {e}")
                time.sleep(1)
                continue
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                elapsed = time.time() - start_time
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model}"
                # Check for 504 Gateway Timeout specifically
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 504: Gateway Timeout for model {model} after {attempts} attempts: {e}")
                    time.sleep(1)
                    continue
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(1)
                continue
            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request failed for model {model} after {attempts} attempts: {e}")
                time.sleep(1)
                continue
        # Fallback (should not reach here due to raises above)
        raise RuntimeError(f"HTTP ERROR: Failed to get response for model {model} after {attempts} attempts")
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
                current_model = models[current_model_idx] if current_model_idx < len(models) else models[-1]
                used_model = current_model
                # index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text, _ = cls.make_request(messages, model=current_model, temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not (is_valid):
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                # Check if it's a 504 error - switch to next model
                is_504_error = "504" in error_body or "HTTP ERROR 504" in error_body or "Gateway Timeout" in error_body
                is_timeout_error = "Agent execution timeout" in error_body
                if is_timeout_error:
                    raise RuntimeError(error_body)
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
                        messages.append({"role": "user", "content": "observation: " + error_body})
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
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("'").strip('"').strip()
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
                    EnhancedToolManager.get_tool_args_for_tool(tool_name, required_only=True),
                    next_tool_args,
                )
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args
    @classmethod
    def is_valid_response(cls, raw_text: str) -> bool:
        if type(raw_text) is dict and raw_text.get("error", None) is not None and raw_text.get("error") != "":
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        stripped = raw_text.strip()
        has_next_thought = "next_thought" in raw_text.lower() or "<next_thought>" in raw_text.lower()
        has_next_tool_name = "next_tool_name" in raw_text.lower() or "<next_tool_name>" in raw_text.lower()
        has_next_tool_args = "next_tool_args" in raw_text.lower() or "<next_tool_args>" in raw_text.lower()
        # Valid endings: JSON format or XML-style tags
        valid_ending = stripped.endswith("}") or stripped.endswith("}]") or stripped.endswith("</next_tool_args>") or stripped.endswith(">")
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
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
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[0].strip()
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
        tool_call_matches = list(re.finditer(tool_call_pattern, text_resp, re.IGNORECASE))
        if tool_call_matches:
            # Multi-tool call format
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = tool_call_matches[i + 1].start() if i + 1 < len(tool_call_matches) else len(text_resp)
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                error_msg = "Multi-tool format detected but no valid tool calls extracted"
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
            if text_resp.find("next_thought:") < text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:"):
                next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
                next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
                try:
                    # Handle array format
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(next_tool_names, next_tool_args_raw)
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
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(1)
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
    def run_code(self, content: str, file_path: str, generated_test_files: list, run_command: list[str]) -> str:
        """
        Runs code by saving it to a file and executing it (supports multiple languages).
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in
            generated_test_files: list to track generated test files
            run_command: command to run the file
        Returns:
            Standard output from code execution or error message
        """
        python_extensions = (".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")
        if file_path.endswith(python_extensions):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
        self.file_ops.save(file_path, content)
        generated_test_files.append(file_path)
        try:
            logger.info(f"Running command in run_code: {run_command}")
            result = subprocess.run(run_command, capture_output=True, text=True, check=False, timeout=60)
        except ValueError as e:
            return f"Error: {e}"
        if result.returncode != 0:
            return f"Error running code: {result.stderr}"
        observation = f"{result.stdout}\n"
        return observation
    def _truncate_output(self, output: str, max_first_lines: int = 500, max_last_lines: int = 500) -> str:
        """Truncate long output to first N and last N lines with summary in middle."""
        lines = output.split("\n")
        total_lines = len(lines)
        if total_lines <= max_first_lines + max_last_lines:
            return output
        first_lines = lines[:max_first_lines]
        last_lines = lines[-max_last_lines:]
        omitted_lines = total_lines - max_first_lines - max_last_lines
        truncated = "\n".join(first_lines)
        truncated += f"\n\n... ({omitted_lines} lines omitted) ...\n\n"
        truncated += "\n".join(last_lines)
        return truncated
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
                    obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
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
            while retry < 3:
                try:
                    response, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
                    return response.strip()
                except Exception as e:
                    retry += 1
                    time.sleep(1)
        except Exception as e:
            return None
        return None
    def is_thought_repeated(self) -> bool:
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False
    def add_action(self, action: EnhancedCOT.Action) -> bool:  # don't add if thought is repeated
        self.thoughts.append(action)
        # Check if we need to summarize older messages
        # Only check when we have enough messages to potentially summarize
        total_thoughts = len(self.thoughts)
        if total_thoughts >= self.latest_observations_to_keep + self.summarize_batch_size:
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
                assistant_str = (
                    f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                )
                # Render list observations as JSON array for the model
                if isinstance(thought.observation, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
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
                    assistant_str = (
                        f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    )
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
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
                        user_str = f"observation: error ocurred. detailed output omitted " f"({_obs_len}) lines\n"
                    else:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        )
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
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
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False
class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        """
        Limit the number of strings to 1000
        """
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        import re
        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
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
    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        """
        List the directory structure (files and folders) from the specified directory up to a maximum depth. Use this to explore the repository structure, understand file organization, locate relevant directories, or get an overview of the codebase layout. Output is automatically truncated if it exceeds token limits by reducing depth. Returns a tree-like structure showing the directory hierarchy.
        Arguments:
            directory_path: Directory path to list (can be relative or absolute). Must exist and be a directory.
            max_depth: Maximum depth of recursion (default 0, meaning only the specified directory). Use 1 for immediate subdirectories, 2 for two levels deep, etc.
        Output:
            Tree-structured directory listing showing files and folders with visual tree characters (├──, └──, │). Includes the directory path and depth information. If the requested depth exceeds token limits, automatically reduces depth and includes a note. Returns error message if directory doesn't exist or is not a directory.
        """
        # Validate directory_path exists and is a directory
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
        ignore = {".git", "__pycache__", ".pytest_cache", "node_modules", ".tox", ".venv", "venv", ".eggs"}
        def tree(path: str, prefix: str = "", depth: int = 0, current_max_depth: int = 0) -> list[str]:
            if depth > current_max_depth:
                return []
            try:
                items = sorted(os.listdir(path))
            except (PermissionError, OSError) as e:
                return [f"{prefix}[Error reading directory: {str(e)}]"]
            # separate dirs and files
            dirs = [
                i for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith(".") and i not in ignore and not i.endswith(".egg-info")
            ]
            files = [i for i in items if os.path.isfile(os.path.join(path, i)) and not i.startswith(".")]
            lines: list[str] = []
            # add directories
            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1, current_max_depth))
            # add files
            for idx, f in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{f}")
            return lines
        def count_tokens(text: str) -> int:
            """Count tokens in text, with fallback to character-based estimation."""
            try:
                if "Utils" in globals() and hasattr(Utils, "count_tokens"):
                    return Utils.count_tokens(text)
            except (NameError, AttributeError):
                pass
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
        MAX_TOKENS = 3000
        current_depth = max_depth
        # Try to find a depth that results in less than MAX_TOKENS
        while current_depth >= 0:
            entries = tree(directory_path, "", 0, current_depth)
            result = f"Directory structure (depth={current_depth}):\n{directory_path}/\n" + "\n".join(entries)
            token_count = count_tokens(result)
            if token_count <= MAX_TOKENS:
                if current_depth < max_depth:
                    result += (
                        f"\n\n[Note: Requested depth {max_depth} exceeded token limit. Showing depth {current_depth} instead ({token_count} tokens).]"
                    )
                return result
            # If we're at depth 0 and still too large, return it anyway with a warning
            if current_depth == 0:
                result += f"\n\n[Warning: Result exceeds token limit ({token_count} tokens > {MAX_TOKENS} tokens). Consider using a more specific directory_path.]"
                return result
            current_depth -= 1
        # Fallback (shouldn't reach here, but just in case)
        entries = tree(directory_path, "", 0, 0)
        result = f"Directory structure (depth=0):\n{directory_path}/\n" + "\n".join(entries)
        return result
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
        return "\n\n".join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])
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
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
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
            self.tool_invocations[fn.__name__] = self.tool_invocations.get(fn.__name__, 0) + 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                # Initialize tool_failure entry if not present
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {j: 0 for j in self.Error.ErrorType.__members__}
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
    def get_tool_args_for_tool(cls, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in cls.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(cls.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return cls.TOOL_LIST[tool_name]["input_schema"]["required"]
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
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
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
        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
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
                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match
            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            # Build the context with line numbers
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            # Extract original description
            description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            return f"{description}\n" + "\n".join(context_lines_list)
        def find_most_similar_content(original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
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
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
            # Calculate similarity scores
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:  # Only include reasonably similar content
                    similarities.append((ratio, chunk_desc, chunk_content))
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]
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
                        content_with_context = add_context_to_similar_match(original, content, context_lines=2)
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
                        start_line = max(0, replace_line_start - 20)
                        end_line = min(len(lines), replace_line_start + 20)
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
        # Initialize root cause analysis tracking
        self.root_cause_counter = 0
        self.root_causes = []  # Track multiple root causes with file associations
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
                    if available_tools is not None and name not in available_tools:  # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
    def _get_current_step(self) -> int:
        """Helper to get current step count."""
        return len(self.tool_invocations)
    @EnhancedToolManager.tool
    def think(self, thought: str) -> str:
        """ "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective. Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests.
        Arguments:
            thought: Your thoughts.
        Output:
            Confirmation that the thought has been logged.
        """
        return "ok"
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
        return self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)
    @EnhancedToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            directory_path: the directory path to list (default: ".")
            max_depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(directory_path=directory_path, max_depth=max_depth)
    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str, run_command: List[str]) -> str:
        """
        Runs any code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
            run_command: command to run the file (i.e., ["python", "file.py"] or ["node", "file.js"] etc)
        """
        return self.test_manager.run_code(
            content=content,
            file_path=file_path,
            generated_test_files=self.generated_test_files,
            run_command=run_command,
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
            # Exclude all files in .checkpoints directory
            checkpoints_dir = getattr(self, "checkpoint_dir", ".checkpoints")
            if os.path.exists(checkpoints_dir):
                try:
                    for root, dirs, files in os.walk(checkpoints_dir):
                        for file in files:
                            file_path = os.path.relpath(os.path.join(root, file))
                            exclude.add(file_path)
                except Exception:
                    pass
            logger.debug(f"Excluding files from patch: {exclude}")
            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)
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
        """
        Signals completion of the current workflow execution
        Arguments:
            None
        """
        return "finish"
    @EnhancedToolManager.tool
    def create_hypothesis(self, description: str, evidence: str) -> str:
        """Log a hypothesis about the bug's root cause.
        Use this tool when you form a theory about what's causing the issue. This helps track
        which theories you've already considered.
        Arguments:
            description: What you believe is causing the issue
            evidence: What observations led to this hypothesis
        Output:
            Confirmation with hypothesis ID for later reference.
        """
        self.hypothesis_counter += 1
        hypothesis = {
            "id": self.hypothesis_counter,
            "description": description,
            "evidence": evidence,
            "status": "untested",
            "findings": None,
            "created_step": len(getattr(self, "tool_invocations", {})),
        }
        self.hypotheses.append(hypothesis)
        return f"Hypothesis #{self.hypothesis_counter} created: {description}\nStatus: untested\nEvidence: {evidence}"
    @EnhancedToolManager.tool
    def test_hypothesis(self, hypothesis_id: int, outcome: str, findings: str) -> str:
        """Record results of testing a hypothesis.
        After you test a hypothesis (by making code changes, running tests, or investigating),
        use this tool to record what you learned. This prevents retrying the same failed approach.
        Arguments:
            hypothesis_id: ID from create_hypothesis (e.g., 1, 2, 3)
            outcome: Result of testing - must be "confirmed", "rejected", or "inconclusive"
            findings: What you learned from testing this hypothesis (e.g., "Fix worked for simple cases but fails on nested structures")
        Output:
            Updated hypothesis status.
        """
        if outcome not in ["confirmed", "rejected", "inconclusive"]:
            return f"Error: outcome must be 'confirmed', 'rejected', or 'inconclusive', got '{outcome}'"
        for hyp in self.hypotheses:
            if hyp["id"] == hypothesis_id:
                hyp["status"] = outcome
                hyp["findings"] = findings
                hyp["tested_step"] = len(getattr(self, "tool_invocations", {}))
                return f"Hypothesis #{hypothesis_id} marked as {outcome}\nFindings: {findings}"
        return f"Error: Hypothesis #{hypothesis_id} not found"
    @EnhancedToolManager.tool
    def list_hypotheses(self) -> str:
        """View all hypotheses with their test status.
        Use this to review what theories you've already considered and tested. Especially useful:
        - When stuck (to avoid retrying rejected hypotheses)
        - During metacognitive reflection checkpoints
        Arguments:
            None
        Output:
            Formatted list of all hypotheses with status and findings.
        """
        if not self.hypotheses:
            return "No hypotheses recorded yet. Use create_hypothesis to log theories about the bug."
        output = ["=== HYPOTHESIS TRACKER ===\n"]
        untested = [h for h in self.hypotheses if h["status"] == "untested"]
        confirmed = [h for h in self.hypotheses if h["status"] == "confirmed"]
        rejected = [h for h in self.hypotheses if h["status"] == "rejected"]
        inconclusive = [h for h in self.hypotheses if h["status"] == "inconclusive"]
        output.append(f"Summary: {len(confirmed)} confirmed, {len(rejected)} rejected, {len(inconclusive)} inconclusive, {len(untested)} untested\n")
        for status, hypotheses in [
            ("CONFIRMED", confirmed),
            ("REJECTED", rejected),
            ("INCONCLUSIVE", inconclusive),
            ("UNTESTED", untested),
        ]:
            if hypotheses:
                output.append(f"\n{status}:")
                for h in hypotheses:
                    output.append(f"\n  [{h['id']}] {h['description']}")
                    output.append(f"      Evidence: {h['evidence']}")
                    if h["findings"]:
                        output.append(f"      Findings: {h['findings']}")
        return "\n".join(output)
    @EnhancedToolManager.tool
    def log_strategy(self, approach: str, reasoning: str) -> str:
        """Record a high-level strategy before attempting it.
        Use this BEFORE making significant code changes to log your planned approach.
        Arguments:
            approach: Brief description of the approach
            reasoning: Why you think this will work
        Output:
            Confirmation with strategy ID for later reference.
        """
        self.strategy_counter += 1
        strategy = {
            "id": self.strategy_counter,
            "approach": approach,
            "reasoning": reasoning,
            "success": None,
            "reason": None,
            "timestamp": time.time(),
            "created_step": len(getattr(self, "tool_invocations", {})),
        }
        self.strategies.append(strategy)
        return f"Strategy #{self.strategy_counter} logged: {approach}\nReasoning: {reasoning}\nUse mark_strategy_outcome to record results."
    @EnhancedToolManager.tool
    def mark_strategy_outcome(self, strategy_id: int, success: bool, reason: str) -> str:
        """Record whether a strategy worked.
        After attempting a strategy, record the outcome. This is crucial for institutional memory - you'll know what you already tried even after reverting changes.
        Arguments:
            strategy_id: ID from log_strategy (e.g., 1, 2, 3)
            success: True if approach worked (tests passed, bug fixed), False otherwise
            reason: Why it succeeded/failed (e.g., "Tests passed but introduced new bug in edge case")
        Output:
            Updated strategy status.
        """
        for strat in self.strategies:
            if strat["id"] == strategy_id:
                strat["success"] = success
                strat["reason"] = reason
                strat["completed_step"] = len(getattr(self, "tool_invocations", {}))
                status = "SUCCEEDED" if success else "FAILED"
                return f"Strategy #{strategy_id} marked as {status}\nReason: {reason}"
        return f"Error: Strategy #{strategy_id} not found"
    @EnhancedToolManager.tool
    def list_attempted_strategies(self) -> str:
        """View all strategies tried, with outcomes.
        Use this to review what approaches you've already attempted. Critical for:
        - Avoiding retry loops
        - Understanding what doesn't work
        - Building on partially successful strategies
        Arguments:
            None
        Output:
            Formatted list of all strategies with outcomes.
        """
        if not self.strategies:
            return "No strategies recorded yet. Use log_strategy before attempting significant changes."
        output = ["=== STRATEGY HISTORY ===\n"]
        succeeded = [s for s in self.strategies if s["success"] is True]
        failed = [s for s in self.strategies if s["success"] is False]
        pending = [s for s in self.strategies if s["success"] is None]
        output.append(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} pending\n")
        for status, strategies in [
            ("SUCCEEDED", succeeded),
            ("FAILED", failed),
            ("PENDING", pending),
        ]:
            if strategies:
                output.append(f"\n{status}:")
                for s in strategies:
                    output.append(f"\n  [{s['id']}] {s['approach']}")
                    output.append(f"      Reasoning: {s['reasoning']}")
                    if s["reason"]:
                        output.append(f"      Outcome: {s['reason']}")
        return "\n".join(output)
    @EnhancedToolManager.tool
    def analyze_saved_observation(self, query_type: str) -> str:
        """Extract key information from the most recent large observation file.
        When observations are too large (>4000 tokens), they get saved to .observation/ directory.
        This tool helps you extract actionable insights from those files without reading them manually.
        Arguments:
            query_type: What to extract. Options:
                - "errors": Extract all error messages and tracebacks
                - "failed_tests": List which specific tests failed
                - "warnings": Extract warning messages
                - "summary": Get a high-level summary of the output
        Output:
            Extracted information from the most recent saved observation.
        """
        # Find most recent observation file
        try:
            obs_files = sorted(
                [f for f in os.listdir(self.observation_dir) if f.startswith("observation_")],
                key=lambda x: os.path.getmtime(os.path.join(self.observation_dir, x)),
                reverse=True,
            )
            if not obs_files:
                return "No saved observations found in .observation/ directory"
            latest_file = os.path.join(self.observation_dir, obs_files[0])
            with open(latest_file, "r", encoding="utf-8") as f:
                content = f.read()
            output = [f"Analyzing: {obs_files[0]}\n"]
            if query_type == "errors":
                # Extract error messages and tracebacks
                errors = []
                lines = content.split("\n")
                in_traceback = False
                current_traceback = []
                for line in lines:
                    if "Error:" in line or "Exception:" in line or "Traceback" in line:
                        in_traceback = True
                        current_traceback = [line]
                    elif in_traceback:
                        if line.strip() and not line[0].isspace():
                            errors.append("\n".join(current_traceback))
                            in_traceback = False
                            current_traceback = []
                        else:
                            current_traceback.append(line)
                if current_traceback:
                    errors.append("\n".join(current_traceback))
                if errors:
                    output.append(f"Found {len(errors)} error(s):\n")
                    for i, error in enumerate(errors[:5], 1):  # Limit to first 5
                        output.append(f"\nError {i}:\n{error}\n")
                    if len(errors) > 5:
                        output.append(f"\n... and {len(errors) - 5} more errors")
                else:
                    output.append("No errors found in observation")
            elif query_type == "failed_tests":
                # Extract test failure information
                lines = content.split("\n")
                failed = []
                for line in lines:
                    if "FAILED" in line or "failed" in line.lower():
                        failed.append(line.strip())
                if failed:
                    output.append(f"Found {len(failed)} failed test reference(s):\n")
                    for line in failed[:20]:  # Limit to 20
                        output.append(f"  {line}")
                    if len(failed) > 20:
                        output.append(f"\n... and {len(failed) - 20} more")
                else:
                    output.append("No failed tests found in observation")
            elif query_type == "warnings":
                # Extract warnings
                lines = content.split("\n")
                warnings = [line.strip() for line in lines if "warning" in line.lower() or "warn" in line.lower()]
                if warnings:
                    output.append(f"Found {len(warnings)} warning(s):\n")
                    for warning in warnings[:10]:
                        output.append(f"  {warning}")
                    if len(warnings) > 10:
                        output.append(f"\n... and {len(warnings) - 10} more")
                else:
                    output.append("No warnings found in observation")
            elif query_type == "summary":
                # Provide summary statistics
                lines = content.split("\n")
                output.append(f"Total lines: {len(lines)}")
                output.append(f"Total characters: {len(content)}")
                error_count = sum(1 for line in lines if "error" in line.lower())
                warning_count = sum(1 for line in lines if "warning" in line.lower())
                passed_count = sum(1 for line in lines if "PASSED" in line or "passed" in line.lower())
                failed_count = sum(1 for line in lines if "FAILED" in line or "failed" in line.lower())
                output.append(f"\nKeyword counts:")
                output.append(f"  Errors: {error_count}")
                output.append(f"  Warnings: {warning_count}")
                output.append(f"  Passed: {passed_count}")
                output.append(f"  Failed: {failed_count}")
                # Show first and last few lines
                output.append(f"\nFirst 5 lines:")
                for line in lines[:5]:
                    output.append(f"  {line}")
                output.append(f"\nLast 5 lines:")
                for line in lines[-5:]:
                    output.append(f"  {line}")
            else:
                return f"Error: query_type must be 'errors', 'failed_tests', 'warnings', or 'summary'. Got '{query_type}'"
            return "\n".join(output)
        except Exception as e:
            return f"Error analyzing observation: {e}"
def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
) -> str:
    global run_id, _current_tool_manager
    run_id = run_id_1
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "search_in_all_files_content",
            "apply_code_edit",
            "run_code",
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "analyze_saved_observation",
            "think",
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
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
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
def llm_select_run_command_for_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    retry = 0
    while retry < 10:
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
                        "command": ["bbb", "aaa.js"]
                    }}
                    ```
                    """,
                }
            ]
            raw_text, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            json_result = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
            return json_result.get("command")
        except Exception as e:
            time.sleep(1)
            retry += 1
    return []
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
    while retry < 10:
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
            time.sleep(1)
    return False
def search_for_similar_problems(concepts: dict) -> str:
    """Search for similar problems and their test cases using LLM-based knowledge."""
    if not concepts.get("search_terms"):
        return ""
    SEARCH_EDGE_CASES_PROMPT = textwrap.dedent(
        """
        You are an expert at identifying edge cases for programming problems.
        Based on the problem domain and common patterns, generate a comprehensive list of edge cases that similar problems typically need to handle.
        
        Think comprehensively about edge cases including:
            - Invalid or malformed inputs that don't match expected patterns
            - Incomplete sequences or data that cannot be fully processed
            - Boundary conditions (empty inputs, maximum sizes)
            - Invalid combinations that should throw errors
            - Partial processing scenarios where valid portions should be processed before invalid ones
            - Error handling for unrecognized or out-of-specification inputs
        Return a JSON array of edge case descriptions, each as a string.
        """
    )
    selected_model = QWEN_MODEL_NAME
    for retry in range(10):
        try:
            domain = concepts.get("domain", "")
            search_terms_str = ", ".join(concepts.get("search_terms", []))
            messages = [
                {"role": "system", "content": SEARCH_EDGE_CASES_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem domain: {domain}\nRelated concepts: {search_terms_str}\n\nGenerate comprehensive edge cases that similar problems typically handle.",
                },
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.3)
            # Try to extract JSON array - handle nested arrays
            json_match = re.search(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", response, re.DOTALL)
            if json_match:
                try:
                    edge_cases = json.loads(json_match.group(0))
                    if isinstance(edge_cases, list) and edge_cases:
                        return "\n".join([f"- {case}" for case in edge_cases[:10]])  # Limit to 10
                except json.JSONDecodeError:
                    # Try to extract from code blocks
                    code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
                    if code_block_match:
                        try:
                            edge_cases = json.loads(code_block_match.group(1))
                            if isinstance(edge_cases, list) and edge_cases:
                                return "\n".join([f"- {case}" for case in edge_cases[:10]])
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            time.sleep(1)
    return ""
def extract_core_concepts_for_search(problem_statement: str) -> dict:
    """Extract core concepts"""
    EXTRACT_CONCEPTS_PROMPT = textwrap.dedent(
        """
        You are an expert at analyzing programming problems and extracting their core concepts.
        Your task is to identify the fundamental concepts and domain of the problem using the exact terminology from the problem statement.
        
        Rules:
        1. Identify the core domain related to the problem statement.
        2. Extract key algorithmic concepts related to the problem statement.
        3. Identify common edge cases that similar problems typically have
        4. Can use synonyms, related terms, or broader categories instead of exact words from the problem
        5. Focus on what types of inputs/outputs and validation might be needed
        
        Return a JSON object with:
        - "search_terms": array of 2-4 search terms
        - "domain": the problem domain in general terms
        - "common_edge_cases": array of typical edge cases for this type of problem
        
        """
    )
    retry = 0
    selected_model = GLM_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": EXTRACT_CONCEPTS_PROMPT},
                {"role": "user", "content": f"Problem Statement:\n{problem_statement}"},
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            # Try to parse JSON from response - handle nested objects
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
            if json_match:
                try:
                    concepts = json.loads(json_match.group(0))
                    if isinstance(concepts, dict) and "search_terms" in concepts:
                        return concepts
                except json.JSONDecodeError:
                    # Try to extract from code blocks
                    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
                    if code_block_match:
                        try:
                            concepts = json.loads(code_block_match.group(1))
                            if isinstance(concepts, dict) and "search_terms" in concepts:
                                return concepts
                        except json.JSONDecodeError:
                            pass
            return {"search_terms": [], "domain": "", "common_edge_cases": []}
        except Exception as e:
            retry += 1
            if retry > 2:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(1)
    return {"search_terms": [], "domain": "", "common_edge_cases": []}
def _generate_metacognitive_prompt(cot, step: int, interval: int, baseline_test_status, last_metacog_step: int) -> str:
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
    recent_steps = cot.thoughts[last_metacog_step:] if last_metacog_step > 0 else cot.thoughts
    if not recent_steps:
        return None
    tool_usage = {}
    error_count = 0
    for thought in recent_steps:
        if isinstance(thought.next_tool_name, list):
            for tool in thought.next_tool_name:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        else:
            tool_usage[thought.next_tool_name] = tool_usage.get(thought.next_tool_name, 0) + 1
        if thought.is_error:
            error_count += 1
    if len(tool_usage) == 1:
        pattern_warning = (
            f"\n⚠️  WARNING: You've only used '{list(tool_usage.keys())[0]}' tool in the last {interval} steps. This may indicate stuck behavior."
        )
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
    Consider: reading different files, using different tools, testing different hypotheses.
    Your next next_thought should briefly address these reflection points before taking action.
    """
    ).strip()
    return prompt
def generate_initial_solution(problem_statement: str, initial_structure: str, temperature: float = 0.7) -> str:
    print("[GENERATE_INITIAL_SOLUTION] Starting solution generation")
    concepts = extract_core_concepts_for_search(problem_statement)
    print("[GENERATE_INITIAL_SOLUTION] Searching for similar problems and edge cases")
    search_results = search_for_similar_problems(concepts)
    edge_case_guidance = ""
    if concepts.get("common_edge_cases"):
        edge_case_guidance = f"\n\n**Common Edge Cases to Consider (based on similar problems):**\n"
        for i, case in enumerate(concepts.get("common_edge_cases", []), 1):
            edge_case_guidance += f"{i}. {case}\n"
    if search_results:
        edge_case_guidance += f"\n**Additional Edge Cases from Similar Problems:**\n{search_results[:500]}\n"
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
        9. **IMPORTANT**: Always handle input flexibly—detect input in any format, and convert or preprocess accordingly so your code works for any valid input format.
        10. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
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
        + edge_case_guidance
    )
    INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated code for potential infinite loops and provide a corrected version if issues are found.
        CRITICAL INFINITE LOOP DETECTION:
        1. Check for while True: loops without guaranteed exit conditions
        2. Verify all while loops have clear termination conditions
        3. Ensure recursive functions have proper base cases
        4. Look for loops that depend on external state that might never change
        5. Check for patterns that could lead to infinite iteration
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
            result = EnhancedNetwork.make_request(code_generation_messages, model=selected_model, temperature=temperature)
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
            result2 = EnhancedNetwork.make_request(loop_check_messages, model=selected_model)
            if isinstance(result2, tuple):
                loop_check_response, _ = result2
            else:
                loop_check_response = result2
            # Clean up the final response (use compat response as it's the final validated version)
            solution = clean_code_response(loop_check_response)
            return solution
        except Exception as e:
            retry += 1
            time.sleep(1)
    if retry >= 10:
        return ""
    return ""
def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, run_id, agent_start_time
    agent_start_time = time.time()
    enhancement = ""
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
            problem_type, enhancement = check_problem_type(input_dict.get("problem_statement"))
            if problem_type == PROBLEM_TYPE_FIX:
                result = process_fix_task(input_dict, enhancement)
            else:
                result = process_create_task(input_dict, enhancement)
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
                        contents.append(f"{file_path}\n{f.read()}")
                except Exception as e:
                    logger.error(f"Failed to open file {file_path}: {e}")
            return "\n\n".join(contents)
        except Exception as e:
            logger.error(f"Error in get files to modify: {e}")
            return ""
    total_timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 150
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
        if time.time() - start_time > 600 and len(initial_solutions) == 0:
            break
        initial_solution, _ = single_process_create_task(problem_statement, initial_structure)
        if initial_solution is not None:
            success_count += 1
            os.system("git reset --hard")
            extract_and_write_files(initial_solution)
            patch = tool_manager.get_final_git_patch()
            return patch
        time.sleep(1)
    temperature = 0.0
    initial_solution, _ = basic_approach(initial_structure, problem_statement, temperature=temperature)
    print(f"Initial solution in process_create_task: {initial_solution}")
    if initial_solution is not None:
        os.system("git reset --hard")
        extract_and_write_files(initial_solution)
        patch = tool_manager.get_final_git_patch()
        return patch
    elapsed_time = time.time() - s_time
    return create_task_solve_workflow(
        problem_statement, timeout=total_timeout - elapsed_time - 60, run_id_1=run_id, enhancement=enhancement, should_review=False
    )
def create_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str,
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True,
) -> str:
    global run_id
    run_id = run_id_1
    cot = EnhancedCOT(
        latest_observations_to_keep=30,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "list_directory_structure",
            "get_file_content",
            "search_in_all_files_content",
            "apply_code_edit",
            "run_code",
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "think",
            "finish",
        ],
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
    )
    logger.info(f"Starting main agent execution... Enhancement: {enhancement}")
    logger.info(f"Available tools: {tool_manager.available_tools}")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT,
    )
    enhanced_problem = problem_statement
    if enhancement:
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
    return execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME],
        log_prefix="CREATE_MAIN_AGENT",
    )
def truncate_code_intelligently(code: str, max_length: int = 3000) -> str:
    """
    Truncate code while preserving structure.
    Tries to keep function/class definitions intact.
    """
    if len(code) <= max_length:
        return code
    # Try to find a good truncation point (end of a function/class)
    lines = code.split("\n")
    truncated_lines = []
    current_length = 0
    for line in lines:
        if current_length + len(line) + 1 > max_length - 100:  # Leave some buffer
            # Try to find a good stopping point
            truncated_lines.append("... [truncated] ...")
            break
        truncated_lines.append(line)
        current_length += len(line) + 1
    return "\n".join(truncated_lines)
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
    while retry < 10:
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
            if retry > 4:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return PROBLEM_TYPE_FIX
def summarize_errors(analysis: dict) -> str:
    """Summarize errors from root cause analysis."""
    root_causes = analysis.get("root_causes", [])
    if root_causes:
        return "; ".join(root_causes[:3])
    return "No specific errors identified"
def build_code_comparison_context(previous_attempts: List[dict]) -> str:
    """
    Build a context string comparing code snippets from previous attempts.
    Helps identify patterns that consistently fail.
    """
    if not previous_attempts:
        return ""
    context_parts = []
    context_parts.append("## Previous Attempts Analysis\n")
    for i, attempt in enumerate(previous_attempts[-3:], 1):  # Last 3 attempts
        iteration = attempt.get("iteration", i)
        code = attempt.get("solution_code", "")
        error_summary = attempt.get("error_summary", "No error summary")
        # Truncate code intelligently
        truncated_code = truncate_code_intelligently(code, max_length=2000)
        context_parts.append(f"### Attempt {iteration}")
        context_parts.append(f"**Error Summary:** {error_summary}")
        context_parts.append(f"**Code Snippet:**\n```\n{truncated_code}\n```\n")
        root_causes = attempt.get("root_causes", [])
        if root_causes:
            context_parts.append(f"**Identified Root Causes:**")
            for cause in root_causes[:3]:
                context_parts.append(f"- {cause}")
        context_parts.append("")
    return "\n".join(context_parts)
def extract_patterns_to_use(root_cause_analyses: List[dict]) -> List[str]:
    """Extract code patterns that were closer to working."""
    patterns = []
    pattern_counts = {}
    for analysis in root_cause_analyses:
        good_patterns = analysis.get("code_patterns_to_use", [])
        for pattern in good_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    # Return patterns that appeared in at least 2 analyses
    for pattern, count in pattern_counts.items():
        if count >= 2:
            patterns.append(pattern)
    return list(set(patterns))  # Remove duplicates
def extract_patterns_to_avoid(root_cause_analyses: List[dict]) -> List[str]:
    """Extract code patterns that consistently failed."""
    patterns = []
    pattern_counts = {}
    for analysis in root_cause_analyses:
        failed_patterns = analysis.get("code_patterns_to_avoid", [])
        for pattern in failed_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    # Return patterns that appeared in at least 2 analyses
    for pattern, count in pattern_counts.items():
        if count >= 2:
            patterns.append(pattern)
    return list(set(patterns))  # Remove duplicates
def analyze_root_cause_with_code_snippets(
    problem_statement: str,
    current_attempt: dict,
    previous_attempts: List[dict],
    test_cases: str = "",
) -> dict:
    """
    Analyze root cause of test failure using LLM, including code snippets from previous attempts.
    """
    current_code = current_attempt.get("solution_code", "")
    current_errors = current_attempt.get("test_output", "")
    current_code_truncated = truncate_code_intelligently(current_code, max_length=3000)
    # Build code comparison context
    code_comparison = build_code_comparison_context(previous_attempts)
    # Add test code context if available (helps understand input format)
    # Also try to get actual test file content from previous attempts
    test_code_context = ""
    actual_test_file_content = ""
    # Try to extract actual test file from previous attempts
    for attempt in previous_attempts[-3:]:
        test_output = attempt.get("test_output", "")
        if "Actual Test File Content" in test_output:
            match = re.search(r"Actual Test File Content.*?```\n(.*?)```", test_output, re.DOTALL)
            if match:
                actual_test_file_content = match.group(1)
                break
    # Prioritize actual test file over generated tests
    if actual_test_file_content:
        test_code_truncated = truncate_code_intelligently(actual_test_file_content, max_length=3000)
        test_code_context = (
            f"\n\n**ACTUAL TEST FILE (READ THIS - THIS IS THE REAL TEST THAT'S FAILING):**\n"
            f"```\n{test_code_truncated}\n```\n\n"
            f"**CRITICAL**: This is the ACTUAL test file. Read it carefully to see:\n"
            f"1. How the function is called (exact arguments)\n"
            f"2. What output is EXPECTED (assert statements)\n"
            f"3. The exact input format used\n"
            f"Your solution MUST produce the output that these tests expect. Don't guess - use the test code as your specification.\n"
        )
    elif test_cases:
        test_code_truncated = truncate_code_intelligently(test_cases, max_length=2000)
        test_code_context = f"\n\n**GENERATED TEST CODE (may not match actual tests):**\n```\n{test_code_truncated}\n```\n**IMPORTANT**: Look at how the function is called in the tests above to understand the ACTUAL input format. Don't guess - use the test code as reference.\n"
    # Build prompt for root cause analysis
    ROOT_CAUSE_ANALYSIS_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer and debugger. Your task is to analyze why a solution failed and identify the root causes.
        
        **CRITICAL**: You must analyze BOTH the error messages AND the code that produced them to identify root causes.
        
        Analyze the following:
        1. **Root Causes**: What are the fundamental issues causing the test failures?
        2. **Failed Code Patterns**: What specific code patterns in the solution are problematic?
        3. **Successful Patterns**: What patterns from previous attempts (if any) were closer to working?
        4. **Suggested Approach**: What should be done differently in the next attempt?
        
        Be specific and actionable. Focus on:
        - Logic errors in the code
        - Missing edge case handling
        - Incorrect algorithm implementation
        - Type mismatches or data structure issues
        - Missing or incorrect function implementations
        
        Return your analysis in the following JSON format:
        {{
            "root_causes": ["cause1", "cause2", ...],
            "code_patterns_to_avoid": ["pattern1", "pattern2", ...],
            "code_patterns_to_use": ["pattern1", "pattern2", ...],
            "suggested_approach": "detailed suggestion for next attempt"
        }}
        """
    )
    analysis_messages = [
        {"role": "system", "content": ROOT_CAUSE_ANALYSIS_PROMPT},
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Problem Statement:
                {problem_statement}
                
                {code_comparison}
                
                Current Attempt (Iteration {current_attempt.get('iteration', 'N')}):
                **Code:**
                ```
                {current_code_truncated}
                ```
                
                **Test Output/Errors:**
                ```
                {current_errors[:2000]}
                ```
                {test_code_context}
                
                **CRITICAL INSTRUCTIONS FOR ROOT CAUSE ANALYSIS:**
                1. **READ THE TEST CODE ABOVE**: The test code shows EXACTLY what the test expects. If the test expects X but gets Y, the IMPLEMENTATION is wrong, NOT the test.
                2. **Compare Expected vs Actual**: Look at what the test expects (assert.strictEqual, assert.deepEqual, etc.) and compare it with the actual output in the error message.
                3. **The Test is Always Correct**: Tests define the requirements. If your implementation doesn't match the test expectations, your implementation needs to be fixed.
                4. **BE CONSISTENT**: If previous root cause analyses said one thing, but the test code clearly shows something different, TRUST THE TEST CODE. Don't contradict previous analyses unless you're CERTAIN they were wrong based on the test code.
                5. **Check for Contradictions**: Before suggesting an approach, check if it contradicts what was suggested in previous iterations. If there's a contradiction, carefully re-read the test code to determine which is correct.
                6. **Don't Blame the Test**: Never suggest "the test expectation is wrong" - the test defines what's correct.
                7. **Identify Implementation Bugs**: Focus on what's wrong with the code logic, not the test expectations.
                
                Analyze the root causes and provide your analysis in the JSON format specified.
                """
            ),
        },
    ]
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 5:
        try:
            result = EnhancedNetwork.make_request(
                analysis_messages,
                model=selected_model,
                temperature=0.0,
            )
            if isinstance(result, tuple):
                analysis_text, _ = result
            else:
                analysis_text = result
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group())
                return analysis_json
            else:
                # Fallback: try to parse as JSON directly
                analysis_json = json.loads(analysis_text)
                return analysis_json
        except Exception as e:
            retry += 1
            print(f"[ANALYZE_ROOT_CAUSE] Retry {retry}/5: {e}")
            if retry < 5:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(2)
    # Fallback: return basic analysis
    return {
        "root_causes": ["Failed to analyze root cause"],
        "code_patterns_to_avoid": [],
        "code_patterns_to_use": [],
        "suggested_approach": "Review the error messages and test code carefully",
    }
def is_stuck_in_loop(root_cause_analyses: List[dict], threshold: float = 0.4) -> bool:
    """
    Detect if the agent is stuck in a loop by comparing root causes.
    Returns True if the same root causes appear repeatedly.
    """
    if len(root_cause_analyses) < 3:
        return False
    # Get root causes from last 3 analyses
    recent_causes = []
    for analysis in root_cause_analyses[-3:]:
        causes = analysis.get("root_causes", [])
        recent_causes.append(set(cause.lower() for cause in causes))
    # Check overlap between consecutive analyses
    overlaps = []
    for i in range(len(recent_causes) - 1):
        set1 = recent_causes[i]
        set2 = recent_causes[i + 1]
        if len(set1) > 0 and len(set2) > 0:
            overlap = len(set1 & set2) / max(len(set1), len(set2))
            overlaps.append(overlap)
    if overlaps:
        avg_overlap = sum(overlaps) / len(overlaps)
        # Also check if any single overlap is high (even if average is lower)
        max_overlap = max(overlaps) if overlaps else 0
        return avg_overlap >= threshold or max_overlap >= 0.6
    return False
def build_learning_guidance(
    root_cause_analyses: List[dict],
    previous_attempts: List[dict],
) -> str:
    """
    Build learning guidance from root cause analyses.
    Includes:
    - Code patterns to avoid
    - Code patterns to use
    - Root cause analyses
    """
    # Build learning context
    patterns_to_avoid = extract_patterns_to_avoid(root_cause_analyses)
    patterns_to_use = extract_patterns_to_use(root_cause_analyses)
    # Get latest root cause analysis
    latest_analysis = root_cause_analyses[-1] if root_cause_analyses else None
    suggested_approach = latest_analysis.get("suggested_approach", "") if latest_analysis else ""
    # Check for contradictions in recent root cause analyses using LLM
    contradiction_warning = detect_contradictions_with_llm(root_cause_analyses)
    # Build learning guidance
    learning_guidance = ""
    if contradiction_warning:
        learning_guidance += contradiction_warning
    if patterns_to_avoid:
        learning_guidance += "\n\n**⚠️ CODE PATTERNS TO AVOID (from previous failures):**\n"
        for i, pattern in enumerate(patterns_to_avoid[:5], 1):
            learning_guidance += f"{i}. {pattern}\n"
    if patterns_to_use:
        learning_guidance += "\n\n**✅ CODE PATTERNS TO USE (from previous attempts):**\n"
        for i, pattern in enumerate(patterns_to_use[:5], 1):
            learning_guidance += f"{i}. {pattern}\n"
    # Show recent root cause analyses to help identify contradictions
    if len(root_cause_analyses) >= 2:
        learning_guidance += "\n\n**📋 RECENT ROOT CAUSE ANALYSES (to avoid contradictions):**\n"
        for i, analysis in enumerate(root_cause_analyses[-3:], 1):
            iteration_num = len(root_cause_analyses) - 3 + i
            causes = analysis.get("root_causes", [])[:2]
            approach = analysis.get("suggested_approach", "")[:150]
            learning_guidance += f"Iteration {iteration_num}: {', '.join(causes)}\n"
            if approach:
                learning_guidance += f"  → Suggested: {approach}...\n"
    if suggested_approach:
        learning_guidance += f"\n\n**💡 LATEST SUGGESTED APPROACH:**\n{suggested_approach}\n"
        if contradiction_warning:
            learning_guidance += (
                "\n**⚠️ WARNING**: If this contradicts previous suggestions, READ THE ACTUAL TEST FILE to determine the correct approach.\n"
            )
    return learning_guidance
def detect_contradictions_with_llm(root_cause_analyses: List[dict]) -> str:
    """
    Use LLM to detect contradictions in recent root cause analyses.
    Returns a warning message if contradictions are found, empty string otherwise.
    """
    if len(root_cause_analyses) < 3:
        return ""
    # Get recent analyses (last 3)
    recent_analyses = root_cause_analyses[-3:]
    # Build context for LLM
    analyses_text = ""
    for i, analysis in enumerate(recent_analyses, 1):
        iteration_num = len(root_cause_analyses) - 3 + i
        root_causes = analysis.get("root_causes", [])
        suggested_approach = analysis.get("suggested_approach", "")
        analyses_text += f"\n--- Iteration {iteration_num} ---\n"
        analyses_text += f"Root Causes: {', '.join(root_causes[:3])}\n"
        analyses_text += f"Suggested Approach: {suggested_approach}\n"
    prompt = textwrap.dedent(
        f"""
        Analyze the following root cause analyses from recent iterations. Determine if there are any CONTRADICTIONS or conflicting advice between the iterations.
        {analyses_text}
        Your task:
        1. Identify if there are contradictions
        2. If contradictions exist, provide a clear warning message in the following format:
        **⚠️ CONTRADICTION DETECTED**: [Brief description of the contradiction]
        - [Specific conflicting advice from different iterations]
        **CRITICAL**: Read the ACTUAL TEST FILE below to see what the test expects. The test file is the source of truth - follow what the test expects, not contradictory analysis.
        3. If NO contradictions are found, respond with only: "NO_CONTRADICTION"
        Respond with either the warning message or "NO_CONTRADICTION" (nothing else)."""
    )
    messages = [
        {
            "role": "system",
            "content": "You are an expert at analyzing code debugging advice and detecting contradictions. Be precise and only flag genuine contradictions.",
        },
        {"role": "user", "content": prompt},
    ]
    selected_model = QWEN_MODEL_NAME
    retry = 0
    max_retries = 3
    while retry < max_retries:
        try:
            response, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0, attempt=1)
            response = response.strip()
            # Check if LLM found no contradiction
            if "NO_CONTRADICTION" in response.upper():
                return ""
            # If we got a warning message, return it
            if "CONTRADICTION" in response.upper() or "⚠️" in response:
                return f"\n\n{response}\n"
            # If response doesn't match expected format, assume no contradiction
            return ""
        except Exception as e:
            retry += 1
            logger.warning(f"[detect_contradictions_with_llm] Retry {retry}/{max_retries}: {e}")
            if retry < max_retries:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(1)
    # Fallback: return empty string if all retries failed
    return ""
def get_run_command_for_file(file_path: str) -> list[str]:
    """Get run command for a test file."""
    return llm_select_run_command_for_file(file_path)
def generate_solution_with_code_context(
    problem_statement: str,
    initial_structure: str,
    root_cause_analyses: List[dict],
    previous_attempts: List[dict],
    test_cases: str = "",
    temperature: float = 0.0,
) -> str:
    """
    Generate a new solution using learning from previous attempts.
    Includes code snippets, error logs, and root cause analyses.
    """
    # Build learning guidance
    learning_guidance = build_learning_guidance(root_cause_analyses, previous_attempts)
    # Build code comparison context
    code_comparison = build_code_comparison_context(previous_attempts)
    # Add test code context if available
    test_context = ""
    actual_test_file_content = ""
    # Try to extract actual test file from previous attempts
    for attempt in previous_attempts[-3:]:
        test_output = attempt.get("test_output", "")
        if "Actual Test File Content" in test_output:
            match = re.search(r"Actual Test File Content.*?```\n(.*?)```", test_output, re.DOTALL)
            if match:
                actual_test_file_content = match.group(1)
                break
    # Prioritize actual test file over generated tests
    if actual_test_file_content:
        test_code_truncated = truncate_code_intelligently(actual_test_file_content, max_length=3000)
        test_context = (
            f"\n\n**ACTUAL TEST FILE (READ THIS - THIS IS THE REAL TEST THAT'S FAILING):**\n"
            f"```\n{test_code_truncated}\n```\n\n"
            f"**CRITICAL**: This is the ACTUAL test file. Read it carefully to see:\n"
            f"1. How the function is called (exact arguments)\n"
            f"2. What output is EXPECTED (assert statements)\n"
            f"3. The exact input format used\n"
            f"Your solution MUST produce the output that these tests expect. Don't guess - use the test code as your specification.\n"
        )
    elif test_cases:
        test_code_truncated = truncate_code_intelligently(test_cases, max_length=2000)
        test_context = f"\n\n**GENERATED TEST CODE (may not match actual tests):**\n```\n{test_code_truncated}\n```\n**IMPORTANT**: Look at how the function is called in the tests above to understand the ACTUAL input format. Don't guess - use the test code as reference.\n"
    GENERATE_SOLUTION_WITH_LEARNING_PROMPT = (
        textwrap.dedent(
            """
        You are an expert software engineer. Your task is to generate a complete, working solution for the given problem statement.
        
        **MOST CRITICAL RULE**: You MUST strictly follow the "CODE PATTERNS TO AVOID" and "SUGGESTED APPROACH" sections. Do NOT repeat past mistakes.
        
        **CRITICAL**: You have access to previous attempts and their failures. Use this information to avoid repeating the same mistakes.
        
        Strict Requirements:
        1. Output the full content of files along with their file names. You **MUST** output the **file name** along with file content.
        2. Do not include explanations, comments, or markdown formatting in the main code.
        3. Use only standard libraries and frameworks (no external libraries).
        4. Implement all required classes and functions exactly with the same names as in the initial code stub.
        5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
        6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
        7. The solution must be executable as-is with no placeholders or TODOs.
        8. **LEARN FROM PREVIOUS ATTEMPTS**: Avoid the patterns that failed before. Use the suggested approach.
        9. **IMPORTANT**: Add clear comments above each edge case handling section.
        10. **IMPORTANT**: Always handle input flexibly—detect input in any format, and convert or preprocess accordingly.
        11. **IF YOU SEE ACTUAL TEST FILE**: Read it carefully. The test file shows exactly what output is expected. Your solution must produce that exact output.
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
        + learning_guidance
    )
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": GENERATE_SOLUTION_WITH_LEARNING_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\nInitial structure:\n{initial_structure}\n{code_comparison}\n{test_context}\n\nGenerate the complete and correct implementation in files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
                },
            ]
            result = EnhancedNetwork.make_request(messages, model=selected_model, temperature=temperature)
            if isinstance(result, tuple):
                code_response, _ = result
            else:
                code_response = result
            return clean_code_response(code_response)
        except Exception as e:
            retry += 1
            print(f"[GENERATE_SOLUTION_WITH_CODE_CONTEXT] Retry {retry}/10: {e}")
            if retry < 10:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(2)
    return ""
def single_process_create_task(problem_statement: str, initial_structure: str) -> tuple[str, str] | tuple[None, None]:
    print("[SINGLE_PROCESS_CREATE_TASK] Using root cause learning with code snippets")
    # Use iterative learning system instead of basic approach
    return iterative_learning_with_code_snippets(
        problem_statement,
        initial_structure,
        max_iterations=12,
    )
def iterative_learning_with_code_snippets(
    problem_statement: str,
    initial_structure: str,
    max_iterations: int = 12,
) -> tuple[str, str] | tuple[None, None]:
    """
    Iterative learning system for CREATE workflow.
    Generates solutions, tests them, learns from failures, and refines.
    """
    print(f"[ROOT_CAUSE_LEARNING] Iteration 1/{max_iterations}")
    previous_attempts = []
    root_cause_analyses = []
    test_cases = ""
    comprehensive_validated = False
    for iteration in range(1, max_iterations + 1):
        print(f"[ROOT_CAUSE_LEARNING] Iteration {iteration}/{max_iterations}")
        os.system("git reset --hard")
        # Check timeout
        global agent_start_time
        if agent_start_time is not None:
            elapsed_time = time.time() - agent_start_time
            if elapsed_time >= 1000:
                print(f"[ITERATION {iteration}] ⏰ Timeout reached, exiting...")
                if previous_attempts:
                    last_attempt = previous_attempts[-1]
                    return (last_attempt["solution_code"], test_cases)
                return (None, None)
        # 1. Generate solution
        if iteration == 1:
            print(f"[ITERATION {iteration}] Generating initial solution...")
            solution = generate_initial_solution(problem_statement, initial_structure, temperature=0.0)
        else:
            print(f"[ITERATION {iteration}] Generating solution with learning context...")
            solution = generate_solution_with_code_context(
                problem_statement,
                initial_structure,
                root_cause_analyses,
                previous_attempts,
                test_cases,
                temperature=0.0,
            )
        if not solution:
            print(f"[ITERATION {iteration}] ❌ Failed to generate solution")
            continue
        # 2. Write solution files
        created_files = extract_and_write_files(solution)
        if not created_files:
            print(f"[ITERATION {iteration}] ❌ Failed to write solution files")
            continue
        # 3. Generate tests (only on first iteration or if test_cases is empty)
        # Use enhanced test generation if we have root cause analyses
        if not test_cases or iteration == 1:
            print(f"[ITERATION {iteration}] Generating test cases...")
            if root_cause_analyses and len(root_cause_analyses) > 0:
                # Use enhanced test generation targeting root causes
                latest_analysis = root_cause_analyses[-1]
                root_causes = latest_analysis.get("root_causes", [])
                if root_causes:
                    print(f"[ITERATION {iteration}] Using enhanced test generation targeting {len(root_causes)} root causes...")
                    test_cases = generate_targeted_testset(problem_statement, str(created_files), initial_structure, root_causes, temperature=0.0)
                    if not test_cases:
                        print(f"[ITERATION {iteration}] ⚠️  Enhanced test generation failed, falling back to standard generation...")
                        test_cases = generate_single_testset(problem_statement, str(created_files), initial_structure, temperature=0.0)
                else:
                    test_cases = generate_single_testset(problem_statement, str(created_files), initial_structure, temperature=0.0)
            else:
                test_cases = generate_single_testset(problem_statement, str(created_files), initial_structure, temperature=0.0)
            if not test_cases:
                print(f"[ITERATION {iteration}] ⚠️  Failed to generate test cases, using previous tests")
        # 4. Run tests
        test_files = extract_and_write_files(test_cases)
        print(f"[ITERATION {iteration}] Running tests: {test_files}...")
        all_passed = True
        test_results = []
        for test_file in test_files:
            try:
                run_command = get_run_command_for_file(test_file)
                result = subprocess.run(
                    run_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                passed = is_all_tests_passed(result.stdout)
                test_results.append({"file": test_file, "output": result.stdout, "passed": passed})
                if not passed:
                    all_passed = False
                    print(f"[ITERATION {iteration}] ❌ Test failed: {test_file}")
            except Exception as e:
                print(f"[ITERATION {iteration}] ⚠️  Test execution error: {test_file} - {str(e)}")
                all_passed = False
                test_results.append({"file": test_file, "output": f"Test execution error: {str(e)}", "passed": False})
        # 5. Collect test output
        combined_test_output = ""
        for result in test_results:
            combined_test_output += f"Test: {result['file']}\nOutput: {result['output']}\n\n"
        # 6. Check if all tests passed
        if all_passed:
            print(f"[ITERATION {iteration}] ✅✅✅ ALL TESTS PASSED! ✅✅✅")
            if not comprehensive_validated:
                comprehensive_validated = True
                print(f"[ITERATION {iteration}] 🔍 Running additional comprehensive validation tests...")
                try:
                    comprehensive_tests = generate_single_testset(problem_statement, str(created_files), initial_structure, temperature=0.3)
                    if comprehensive_tests:
                        comprehensive_test_files = extract_and_write_files(comprehensive_tests)
                        comprehensive_all_passed = True
                        for test_file in comprehensive_test_files:
                            try:
                                run_command = get_run_command_for_file(test_file)
                                result = subprocess.run(run_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
                                passed = is_all_tests_passed(result.stdout)
                                if not passed:
                                    comprehensive_all_passed = False
                                    print(f"[ITERATION {iteration}] ⚠️  Comprehensive test failed: {test_file}")
                                    print(f"[ITERATION {iteration}] Test output: {result.stdout[:500]}")
                                    break
                            except Exception as e:
                                print(f"[ITERATION {iteration}] ⚠️  Comprehensive test error: {test_file} - {str(e)}")
                                comprehensive_all_passed = False
                                break
                        if not comprehensive_all_passed:
                            print(
                                f"[ITERATION {iteration}] ⚠️  Initial tests passed but comprehensive tests failed - continuing to refine solution..."
                            )
                            # Don't return, treat as failure and continue to root cause analysis
                            all_passed = False
                            # Collect comprehensive test failure details
                            comprehensive_failure_output = ""
                            for test_file in comprehensive_test_files:
                                try:
                                    run_command = get_run_command_for_file(test_file)
                                    result = subprocess.run(run_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
                                    comprehensive_failure_output += f"\nTest: {test_file}\nOutput: {result.stdout}"
                                except Exception:
                                    pass
                            test_results.append(
                                {
                                    "file": "comprehensive_validation",
                                    "output": f"Initial tests passed but comprehensive validation tests failed - solution may not handle all edge cases. Comprehensive test output: {comprehensive_failure_output[:2000]}",
                                    "passed": False,
                                }
                            )
                            combined_test_output += comprehensive_failure_output
                            # Continue to root cause analysis section below - DON'T return
                        else:
                            print(f"[ITERATION {iteration}] ✅ Comprehensive validation tests also passed!")
                            # Run code review pass before returning
                            print(f"[ITERATION {iteration}] 🔍 Running code review pass...")
                            review_passed, review_issues = review_solution_code(
                                solution,
                                problem_statement,
                                test_cases,
                            )
                            if not review_passed:
                                print(f"[ITERATION {iteration}] ⚠️  Code review found issues: {review_issues[:200]}")
                                print(f"[ITERATION {iteration}] Continuing to refine solution...")
                                # Treat as failure and continue to root cause analysis
                                all_passed = False
                                combined_test_output += f"\n\n**Code Review Issues:**\n{review_issues}"
                                # Continue to root cause analysis section below - DON'T return
                            else:
                                print(f"[ITERATION {iteration}] ✅ Code review passed!")
                                # Only return if comprehensive validation AND code review passed
                                return (solution, test_cases)
                    else:
                        # No comprehensive tests generated, but initial tests passed
                        # Run code review pass before returning
                        print(f"[ITERATION {iteration}] 🔍 Running code review pass...")
                        review_passed, review_issues = review_solution_code(
                            solution,
                            problem_statement,
                            test_cases,
                        )
                        if not review_passed:
                            print(f"[ITERATION {iteration}] ⚠️  Code review found issues: {review_issues[:200]}")
                            print(f"[ITERATION {iteration}] Continuing to refine solution...")
                            # Treat as failure and continue to root cause analysis
                            all_passed = False
                            combined_test_output += f"\n\n**Code Review Issues:**\n{review_issues}"
                            # Continue to root cause analysis section below - DON'T return
                        else:
                            print(f"[ITERATION {iteration}] ✅ Code review passed!")
                            return (solution, test_cases)
                except Exception as e:
                    print(f"[ITERATION {iteration}] ⚠️  Comprehensive validation failed: {str(e)} - proceeding with initial solution")
                    # If comprehensive validation fails due to exception, still return the solution
                    # (initial tests passed, so it's better than nothing)
                    return (solution, test_cases)
            else:
                # Comprehensive validation already ran in a previous iteration
                # Initial tests passed, but run code review pass before returning
                print(f"[ITERATION {iteration}] 🔍 Running code review pass...")
                review_passed, review_issues = review_solution_code(
                    solution,
                    problem_statement,
                    test_cases,
                )
                if not review_passed:
                    print(f"[ITERATION {iteration}] ⚠️  Code review found issues: {review_issues[:200]}")
                    print(f"[ITERATION {iteration}] Continuing to refine solution...")
                    # Treat as failure and continue to root cause analysis
                    all_passed = False
                    combined_test_output += f"\n\n**Code Review Issues:**\n{review_issues}"
                    # Continue to root cause analysis section below - DON'T return
                else:
                    print(f"[ITERATION {iteration}] ✅ Code review passed!")
                    return (solution, test_cases)
        # 7. Analyze root cause if tests failed
        print(f"[ITERATION {iteration}] 🔍 Analyzing root cause with code context...")
        attempt_record = {
            "iteration": iteration,
            "solution_code": solution,
            "test_output": combined_test_output,
            "error_summary": summarize_errors(root_cause_analyses[-1]) if root_cause_analyses else "Test failures",
        }
        root_cause_analysis = analyze_root_cause_with_code_snippets(
            problem_statement,
            attempt_record,
            previous_attempts,
            test_cases,
        )
        root_cause_analyses.append(root_cause_analysis)
        attempt_record["root_causes"] = root_cause_analysis.get("root_causes", [])
        previous_attempts.append(attempt_record)
        print(f"[ITERATION {iteration}] 📊 Root causes identified:")
        for i, cause in enumerate(root_cause_analysis.get("root_causes", [])[:5], 1):
            print(f"  {i}. {cause}")
        print(f"[ITERATION {iteration}] ⚠️  Code patterns to avoid:")
        for pattern in root_cause_analysis.get("code_patterns_to_avoid", [])[:3]:
            print(f"  - {pattern}")
        suggested = root_cause_analysis.get("suggested_approach", "")
        if suggested:
            print(f"[ITERATION {iteration}] 💡 Suggested approach: {suggested[:200]}")
        # 8. Check for loops
        if iteration >= 3:
            if is_stuck_in_loop(root_cause_analyses[-3:]):
                print(f"[ITERATION {iteration}] ⚠️  Detected loop - same root causes repeatedly")
                # Early exit if stuck for too long (save time)
                stuck_iterations = iteration - 2  # Approximate iterations stuck
                if stuck_iterations >= 5:  # Exit after 5 stuck iterations
                    print(f"[ITERATION {iteration}] ⚠️  Stuck in loop for {stuck_iterations} iterations - exiting early to save time")
                    print(f"[ITERATION {iteration}] Returning best attempt so far...")
                    if previous_attempts:
                        last_attempt = previous_attempts[-1]
                        return (last_attempt["solution_code"], test_cases)
                    return (None, None)
                print(f"[ITERATION {iteration}] 🔄 Injecting stronger guidance to break the loop...")
                # Inject a strong hint based on common issues
                if root_cause_analyses:
                    latest_analysis = root_cause_analyses[-1]
                    # Check if agent is blaming the test instead of fixing the code
                    root_causes_text = " ".join(latest_analysis.get("root_causes", [])).lower()
                    # Build a comprehensive summary of what's been tried
                    all_attempted_patterns = []
                    for analysis in root_cause_analyses[-5:]:
                        all_attempted_patterns.extend(analysis.get("code_patterns_to_avoid", []))
                    if "test expectation" in root_causes_text or "test is wrong" in root_causes_text or "test input" in root_causes_text:
                        root_cause_analyses[-1]["code_patterns_to_avoid"].append(
                            "Blaming the test - the test defines what's correct. If output doesn't match test expectations, fix the implementation."
                        )
                        root_cause_analyses[-1]["suggested_approach"] = (
                            "STOP blaming the test. The test is ALWAYS correct. Read the test code carefully to see what output is expected. "
                            "Compare your actual output with the expected output. Fix your implementation logic to match what the test expects. "
                            "The test shows the correct behavior - your code must produce that behavior."
                        )
                    else:
                        # Generic simplification hint
                        root_cause_analyses[-1]["code_patterns_to_avoid"].append(
                            "Over-complicating the solution - try a simpler, more direct approach"
                        )
                        root_cause_analyses[-1]["suggested_approach"] = (
                            "Try a completely different, simpler approach. Read the test code to understand exactly what's expected, "
                            "then implement the simplest solution that produces that expected output."
                        )
        # 9. Sleep before next iteration
        time.sleep(2)
    # Max iterations reached
    print(f"\n[FINAL] Max iterations ({max_iterations}) reached")
    print(f"Total attempts: {len(previous_attempts)}")
    print(f"Total root cause analyses: {len(root_cause_analyses)}")
    # Return the last attempt if available
    if previous_attempts:
        last_attempt = previous_attempts[-1]
        return (last_attempt["solution_code"], test_cases)
    return (None, None)
def process_fix_task(input_dict: Dict[str, Any], enhancement: str):
    """Main entry point for task processing and code modification.
    Parameters
    ----------
    input_dict : dict
        Configuration dictionary containing the task specification.
        Required key: 'problem_statement' with task details.
        Optional keys: 'run_id', 'instance_id' for tracking purposes.
    """
    global run_id
    # setting environment to include current working directory and lib directory
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 150
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()
    try:
        patch_text = fix_task_solve_workflow(problem_text, timeout=timeout - 60, run_id_1=run_id, enhancement=enhancement, should_review=True)
        os.system("git reset --hard")
    except Exception as e:
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logs.append(error_info)
    finally:
        os.chdir(cwd)
    return patch_text
def clean_code_response(response: str) -> str:
    """Clean code response by removing markdown code blocks for any language"""
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response
def generate_targeted_testset(
    problem_statement: str,
    files_to_test: str,
    initial_structure: str,
    root_causes: List[str],
    temperature: float = 0.0,
) -> str:
    """Generate tests specifically targeting identified root causes"""
    GENERATE_TARGETED_TESTCASES_PROMPT = textwrap.dedent(
        """
        You are an expert testcase developer specializing in targeted test generation.
        
        **CRITICAL**: You have been given specific root causes of previous test failures. Your task is to generate tests that specifically target and verify fixes for these issues.
        
        Important points:-
        - Follow the best practices and conventions of the language of the code skeleton.
        - You have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
        - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
        - Use the only built-in testing framework for the language of the code skeleton. **MUST** use the built-in testing framework.
            - For python, use `unittest` to write a test.
            - For javascript, **MUST** use **`node:test` and `node:assert`** to write a test.
            - For other languages, use built-in test frameworks as well.
        - **TARGET SPECIFIC ROOT CAUSES**: Generate tests that specifically check for the root causes provided. Each root cause should be addressed by at least one test case.
        - **EDGE CASES**: Generate tests for edge cases related to the root causes.
        - **BOUNDARY CONDITIONS**: Test boundary conditions that might have caused the failures.
        - **COMPREHENSIVE COVERAGE**: Ensure your tests cover all aspects mentioned in the root causes.
        
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
    root_causes_text = "\n".join([f"- {cause}" for cause in root_causes[:5]])  # Limit to top 5
    retry = 0
    test_generation_messages = [
        {"role": "system", "content": GENERATE_TARGETED_TESTCASES_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nInitial structure:\n{initial_structure}\n\n**ROOT CAUSES TO TARGET (generate tests that specifically check for these):**\n{root_causes_text}\n\nGenerate comprehensive test cases that specifically target the root causes above. Make sure each root cause is addressed by at least one test case.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```\n```javascript\ntest_a.js\ncontents of test_a.js\n\ntest_b.js\ncontents of test_b.js\n```",
        },
    ]
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(test_generation_messages, model=selected_model, temperature=temperature)
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
                test_generation_messages.append({"role": "assistant", "content": testcode_response})
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
            print(f"Exception in generate_targeted_testset: {e}")
            time.sleep(1)
    return ""
def basic_approach(initial_structure: str, problem_statement: str, temperature: float = 0.0) -> tuple[str, str] | tuple[None, None]:
    initial_solution = generate_initial_solution(problem_statement, initial_structure, temperature)
    if not initial_solution:
        return (None, None)
    created_files = extract_and_write_files(initial_solution)
    test_cases = generate_single_testset(problem_statement, str(created_files), initial_structure, temperature)
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
            result = EnhancedNetwork.make_request(test_generation_messages, model=selected_model, temperature=temperature)
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
                test_generation_messages.append({"role": "assistant", "content": testcode_response})
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
            time.sleep(1)
    return ""
def review_solution_code(
    solution_code: str,
    problem_statement: str,
    test_cases: str,
) -> tuple[bool, str]:
    """
    Review code for potential issues even if tests pass.
    Returns (passed, issues_found) tuple.
    """
    CODE_REVIEW_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer. Your task is to review code that has passed all tests but might still have subtle issues.
        
        **CRITICAL**: Even though tests pass, the code might have:
        1. **Edge cases not covered by tests** - Are there edge cases that the tests don't cover?
        2. **Common anti-patterns** - Are there any code smells or anti-patterns?
        3. **Potential bugs** - Are there any subtle bugs that tests might miss?
        4. **Code quality issues** - Is the code maintainable and correct?
        5. **Logic errors** - Are there any logical issues that might not be caught by current tests?
        6. **Missing error handling** - Are edge cases and error conditions properly handled?
        7. **Incorrect assumptions** - Are there any assumptions that might be wrong?
        
        Review the code carefully and identify any potential issues. Be thorough but fair.
        
        Return your review in JSON format:
        {{
            "passed": true/false,
            "issues": ["issue1", "issue2", ...],
            "recommendations": ["recommendation1", "recommendation2", ...]
        }}
        
        If no issues are found, set "passed" to true and "issues" to an empty array.
        """
    )
    solution_truncated = truncate_code_intelligently(solution_code, max_length=4000)
    test_cases_truncated = truncate_code_intelligently(test_cases, max_length=2000)
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 5:
        try:
            messages = [
                {"role": "system", "content": CODE_REVIEW_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\nSolution Code:\n```\n{solution_truncated}\n```\n\nTest Cases:\n```\n{test_cases_truncated}\n```\n\nReview this code for potential issues even though it passes all tests.",
                },
            ]
            result = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            if isinstance(result, tuple):
                review_text, _ = result
            else:
                review_text = result
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", review_text, re.DOTALL)
            if json_match:
                review_json = json.loads(json_match.group())
                passed = review_json.get("passed", True)
                issues = review_json.get("issues", [])
                recommendations = review_json.get("recommendations", [])
                issues_text = "\n".join([f"- {issue}" for issue in issues])
                recommendations_text = "\n".join([f"- {rec}" for rec in recommendations])
                full_issues = (
                    f"Issues Found:\n{issues_text}\n\nRecommendations:\n{recommendations_text}" if issues or recommendations else "No issues found."
                )
                return passed, full_issues
            else:
                # Fallback: try to parse as JSON directly
                review_json = json.loads(review_text)
                passed = review_json.get("passed", True)
                issues = review_json.get("issues", [])
                issues_text = "\n".join([f"- {issue}" for issue in issues]) if issues else "No issues found."
                return passed, issues_text
        except Exception as e:
            retry += 1
            print(f"[REVIEW_SOLUTION_CODE] Retry {retry}/5: {e}")
            if retry < 5:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                if other_models:
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(2)
    # Fallback: if review fails, pass (don't block on review failure)
    print(f"[REVIEW_SOLUTION_CODE] ⚠️  Review failed after retries, passing by default")
    return True, "Review system unavailable - passed by default"
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
            f"[{log_prefix}] Elapsed time: {elapsed_time}/{timeout} seconds, Usage: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
        )
        if cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0):
            logger.warning(f"[{log_prefix}] Usage exceeded limit: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD")
            break
        if time.time() - start_time > timeout:
            logger.info(f"[{log_prefix}] Global timeout reached")
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
            logger.info(f"[ADAPTIVE_STRATEGY] Thought repeated {cot.repeated_thoughts} times")
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
            models_to_try = [selected_model] + [m for m in models if m != selected_model]
            (
                next_thought,
                next_tool_name,
                next_tool_args,
                raw_text,
                total_attempts,
                error_counter,
                messages,
                used_model,
            ) = EnhancedNetwork.inference(messages, model=models_to_try, run_id=run_id, temperature=temperature)
            selected_model = used_model
            inference_duration = time.time() - inference_start_time
        except Exception as e:
            inference_duration = 0
            logger.error(f"[{log_prefix}] Inference error: {e}")
            is_timeout_error = "Agent execution timeout" in str(e)
            if is_timeout_error:
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
                return tool_manager.get_final_git_patch()
        # Handle both single and multiple tool calls
        tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]
        logger.info(f"[{log_prefix}] Used model: {selected_model}, Inference time: {inference_duration:.2f}s")
        logger.info(f"[{log_prefix}] Next thought: {next_thought}\n\n")
        logger.info(f"[{log_prefix}] About to execute {len(tool_names_list)} tool call(s): {tool_names_list}\n")
        logger.info(f"[{log_prefix}] Tool arguments: {json.dumps(tool_args_list, indent=4)}\n\n")
        all_observations = []
        all_successful = True
        for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
            try:
                if '"' in tool_name or "'" in tool_name:
                    tool_name = tool_name.replace('"', "").replace("'", "")
                observation = tool_manager.get_tool(tool_name)(**tool_args) if tool_args else tool_manager.get_tool(tool_name)()
                # Track file modifications for multi-file change awareness
                if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
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
                    observation_path = tool_manager._save_large_observation(str(observation), tool_name)
                    observation = f"Tool output from `{tool_name}` exceeded token limit ({estimated_tokens} tokens > 4000 tokens limit). The full output has been saved to: {observation_path}. You can read this file using the get_file_content tool if needed."
                all_observations.append(observation)
            except EnhancedToolManager.Error as e:
                error_msg = f"Tool {idx+1} ({tool_name}) error: {e.message}"
                all_observations.append(error_msg)
                all_successful = False
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                error_msg = f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                all_observations.append(error_msg)
                all_successful = False
        # Combine observations
        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                [f"Tool {i+1} ({tool_names_list[i]}):\n{obs}" for i, obs in enumerate(all_observations)]
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
def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
    if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(
                ["git", "config", "--global", "user.email", "agent@sandbox.local"],
                check=True,
            )
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            result = subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.error(f"ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)
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
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": ENHANCEMENT_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n\n{problem_statement}",
                },
            ]
            enhanced, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            return enhanced
        except Exception as e:
            retry += 1
            if retry > 2:
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return ""
def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    def extract_file_names_using_llm(initial_solution: str) -> list:
        retry = 0
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
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
                    model=selected_model,
                )
                return json.loads(result.replace("```json", "").replace("```", "").strip())
            except Exception as e:
                retry += 1
                if retry > 2:
                    other_models = [model for model in AGENT_MODELS if model != selected_model]
                    selected_model = other_models[random.randint(0, len(other_models) - 1)]
                time.sleep(1)
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
                file_content = file_content.rstrip() + "\n" if file_content.strip() else file_content
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
                (f for f in file_names if f == stripped or f.endswith("/" + stripped) or f.split("/")[-1] == stripped),
                stripped,
            )
            current_file, content = current_file, []
        elif current_file:
            content.append(line)
    write_file()
    return created_files