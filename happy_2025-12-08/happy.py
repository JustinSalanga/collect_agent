from __future__ import annotations
import difflib
import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import requests
run_id = None
agent_start_time = None
_current_tool_manager = None
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))
PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
MAX_FIX_TASK_STEPS = 200
LATEST_OBSERVATIONS_TO_KEEP = 20
SUMMARIZE_BATCH_SIZE = 6
MAX_SUMMARY_RANGES = 6
GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
GLM_OLD_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [
    model
    for model in [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]
    for _ in range(2)
]
STOP_INSTRUCTION = textwrap.dedent("""
# 🎯 RESPONSE REQUIREMENTS
- DO NOT generate `observation:` - it will be provided by the system
- You can make MULTIPLE tool calls in one response using tool_call_1, tool_call_2, tool_call_3, etc.
- For efficiency: Batch related operations together (e.g., edit + test in ONE response)
- Format: next_thought: ... followed by one or more tool_call_N blocks
""")
DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response:
{previous_response}
Try a different approach:
1. If you just searched, try reading the file instead
2. If you just edited, try running tests to verify
3. If tests failed, try a different fix approach
""")
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
Now let's start.
```
{problem_statement}
```
""")
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
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
- Start with exploration:
`list_directory_structure` - See what files exist
`search_in_all_files_content` - Find relevant code (functions, errors, patterns)
`get_file_content` - Read specific files or sections
- For deeper understanding:
`summarize_file` - Quick overview of large files
`analyze_file_structure` - Detailed analysis of complex code
`get_function_body` - Extract specific functions/methods
- Make changes and test:
`apply_code_edit` - Fix code (use precise search/replace)
`run_code` - Test your changes (ALWAYS test after editing!)
`create_new_file` - Add new files if needed
- When done:
`finish` - Only when everything works and all tests pass
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
""")
FORMAT_PROMPT = textwrap.dedent("""
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
    tool_args: {"file_path": "file.js", "run_command": ["node", "file.js"]}
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
""")
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
try:
    from tree_sitter import Parser
    from tree_sitter_language_pack import get_language
except ImportError:
    Parser = None
    get_language = None
_codeparse_util_language_cache = {}
class CodeParseUtil:
    """Utility class for parsing and extracting code from files and patches(replace ast.parse functionality to work with any languages powered by LLMs)"""
    def __init__(self):
        self._parsers = {}
    
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
        return ("other", None)
    def detect_language(self, source: str, file_path: str | None = None) -> str | None:
        global _codeparse_util_language_cache
        if file_path and not os.path.exists(file_path) or not source or not source.strip():
            return None
        if file_path:
            file_path = os.path.abspath(file_path) if file_path else None
            if file_path and file_path in _codeparse_util_language_cache:
                return _codeparse_util_language_cache[file_path]
        stripped_source = source.strip()
        sample = stripped_source if len(stripped_source) <= 1000 else f"{stripped_source[:500]}\n\n... [middle content omitted] ...\n\n{stripped_source[-500:]}"
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
                result, _ = EnhancedNetwork.make_request(messages=messages, model=models_to_try[retry % len(models_to_try)], temperature=0.0)
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
        if not node.children:
            return None
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
                    if result is not None:
                        return result
        elif node_type == "function":
            name = internal_name = None
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
                        if source_lines[i].strip().startswith('@'):
                            at_start = i
                        elif source_lines[i].strip():
                            break
                    return {'start_line': at_start + 1, 'end_line': node.end_point[0] + 1}
            for child in node.children:
                result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
                if result is not None:
                    return result
        for child in node.children:
            result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
            if result is not None:
                return result
        return None
    def get_function_body(self, file_path: str, function_name: str, add_line_numbers: bool = False) -> str:
        if not function_name or not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return ""
        if not source or Parser is None:
            return ""
        try:
            source_bytes, source_lines = bytes(source, 'utf8'), source.splitlines()
            language = self.detect_language(source, file_path=file_path)
            if not language:
                return ""
            parser = self._get_parser(language)
            if parser is None:
                return ""
            tree = parser.parse(source_bytes)
            target_qualified, target_simple = function_name, function_name.split('.')[-1]
            func_info = self._find_specific_function(tree.root_node, source_lines, target_qualified, target_simple)
            if func_info is None:
                return ""
            start_idx, end_idx = func_info['start_line'] - 1, func_info['end_line'] - 1
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines):
                body_lines = source_lines[start_idx:end_idx + 1]
                return '\n'.join(f"{start_idx + i + 1}| {line}" for i, line in enumerate(body_lines)) if add_line_numbers else '\n'.join(body_lines)
        except Exception as e:
            logger.warning(f"Error finding function {function_name} in {file_path}: {e}")
        return ""
class SearchManager:
    def search_in_all_files(self, grep_search_command: str) -> str:
        cmd = grep_search_command.lstrip()
        if not cmd.startswith("grep"):
            return f"Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        try:
            result = subprocess.run(
                ["bash", "-c", grep_search_command],
                capture_output=True,
                text=True,
                timeout=45
            )
        except Exception as e:
            return f"Error: Failed to execute grep command: {e}"
        if result.returncode > 1:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout
        if not output.strip():
            return "No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output
    def search_in_file(self, file_path: str, search_term: str) -> str:
        def extract_matches(filepath, term, max_output_lines=1000):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception as e:
                return f"Error reading '{filepath}': {e}"
            match = re.escape(term)
            match_lines = [i + 1 for i, line in enumerate(lines) if match in line]
            if not match_lines:
                return f"'{term}' not found in file '{filepath}'"
            context = 20
            seen = set()
            chunks = []
            for ln in match_lines:
                start = max(1, ln - context)
                end = min(len(lines), ln + context)
                rkey = (start, end)
                if rkey in seen:
                    continue
                seen.add(rkey)
                chunk = lines[start - 1:end]
                chunks.append(f"(lines {start}-{end}):\n" + "\n".join(chunk))
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)
        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output
class ModelSwitcher:
    """
    Intelligent model switching strategy based on research:
    - Error-type aware routing (MixLLM approach)
    - Performance tracking per error type
    - Context-aware fallback chains
    - Exponential backoff with smart model selection
    """
    def __init__(self):
        # Track model performance by error type (error_type -> {model: success_count})
        self.model_performance: Dict[str, Dict[str, int]] = {}
        # Track recent failures to avoid immediate retry on same model
        self.recent_failures: List[tuple[str, str]] = []  # (model, error_type)
        self.max_recent_failures = 10
        
        # Error-type specific fallback chains (prioritized order)
        # Based on research: different errors benefit from different model characteristics
        self.error_fallback_chains = {
            "RATE_LIMIT_EXCEEDED": [KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME, GLM_MODEL_NAME],
            "TIMEOUT": [DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, GLM_MODEL_NAME],
            "NETWORK_ERROR": [GLM_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME],
            "INVALID_RESPONSE_FORMAT": [QWEN_MODEL_NAME, GLM_MODEL_NAME, DEEPSEEK_MODEL_NAME, KIMI_MODEL_NAME],
            "INCOMPLETE_RESPONSE": [QWEN_MODEL_NAME, GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME],
            "RESERVED_TOKEN_PRESENT": [DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME, GLM_MODEL_NAME, KIMI_MODEL_NAME],
            "EMPTY_RESPONSE": [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME],
            "UNKNOWN": [QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME, KIMI_MODEL_NAME, GLM_MODEL_NAME],
        }
        
        # Default fallback chain (round-robin from AGENT_MODELS)
        self.default_fallback = AGENT_MODELS.copy()
    
    def get_next_model(self, current_model: str, error_type: str = None, attempt: int = 0) -> str:
        """
        Get next model based on error type and performance history.
        
        Strategy:
        1. If error_type specified, use error-specific fallback chain
        2. Avoid recently failed (model, error) combinations
        3. Prefer models with better performance for this error type
        4. Fallback to round-robin if no specific strategy
        """
        # Clean recent failures if too many
        if len(self.recent_failures) > self.max_recent_failures:
            self.recent_failures = self.recent_failures[-self.max_recent_failures:]
        
        # Get error-specific fallback chain
        if error_type and error_type in self.error_fallback_chains:
            fallback_chain = self.error_fallback_chains[error_type]
        else:
            fallback_chain = self.default_fallback
        
        # Filter out recently failed models for this error type
        # If error_type is None, filter by model only (any error type)
        if error_type:
            recent_failed_models = {
                model for model, err in self.recent_failures[-3:] 
                if err == error_type
            }
        else:
            # For None error_type, avoid models that failed recently for any reason
            recent_failed_models = {
                model for model, err in self.recent_failures[-3:]
            }
        
        # Try to find a model not in recent failures
        for model in fallback_chain:
            if model not in recent_failed_models:
                # Check performance history if available
                if error_type and error_type in self.model_performance:
                    perf = self.model_performance[error_type].get(model, 0)
                    # Prefer models with better performance, but don't exclude others
                    if perf >= 0:  # Only exclude if negative (many failures)
                        return model
                else:
                    return model
        
        # If all models recently failed, use round-robin with offset
        if current_model in fallback_chain:
            current_idx = fallback_chain.index(current_model)
            next_idx = (current_idx + attempt + 1) % len(fallback_chain)
            return fallback_chain[next_idx]
        
        # Final fallback: round-robin from AGENT_MODELS
        if current_model in AGENT_MODELS:
            current_idx = AGENT_MODELS.index(current_model)
            next_idx = (current_idx + attempt + 1) % len(AGENT_MODELS)
            return AGENT_MODELS[next_idx]
        
        # Ultimate fallback
        return fallback_chain[attempt % len(fallback_chain)]
    
    def record_success(self, model: str, error_type: str = None):
        """Record successful model usage for performance tracking."""
        if error_type:
            if error_type not in self.model_performance:
                self.model_performance[error_type] = {}
            self.model_performance[error_type][model] = \
                self.model_performance[error_type].get(model, 0) + 1
        else:
            # Track general success (no specific error type)
            # This helps build overall model preference
            if "GENERAL" not in self.model_performance:
                self.model_performance["GENERAL"] = {}
            self.model_performance["GENERAL"][model] = \
                self.model_performance["GENERAL"].get(model, 0) + 1
    
    def record_failure(self, model: str, error_type: str = None):
        """Record failed model usage."""
        if error_type:
            # Track in recent failures
            self.recent_failures.append((model, error_type))
            
            # Update performance (decrement success count)
            if error_type not in self.model_performance:
                self.model_performance[error_type] = {}
            self.model_performance[error_type][model] = \
                self.model_performance[error_type].get(model, 0) - 1
        else:
            # Track general failure
            if "GENERAL" not in self.model_performance:
                self.model_performance["GENERAL"] = {}
            self.model_performance["GENERAL"][model] = \
                self.model_performance["GENERAL"].get(model, 0) - 1
    
    def get_adaptive_delay(self, error_type: str, attempt: int, base_delay: float = 1.0) -> float:
        """
        Calculate adaptive delay based on error type and attempt number.
        Implements exponential backoff with error-type specific adjustments.
        """
        # Base exponential backoff
        delay = base_delay * (1.5 ** attempt)
        
        # Error-type specific adjustments
        if error_type == "RATE_LIMIT_EXCEEDED":
            # Longer delay for rate limits
            delay *= 2.0
        elif error_type == "TIMEOUT":
            # Moderate delay for timeouts
            delay *= 1.5
        elif error_type == "NETWORK_ERROR":
            # Longer delay for network issues
            delay *= 2.5
        
        # Cap maximum delay
        max_delay = 30.0
        return min(delay, max_delay)
    
    def get_statistics(self) -> dict:
        """
        Get ModelSwitcher performance statistics for monitoring/debugging.
        Returns a summary of model performance by error type.
        """
        stats = {
            "model_performance": dict(self.model_performance),
            "recent_failures_count": len(self.recent_failures),
            "recent_failures": self.recent_failures[-5:] if self.recent_failures else []
        }
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics (useful for new workflow runs)."""
        self.model_performance.clear()
        self.recent_failures.clear()
_model_switcher = ModelSwitcher()
class EnhancedNetwork:
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        tool_name_match = re.search(r"tool_name\s*:\s*([^\s]+)", block, re.IGNORECASE)
        if not tool_name_match:
            return None
        tool_name = tool_name_match.group(1).strip('"\'')
        args_match = re.search(r"tool_args\s*:\s*\{", block, re.IGNORECASE)
        if not args_match:
            return None
        args_start = args_match.end() - 1
        json_str = cls._extract_balanced_braces(block, args_start)
        if json_str:
            try:
                tool_args = json.loads(json_str)
                return {"tool_name": tool_name, "tool_args": tool_args}
            except json.JSONDecodeError:
                try:
                    tool_args = json.loads(json_str.replace("'", '"'))
                    return {"tool_name": tool_name, "tool_args": tool_args}
                except Exception:
                    pass
        return None
    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        if start_pos >= len(text):
            return None
        brace_count, in_string, escape_next, start = 0, False, False, -1
        for i in range(start_pos, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    if start == -1:
                        start = i
                    brace_count += 1
                elif c == "}":
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
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
        except Exception:
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
    @classmethod
    def inference(
        cls,
        messages: list[dict],
        model: str,
        run_id: str = str(uuid4()),
        temperature: float = 0.0,
    ) -> dict:
        models = [model] if isinstance(model, str) else model
        cleaned_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"system", "user", "assistant", "tool"}
            and (m.get("role") != "assistant" or m.get("content", "").strip())
        ]
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")
        result = cls._request_next_action_with_retry(
            cleaned_msgs, models=models, temperature=temperature
        )
        return result
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
    ):
        global run_id, DEFAULT_TIMEOUT, agent_start_time
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 200
        if agent_start_time is not None:
            elapsed_time = time.time() - agent_start_time
            if elapsed_time >= timeout:
                raise RuntimeError(f"HTTP ERROR: Agent execution timeout after {elapsed_time:.2f} seconds (limit: {DEFAULT_TIMEOUT} seconds)")
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
        for i in range(attempts):
            try:
                resp = requests.post(
                    url, json=request_data, timeout=(30, timeout), headers=headers
                )
                resp.raise_for_status()
                try:
                    resp_json = resp.json()
                except JSONDecodeError as e:
                    if i >= attempts - 1:
                        raise ValueError(f"HTTP ERROR: Invalid JSON response for model {model} after {attempts} attempts: {e}")
                    continue
                try:
                    raw_text = resp_json["content"]
                    tool_calls = resp_json["tool_calls"]
                except Exception:
                    raise RuntimeError(
                        f"HTTP ERROR: Response Parse Error timeout for model {model} after {attempts} attempts"
                    )
                if (tool_mode == "none" and not raw_text) or (
                    tool_mode != "none" and not tool_calls
                ):
                    raise RuntimeError(
                        f"HTTP ERROR: NO RESPONSE FOUND Tool model {model} after {attempts} attempts"
                    )
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                if i >= attempts - 1:
                    raise RuntimeError(
                        f"HTTP ERROR: Request timeout for model {model} after {attempts} attempts"
                    )
                time.sleep(1)
            except requests.exceptions.ConnectionError as e:
                if i >= attempts - 1:
                    raise RuntimeError(
                        f"HTTP ERROR: Connection error for model {model} after {attempts} attempts: {e}"
                    )
                time.sleep(1)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(
                            f"HTTP ERROR 504: Gateway Timeout for model {model} after {attempts} attempts: {e}"
                        )
                    time.sleep(1)
                    continue
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model}"
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if i >= attempts - 1:
                    raise RuntimeError(
                        f"HTTP ERROR: Request failed for model {model} after {attempts} attempts: {e}"
                    )
                time.sleep(1)
        raise RuntimeError(
            f"HTTP ERROR: Failed to get response for model {model} after {attempts} attempts"
        )
    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = r",\s*".join(rf'"{k}": (.*)' for k in arguments)
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        return {
            k: match.group(i + 1).strip().strip('"').replace("\\n", "\n")
            for i, k in enumerate(arguments)
        }
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
        models: list[str],
        max_retries: int = 3,
        base_delay: float = 1.0,
        temperature: float = 0.0,
    ) -> str:
        raw_text = None
        error_counter = cls.get_error_counter()
        next_thought = next_tool_name = next_tool_args = None
        total_attempts = 0
        current_model_idx = 0
        used_model = models[0] if models else None
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                current_model = models[min(current_model_idx, len(models) - 1)]
                used_model = current_model
                raw_text, _ = cls.make_request(
                    messages, model=current_model, temperature=temperature
                )
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = (
                    cls.parse_response(raw_text)
                )
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                is_504_error = "504" in error_body or "HTTP ERROR 504" in error_body or "Gateway Timeout" in error_body
                if "Agent execution timeout" in error_body:
                    raise RuntimeError(error_body)
                if is_504_error and current_model_idx < len(models) - 1:
                    current_model_idx += 1
                    time.sleep(3)
                    continue
                if attempt < max_retries - 1:
                    matched = False
                    for key in [
                        "RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", "EMPTY_RESPONSE",
                        "TIMEOUT", "Invalid JSON", "Invalid response"
                    ]:
                        if key in error_body:
                            attr_name = key if key in cls.ErrorType.__members__ else "INVALID_RESPONSE_FORMAT"
                            error_counter[attr_name] += 1
                            matched = True
                            break
                    if not matched:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    skip_http = any(
                        x in error_body for x in [
                            "HTTP ERROR", "RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", 
                            "EMPTY_RESPONSE", "TIMEOUT", "NETWORK_ERROR", "HTTP ERROR 429", "INCOMPLETE_RESPONSE"
                        ]
                    )
                    if not skip_http:
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
        text_resp = re.sub(r"['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub(r"['\"]*observation['\"]*:", "observation:", text_resp)
        text_resp = re.sub(r"['\"]*tool_call_['\"]*", "tool_call_", text_resp)
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
                text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0]
                .strip().strip("\n").strip("'").strip('"').strip()
            )
            # Overwrite with normalized line
            text_resp = re.sub(
                f"next_tool_name:['\" ]*{re.escape(next_tool_name)}['\" ]*",
                "next_tool_name: " + next_tool_name,
                text_resp,
            )
        return text_resp
    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        try:
            return Utils.load_json(next_tool_args.strip())
        except JSONDecodeError:
            try:
                return cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(
                        tool_name, required=True
                    ),
                    next_tool_args,
                )
            except (EnhancedToolManager.Error, Exception):
                raise Exception(f"Invalid JSON: {next_tool_args}")
    @classmethod
    def is_valid_response(cls, raw_text: str) -> bool:
        if isinstance(raw_text, dict) and raw_text.get("error"):
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        stripped = raw_text.strip()
        lower = raw_text.lower()
        has_next_thought = "next_thought" in lower or "<next_thought>" in lower
        has_next_tool_name = "next_tool_name" in lower or "<next_tool_name>" in lower
        has_next_tool_args = "next_tool_args" in lower or "<next_tool_args>" in lower
        valid_ending = (
            stripped.endswith("}")
            or stripped.endswith("}]")
            or stripped.endswith("</next_tool_args>")
            or stripped.endswith(">")
        )
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
            return False, cls.ErrorType.INCOMPLETE_RESPONSE.name
        if not raw_text:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        return cls.is_http_response(raw_text)
    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}
    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str | None, any, any, str | None]:
        error_msg = None
        text_resp = text_resp.strip()
        if "observation:" in text_resp.lower():
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[0].strip()
        text_resp = cls.sanitise_text_resp(text_resp)
        if "Infrastructure is at maximum capacity" in text_resp:
            return None, None, None, "HTTP ERROR Maximum Capacity"
        if "No instances available" in text_resp:
            return None, None, None, "HTTP ERROR NO INSTANCES AVAILABLE"
        next_thought = None
        for pat in [
            r"next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))",
            r"next_thought\s*:\s*(.*?)(?=\ntool_call_)",
            r"next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)",
            r"next_thought\s*:\s*(.*)"
        ]:
            match = re.search(pat, text_resp, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) > 2:
                    next_thought = candidate
                    break
        if not next_thought:
            next_thought = "Processing request"
        tool_call_matches = list(re.finditer(r"tool_call_(\d+)\s*:", text_resp, re.IGNORECASE))
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = tool_call_matches[i + 1].start() if i + 1 < len(tool_call_matches) else len(text_resp)
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                return next_thought, None, None, "Multi-tool format detected but no valid tool calls extracted"
            tool_names = [c["tool_name"] for c in tool_calls]
            tool_args_list = [c["tool_args"] for c in tool_calls]
            if len(tool_names) == 1:
                return next_thought, tool_names[0], tool_args_list[0], error_msg
            return next_thought, tool_names, tool_args_list, error_msg
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp:
            name_idx = text_resp.find("next_tool_name:")
            args_idx = text_resp.find("next_tool_args:")
            if text_resp.find("next_thought:") < name_idx < args_idx:
                next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip()
                next_tool_args_raw = (
                    text_resp.split("next_tool_args:")[1].strip()
                    .split("next_thought:")[0].strip()
                )
                try:
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(next_tool_names, next_tool_args_raw)
                    next_tool_args_list = parsed_args if isinstance(parsed_args, list) else [parsed_args for _ in next_tool_names]
                    if len(next_tool_names) == 1:
                        return next_thought, next_tool_names[0], next_tool_args_list[0], error_msg
                    return next_thought, next_tool_names, next_tool_args_list, error_msg
                except (JSONDecodeError, Exception) as e:
                    error_msg = f"Invalid JSON in tool args: {str(e)}"
                    return next_thought, None, None, error_msg
        if "next_thought:" not in text_resp:
            error_msg = "Invalid response. next_thought not found"
        elif "next_tool_name:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. No tool calls found (expected next_tool_name: or tool_call_N:)"
        elif "next_tool_args:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. next_tool_args not found"
        else:
            error_msg = "Invalid response format. Could not parse tool calls."
        return next_thought, None, None, error_msg
    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        """
        Fix JSON string using LLM with intelligent model switching.
        Uses QWEN as primary (good at structured output) with error-aware fallback.
        """
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        selected_model = QWEN_MODEL_NAME  # Primary choice for structured output
        retry = 0
        last_error_type = None
        while retry < 10:
            try:
                response, _ = cls.make_request(messages, model=selected_model)                
                # Record success - always record to build performance history
                if last_error_type:
                    _model_switcher.record_success(selected_model, last_error_type)
                else:
                    _model_switcher.record_success(selected_model, None)
                break
            except Exception as e:
                retry += 1
                error_body = str(e)
                # Extract error type for intelligent switching
                if "RATE_LIMIT_EXCEEDED" in error_body:
                    last_error_type = "RATE_LIMIT_EXCEEDED"
                elif "TIMEOUT" in error_body:
                    last_error_type = "TIMEOUT"
                elif "NETWORK_ERROR" in error_body:
                    last_error_type = "NETWORK_ERROR"
                else:
                    last_error_type = "INVALID_RESPONSE_FORMAT"  # JSON fixing likely format issues
                
                # Record failure and get next model
                _model_switcher.record_failure(selected_model, last_error_type)
                selected_model = _model_switcher.get_next_model(selected_model, last_error_type, retry)
                
                # Adaptive delay
                delay = _model_switcher.get_adaptive_delay(last_error_type, retry, 2.0)
                time.sleep(delay)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
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
        if isinstance(messages, list):
            text = " ".join(
                str(m.get("content", "") if isinstance(m, dict) else m)
                for m in messages
            )
        else:
            text = messages
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for token in tokens:
            if token.isspace():
                continue
            elif len(token) == 1:
                count += 1
            else:
                count += max(1, (len(token) + 2) // 3)
        return count
    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
            if fixed_json:
                return fixed_json
            else:
                raise JSONDecodeError("Invalid JSON", json_string, 0)
class EnhancedCOT:
    def __init__(self, latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        self.summaries = {}
        self.summarized_ranges = []
    def _get_summary_for_index(self, idx):
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None
    def _check_and_summarize_if_needed(self):
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep
        if cutoff_idx < self.summarize_batch_size:
            return
        unsummarized = 0
        for s, e in sorted(self.summarized_ranges):
            if s <= unsummarized < e:
                unsummarized = e
            elif s > unsummarized:
                break
        if unsummarized >= cutoff_idx:
            return
        summarize_start = unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        batch_size = summarize_end - summarize_start
        if batch_size >= self.summarize_batch_size:
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()
    def _summarize_messages_batch(self, start_idx, end_idx):
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if getattr(thought, 'is_deleted', False):
                continue
            assistant_part = (f"next_thought: {thought.next_thought}\n"
                              f"next_tool_name: {thought.next_tool_name}\n"
                              f"next_tool_args: {thought.next_tool_args}\n")
            obs = thought.observation
            if isinstance(obs, (list, tuple)):
                try:
                    obs_render = json.dumps(list(obs), ensure_ascii=False)
                except Exception:
                    obs_render = str(obs)
            else:
                obs_render = str(obs) if obs else ""
            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append(
                {
                    "assistant": assistant_part,
                    "user": user_part,
                    "is_error": getattr(thought, 'is_error', False),
                }
            )
        if not conversation_parts:
            return None
        conv_lines = []
        for idx, part in enumerate(conversation_parts, 1):
            conv_lines.append(f"\n--- Step {idx} ---")
            conv_lines.append(f"Assistant: {part['assistant']}")
            user_obs = part["user"]
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conv_lines.append(f"User: {user_obs}")
            if part.get("is_error"):
                conv_lines.append("[Error occurred]")
        conversation_text = "\n".join(conv_lines)
        summarization_prompt = textwrap.dedent(f"""
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
        """)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversation history concisely.",
            },
            {"role": "user", "content": summarization_prompt},
        ]
        for _ in range(3):
            try:
                response, _ = EnhancedNetwork.make_request(
                    messages, model=QWEN_MODEL_NAME, temperature=0.0
                )
                return response.strip()
            except Exception:
                time.sleep(1)
        return None
    def is_thought_repeated(self):
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if (last.next_tool_name == prev.next_tool_name and
            last.next_tool_args == prev.next_tool_args):
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False
    def add_action(self, action):
        self.thoughts.append(action)
        if len(self.thoughts) >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True
    def to_str(self):
        messages = []
        last_summary_range = None
        allowed_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:]) if self.summarized_ranges else set()
        total = len(self.thoughts)
        keep_last = self.latest_observations_to_keep
        for i, thought in enumerate(self.thoughts):
            if getattr(thought, 'is_deleted', False):
                continue
            recent = i >= total - keep_last
            if not recent:
                summary = self._get_summary_for_index(i)
                if summary:
                    found_range = False
                    for (start, end), _ in self.summaries.items():
                        if start <= i < end:
                            cur_range = (start, end)
                            if cur_range not in allowed_ranges:
                                found_range = True
                                break
                            if cur_range != last_summary_range:
                                messages.append({
                                    "role": "system",
                                    "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"
                                })
                                last_summary_range = cur_range
                            found_range = True
                            break
                    if found_range:
                        continue
                assistant_str = (f"next_thought:{thought.next_thought}\n"
                                 f"next_tool_name:{thought.next_tool_name}\n"
                                 f"next_tool_args:{thought.next_tool_args}")
                obs = thought.observation
                if isinstance(obs, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(obs), ensure_ascii=False)
                    except Exception:
                        obs_render = str(obs)
                else:
                    obs_render = str(obs) if obs else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
            else:
                if thought.is_error is None or i == total - 1:
                    assistant_str = (f"next_thought:{thought.next_thought}\n"
                                     f"next_tool_name:{thought.next_tool_name}\n"
                                     f"next_tool_args:{thought.next_tool_args}")
                    obs = thought.observation
                    if isinstance(obs, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(obs), ensure_ascii=False)
                        except Exception:
                            obs_render = str(obs)
                    else:
                        obs_render = str(obs)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error is None and thought.is_error is not None:
                        assistant_str = (f"next_thought:{thought.next_thought}\n"
                                         f"next_tool_name:{thought.next_tool_name}\n"
                                         f"next_tool_args:{thought.next_tool_args}")
                        obs = thought.observation
                        if obs is None:
                            obs_len = 0
                        elif isinstance(obs, (list, tuple)):
                            obs_len = len(obs)
                        else:
                            obs_len = len(str(obs).splitlines())
                        user_str = f"observation: error ocurred. detailed output omitted ({obs_len}) lines\n"
                    else:
                        assistant_str = (f"next_thought:{thought.next_thought}\n"
                                         f"next_tool_name:{thought.next_tool_name}\n"
                                         f"next_tool_args:{thought.next_tool_args}")
                        obs = thought.observation
                        if isinstance(obs, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(obs), ensure_ascii=False)
                            except Exception:
                                obs_render = str(obs)
                        else:
                            obs_render = str(obs)
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
            observation,
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
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return "\n".join(numbered_lines)
        if search_term and search_in_file_callback:
            return search_in_file_callback(file_path, search_term)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, start_idx + 1)
                else:
                    result = content
            else:
                content = f.read()
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, 1)
                else:
                    result = content
        return Utils.limit_strings(result, n=limit) if limit != -1 else result
    def summarize_file_content(
        self,
        code_parser: CodeParseUtil,
        file_path: str,
        summary_type: str = "overview",
        max_lines: int = None,
        specific_aspect: str = None
    ) -> str:
        """
        Use LLM to intelligently summarize file content.
        
        Arguments:
            file_path: Path to the file to summarize
            summary_type: Type of summary:
                - "overview": General file overview
                - "structural": Code structure (classes, functions)
                - "logic": Main logic and algorithms
                - "dependencies": Imports and dependencies
                - "custom": Custom aspect specified in specific_aspect
            max_lines: Maximum lines to analyze (None for whole file)
            specific_aspect: Custom aspect to focus on for "custom" type
                
        Returns:
            LLM-generated summary of the file
        """
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        
        try:
            # Read file content (with optional line limit)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                    total_lines = max_lines
                else:
                    content = f.read()
                    # Count lines
                    f.seek(0)
                    total_lines = sum(1 for _ in f)
            
            # Get file stats
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Use CodeParseUtil for language detection
            language = code_parser.detect_language(content[:5000], file_path=file_path)
            
            # Prepare context for LLM
            context = {
                'file_path': file_path,
                'file_size': file_size,
                'total_lines': total_lines,
                'file_extension': file_ext,
                'detected_language': language or 'unknown',
                'summary_type': summary_type,
                'specific_aspect': specific_aspect
            }
            
            # Call LLM for summarization
            return self._llm_summarize_file(content, context)
            
        except Exception as e:
            return f"Error summarizing file '{file_path}': {str(e)}"
    
    def _llm_summarize_file(self, content: str, context: dict) -> str:
        """Use LLM to generate intelligent file summary."""
        
        # Determine how much content to send to LLM
        max_content_chars = 30000  # Conservative limit
        if len(content) > max_content_chars:
            # Sample beginning, middle, and end
            sample_size = max_content_chars // 3
            sample = (
                content[:sample_size] + 
                "\n\n...[middle content omitted for brevity]...\n\n" +
                content[len(content)//2 - sample_size//2:len(content)//2 + sample_size//2] +
                "\n\n...[end of file omitted]...\n\n" +
                content[-sample_size:]
            )
            content_preview = sample
            content_truncated = True
        else:
            content_preview = content
            content_truncated = False
        
        # Build prompt based on summary type
        prompts = {
            "overview": textwrap.dedent("""
                Analyze this file and provide a comprehensive overview.
                Include:
                1. File type and likely purpose
                2. Overall structure and organization
                3. Key components or sections
                4. Notable patterns or conventions used
                5. Any potential issues or unusual aspects
                
                Be concise but informative. Focus on what a developer needs to know
                to understand this file quickly.
            """),
            "structural": textwrap.dedent("""
                Analyze the structure of this code file.
                Identify and list:
                1. Main classes, functions, or modules
                2. Their relationships and hierarchy
                3. Entry points or main execution flow
                4. Key data structures or interfaces
                5. Important constants or configurations
                
                Provide line numbers or approximate locations when relevant.
            """),
            "logic": textwrap.dedent("""
                Analyze the main logic and algorithms in this file.
                Focus on:
                1. Core algorithms or business logic
                2. Data flow and transformations
                3. Control flow patterns
                4. Key computations or processing steps
                5. Error handling and edge cases
                
                Explain what the code DOES, not just its structure.
            """),
            "dependencies": textwrap.dedent("""
                Analyze dependencies and external relationships:
                1. Imports and external modules/libraries used
                2. Dependencies between internal components
                3. External APIs or services called
                4. Configuration or environment dependencies
                5. Build or runtime requirements
            """),
            "custom": textwrap.dedent("""
                Analyze this file with focus on: {specific_aspect}
                
                Provide detailed insights about this specific aspect.
                Include relevant code snippets or patterns.
            """)
        }
        
        # Get the appropriate prompt
        prompt_template = prompts.get(context['summary_type'], prompts["overview"])
        if context['summary_type'] == 'custom' and context.get('specific_aspect'):
            prompt_template = prompt_template.format(specific_aspect=context['specific_aspect'])
        
        # Build the full prompt
        system_prompt = textwrap.dedent(f"""
            You are an expert code analyst. Analyze the following file and provide 
            a helpful summary for a developer who needs to understand it quickly.
            
            File Context:
            - Path: {context['file_path']}
            - Size: {context['file_size']:,} bytes
            - Lines: {context['total_lines']:,}
            - Extension: {context['file_extension']}
            - Language: {context['detected_language']}
            {f"- Note: Content truncated to {max_content_chars:,} chars for analysis" if content_truncated else ""}
            
            Analysis Focus: {context['summary_type']}
            
            {prompt_template}
            
            Format your response clearly with sections and bullet points as appropriate.
            Be specific and reference actual code patterns when possible.
        """)
        
        user_prompt = textwrap.dedent(f"""
            Here is the file content:
            
            ```{context['detected_language'] or 'text'}
            {content_preview}
            ```
            
            Please provide your analysis.
        """)
        
        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use existing network infrastructure
            result, _ = EnhancedNetwork.make_request(
                messages=messages,
                model=DEEPSEEK_MODEL_NAME,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Add metadata to the summary
            summary = f"""📁 FILE: {context['file_path']}
📊 Metadata: {context['file_size']:,} bytes, {context['total_lines']:,} lines, {context['detected_language']}
{'⚠️  Note: Analyzed truncated content' if content_truncated else ''}
{'='*60}
{result}"""
            
            return summary
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            # Fallback to simple analysis
            return self._fallback_summary(content, context)
    
    def _fallback_summary(self, content: str, context: dict) -> str:
        """Fallback summary when LLM fails."""
        lines = content.split('\n')
        
        summary_lines = [
            f"📁 File: {context['file_path']}",
            f"📊 Size: {context['file_size']:,} bytes, Lines: {context['total_lines']:,}",
            f"🌐 Language: {context['detected_language'] or 'unknown'}",
            "=" * 60
        ]
        
        # Simple heuristics
        if context['total_lines'] > 1000:
            summary_lines.append("⚠️  Large file - consider examining specific sections")
        
        # Show first and last few lines
        if lines:
            summary_lines.append("\n📝 First 10 lines:")
            for i, line in enumerate(lines[:10]):
                if line.strip():
                    summary_lines.append(f"{i+1:4}: {line.rstrip()}")
            
            if len(lines) > 20:
                summary_lines.append("\n📝 Last 10 lines:")
                for i, line in enumerate(lines[-10:], start=len(lines)-9):
                    if line.strip():
                        summary_lines.append(f"{i:4}: {line.rstrip()}")
        
        return "\n".join(summary_lines)
    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
        
        ignore = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".tox",
            ".venv",
            "venv",
            ".eggs"
        }
        def tree(path: str, prefix: str = "", depth: int = 0, current_max_depth: int = 0) -> list[str]:
            if depth > current_max_depth:
                return []
            try:
                items = sorted(os.listdir(path))
            except (PermissionError, OSError) as e:
                return [f"{prefix}[Error reading directory: {str(e)}]"]
            dirs = [
                i for i in items
                if os.path.isdir(os.path.join(path, i))
                and not i.startswith(".")
                and i not in ignore
                and not i.endswith(".egg-info")
            ]
            files = [
                i for i in items
                if os.path.isfile(os.path.join(path, i))
                and not i.startswith(".")
            ]
            lines: list[str] = []
            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1, current_max_depth))
            for idx, f in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{f}")
            return lines
        def count_tokens(text: str) -> int:
            try:
                if 'Utils' in globals() and hasattr(Utils, 'count_tokens'):
                    return Utils.count_tokens(text)
            except (NameError, AttributeError):
                pass
            return len(text) // 4
        MAX_TOKENS = 3000
        current_depth = max_depth
        
        while current_depth >= 0:
            entries = tree(directory_path, "", 0, current_depth)
            result = f"Directory structure (depth={current_depth}):\n{directory_path}/\n" + "\n".join(entries)
            
            token_count = count_tokens(result)
            
            if token_count <= MAX_TOKENS:
                if current_depth < max_depth:
                    result += f"\n\n[Note: Requested depth {max_depth} exceeded token limit. Showing depth {current_depth} instead ({token_count} tokens).]"
                return result
            
            if current_depth == 0:
                result += f"\n\n[Warning: Result exceeds token limit ({token_count} tokens > {MAX_TOKENS} tokens). Consider using a more specific directory_path.]"
                return result
            
            current_depth -= 1
        
        entries = tree(directory_path, "", 0, 0)
        result = f"Directory structure (depth=0):\n{directory_path}/\n" + "\n".join(entries)
        return result
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
    def apply_code_edit(
        self,
        file_path: str,
        search: str,
        replace: str,
        search_start_line: int = None,
        search_end_line: int = None
    ) -> str:
        def add_context_to_similar_match(
            original_content: str, formatted_match: str, context_lines: int = 2
        ) -> str:
            """
            Add context lines around a similar match for better understanding.
            """
            lines = original_content.split("\n")
            match_lines = formatted_match.split("\n")
            if len(match_lines) < 2:
                return formatted_match
            # Remove the first line which is description for matching
            actual_content_lines = match_lines[1:]
            actual_content = "\n".join(actual_content_lines)
            best_match_start = -1
            best_similarity = 0.0
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i : i + len(actual_content_lines)]
                candidate_content = "\n".join(candidate_lines)
                similarity = difflib.SequenceMatcher(
                    None, actual_content.strip(), candidate_content.strip()
                ).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            if best_match_start == -1:
                return formatted_match
            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = (
                    ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                )
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            description = (
                match_lines[0]
                if match_lines
                else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            )
            return f"{description}\n" + "\n".join(context_lines_list)
        def find_most_similar_content(
            original_content: str, search_string: str, max_results: int = 3
        ) -> list[tuple[float, str]]:
            """
            Find the most similar content chunks to the search string.
            """
            lines = original_content.split("\n")
            chunks = []
            search_lines = [line for line in search_string.split("\n") if line.strip()]
            target_chunk_size = max(3, len(search_lines)) if search_lines else 3
            # Individual non-empty line chunks, with descriptions
            for i, line in enumerate(lines):
                if line.strip():
                    chunks.append((f"Line {i+1}:", line.strip()))
            # Sequential chunk slices
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i : i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}:", chunk_content))
            similarities = []
            s_string = search_string.strip()
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(
                    None, s_string, chunk_content
                ).ratio()
                if ratio > 0.3:
                    similarities.append((ratio, f"{chunk_desc}\n{chunk_content}"))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return similarities[:max_results]
        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        try:
            # Get the full content to work with
            full_content = self.file_ops.get_file_content(file_path, limit=-1)
        except Exception as e:
            return f"Error retrieving file content for '{file_path}': {str(e)}"
        lines = full_content.split("\n")
        def _show_context(new_content:str, replace:str) -> str:
            """Show context of the replaced lines in the file."""
            lines = new_content.split("\n")
            replace_pos = new_content.find(replace)
            if replace_pos == -1:
                return None
            # Find which line the replacement starts on
            chars_so_far = 0
            replace_line_start = None
            for i, line in enumerate(lines):
                if chars_so_far + len(line) >= replace_pos:
                    replace_line_start = i
                    break
                chars_so_far += len(line) + 1  # +1 for newline
            if replace_line_start is None:
                return None
            replace_lines_count = replace.count("\n") + 1
            replace_line_end = replace_line_start + replace_lines_count - 1
            start_line = max(0, replace_line_start - 5)
            end_line = min(len(lines), replace_line_end + 5 + 1)
            context_lines = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if replace_line_start <= i <= replace_line_end else "    "
                context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
            return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n" + "\n".join(context_lines)
        # Handle line range restriction
        use_range = (search_start_line is not None or search_end_line is not None)
        if use_range:
            start_idx = max(0, (search_start_line if search_start_line is not None else 1) - 1)
            end_idx = min(len(lines), search_end_line if search_end_line is not None else len(lines))
            range_lines = lines[start_idx:end_idx]
            range_content = "\n".join(range_lines)
            occurrences_in_range = range_content.count(search)
            if occurrences_in_range == 0:
                # Try to provide similar content in the range
                similar_matches = find_most_similar_content(range_content, search, 1)
                error_msg = f"Error: search string not found in specified line range ({start_idx+1}-{end_idx}) of file {file_path}."
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found in specified range:"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        context = add_context_to_similar_match(range_content, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{context}"
                else:
                    error_msg += " No similar content found in specified range."
                return error_msg
            elif occurrences_in_range == 1:
                # Replace and reconstruct the whole file
                new_range_content = range_content.replace(search, replace, 1)
                new_lines = lines[:start_idx] + new_range_content.split("\n") + lines[end_idx:]
                new_content = "\n".join(new_lines)
                try:
                    self.file_ops.save(file_path, new_content)
                except Exception as e:
                    return f"Error: could not save file '{file_path}'. {str(e)}"
                context = _show_context(new_content, replace)
                return context if context else "ok, code edit applied successfully"
            else:
                return (
                    f"Error: search string found {occurrences_in_range} times in specified line range "
                    f"({start_idx+1}-{end_idx}) of file '{file_path}'.\n"
                    f"Please reformulate your search and replace to apply only one change."
                )
        else:
            count = full_content.count(search)
            if count == 0:
                similar_matches = find_most_similar_content(full_content, search, 1)
                error_msg = f"Error: search string not found in file {file_path}."
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        context = add_context_to_similar_match(full_content, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{context}"
                else:
                    error_msg += " No similar content found."
                return error_msg
            elif count == 1:
                new_content = full_content.replace(search, replace, 1)
                try:
                    self.file_ops.save(file_path, new_content)
                except Exception as e:
                    return f"Error: could not save file '{file_path}'. {str(e)}"
                context = _show_context(new_content, replace)
                return context if context else "ok, code edit applied successfully"
            else:
                return (
                    f"Error: search string found {count} times in file '{file_path}'.\n"
                    "Please reformulate your search and replace to apply only one change."
                )
class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}
    def __init__(self, **kwargs):
        pass
    def _save(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as file:
            file.write(content)
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
            self.tool_invocations[fn.__name__] = (
                self.tool_invocations.get(fn.__name__, 0) + 1
            )
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {
                        j: 0 for j in self.Error.ErrorType.__members__
                    }
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message
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
        self.repo_dir = "."
        self.saved_observation_counter = 0
        if should_review:
            self.is_reviewed = False
            self.file_by_file_reviewed = False
        else:
            self.is_reviewed = True
            self.file_by_file_reviewed = True
        os.makedirs(self.observation_dir, exist_ok=True)
        
        self.file_system_manager = FileSystemManager()
        self.search_manager = SearchManager()
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
        self.code_parser = CodeParseUtil()
        self.TOOL_LIST = {}
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if (
                        available_tools is not None and name not in available_tools
                    ):
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {
            k: {j: 0 for j in self.Error.ErrorType.__members__}
            for k in self.TOOL_LIST.keys()
        }
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
    @EnhancedToolManager.tool
    def apply_code_edit(
        self, 
        file_path: str, 
        search: str, 
        replace: str,
        search_start_line: int = None,
        search_end_line: int = None
    ) -> str:
        """
        Performs targeted text replacement within source files.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
            search_start_line: optional start line number to limit search to a range (1-indexed)
            search_end_line: optional end line number to limit search to a range (1-indexed)
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(
            file_path=file_path, 
            search=search, 
            replace=replace,
            search_start_line=search_start_line,
            search_end_line=search_end_line
        )
    @EnhancedToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            directory_path: the directory path to list (default: ".")
            max_depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(
            directory_path=directory_path, max_depth=max_depth
        )
    @EnhancedToolManager.tool
    def run_code(self, file_path: str, run_command: List[str]) -> str:
        """
        Runs code. Executes the code with run_command.
        Arguments:
            file_path: Code file path to run, relative to current directory (e.g., "file.py", "file.js")
            run_command: Command to execute the code (e.g., ["python", "file.py"], ["node", "file.js"])
        """
        # Load file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error: Could not read file '{file_path}': {e}"
        # Inject version compatibility fix if needed
        version_fix_needed = file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz"))
        fix_missing = version_fix_needed and VERSION_COMPATIBILITY_FIX not in content
        original_content = content if fix_missing else None
        if fix_missing:
            content = f"{VERSION_COMPATIBILITY_FIX}\n\n{content}"
        # Save (possibly modified) file for execution
        try:
            self.file_ops.save(file_path, content)
            self.generated_test_files.append(file_path)
        except Exception as e:
            return f"Error: Failed to save file '{file_path}': {e}"
        # Run the code
        try:
            logger.info(f"Running command in run_code: {run_command}")
            result = subprocess.run(run_command, capture_output=True, text=True, timeout=60)
            output = result.stdout.strip()
            error = result.stderr.strip()
            if result.returncode != 0:
                exec_result = f"Error running code (exit code {result.returncode}): {error or output}"
            else:
                exec_result = f"{output}\n" if output else "Execution succeeded, no output.\n"
        except Exception as e:
            exec_result = f"Error executing command: {e}"
        # Restore original if modified
        if fix_missing and original_content is not None:
            try:
                self.file_ops.save(file_path, original_content)
            except Exception as e:
                exec_result = (exec_result or "") + f"\nWarning: Failed to restore original file after execution: {e}"
        return exec_result
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
    @EnhancedToolManager.tool
    def summarize_file(
        self,
        file_path: str,
        summary_type: str = "auto",
        focus: str = None,
        max_content_lines: int = 500
    ) -> str:
        """
        Intelligently summarize files using LLM analysis.
        
        Arguments:
            file_path: Path to the file
            summary_type: Type of summary (auto, overview, structure, logic, dependencies, custom)
            focus: For custom type, specify what to focus on (e.g., "security", "performance")
            max_content_lines: Max lines to analyze (for very large files)
            
        Returns:
            LLM-generated file summary
        """        
        if summary_type == "auto":
            summary_type = "overview"
        elif summary_type == "structural":
            summary_type = "structure"
        elif summary_type == "custom" and focus:
            summary_type = "custom"
        
        return self.file_system_manager.summarize_file_content(
            code_parser=self.code_parser,
            file_path=file_path,
            summary_type=summary_type,
            max_lines=max_content_lines,
            specific_aspect=focus
        )
    
    @EnhancedToolManager.tool  
    def analyze_code_file(
        self,
        file_path: str,
        analysis_type: str = "comprehensive"
    ) -> str:
        """
        Advanced code file analysis using CodeParseUtil's LLM integration.
        
        Arguments:
            file_path: Path to the file
            analysis_type: comprehensive, structure_only, dependencies, logic
            
        Returns:
            Detailed LLM analysis of the code file
        """
        return self.code_parser.analyze_file_structure(file_path, analysis_type)
    @EnhancedToolManager.tool
    def create_new_file(
        self,
        file_path: str,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """
        Creates a new file with the specified content.
        Arguments:
            file_path: Path where the new file should be created.
            content: The content to write into the file.
            overwrite: If True, will overwrite the file if it exists. If False and file exists, returns an error.
        Returns:
            Status message indicating success or error.
        """
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Set overwrite=True to overwrite."
        try:
            # If file_path is just a filename (no directories), os.path.dirname returns ""
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            if hasattr(self, "file_ops") and hasattr(self.file_ops, "new_files_created"):
                self.file_ops.new_files_created.append(file_path)
            return f"File '{file_path}' created successfully."
        except Exception as e:
            return f"Error creating file '{file_path}': {e}"
    def get_final_git_patch(self) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            observation_dir = getattr(self, "observation_dir", ".observation")
            if os.path.exists(observation_dir):
                try:
                    for root, dirs, files in os.walk(observation_dir):
                        for file in files:
                            file_path = os.path.relpath(os.path.join(root, file))
                            exclude.add(file_path)
                except Exception:
                    pass
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
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    def _save_large_observation(self, observation: str, tool_name: str) -> str:
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
    def get_function_body(self, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve (supports both qualified names like "ClassName.method_name" and simple names like "method_name").
        Returns:
            The complete function body including decorators, or empty string if function not found.
        """
        return self.code_parser.get_function_body(file_path, function_name)
    def _can_parse_structurally(self, file_path: str) -> bool:
        """Helper to check if we can parse a file structurally."""
        functions = self.code_parser.extract_all_functions(file_path)
        return bool(functions)
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
            "list_directory_structure",
            "create_new_file",
            "search_in_all_files_content",
            "get_file_content",
            "summarize_file",
            "analyze_file_structure",
            "get_function_body",
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
        [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME],
        log_prefix="MAIN_WORKFLOW",
    )
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
                messages=[{
                    "role": "user",
                    "content": check_all_tests_passed_prompt.format(output=output),
                }],
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
def llm_select_run_command_for_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    retry = 0
    while retry < 10:
        try:
            messages = [{
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
            }]
            raw_text, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            json_result = json.loads(
                raw_text.replace("```json", "").replace("```", "").strip()
            )
            return json_result.get("command")
        except Exception as e:
            time.sleep(1)
            retry += 1
    return []
def extract_core_concepts_for_search(problem_statement: str) -> dict:
    EXTRACT_CONCEPTS_PROMPT = textwrap.dedent("""
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
    """)
    retry = 0
    selected_model = GLM_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": EXTRACT_CONCEPTS_PROMPT},
                {"role": "user", "content": f"Problem Statement:\n{problem_statement}"},
            ]
            response, _ = EnhancedNetwork.make_request(
                messages, model=selected_model, temperature=0.0
            )
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL
            )
            if json_match:
                try:
                    concepts = json.loads(json_match.group(0))
                    if isinstance(concepts, dict) and "search_terms" in concepts:
                        return concepts
                except json.JSONDecodeError:
                    code_block_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
                    )
                    if code_block_match:
                        try:
                            concepts = json.loads(code_block_match.group(1))
                            if (
                                isinstance(concepts, dict)
                                and "search_terms" in concepts
                            ):
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
def generate_initial_solution(
    problem_statement: str, initial_structure: str, temperature: float = 0.7
) -> str:
    print("[GENERATE_INITIAL_SOLUTION] Starting solution generation")
    concepts = extract_core_concepts_for_search(problem_statement)
    edge_case_guidance = ""
    if concepts.get("common_edge_cases"):
        edge_case_guidance = (
            f"\n\n**Common Edge Cases to Consider (based on similar problems):**\n"
        )
        for i, case in enumerate(concepts.get("common_edge_cases", []), 1):
            edge_case_guidance += f"{i}. {case}\n"
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
        9. **IMPORTANT**: Design your code to robustly handle input in all possible formats-implement logic to detect and preprocess various valid input types so the program works regardless of input format.
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
    INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent("""
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
    """)
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
            solution = clean_code_response(loop_check_response)
            return solution
        except Exception as e:
            retry += 1
            time.sleep(1)
    if retry >= 10:
        return ""
    return ""
def check_problem_type(problem_statement):  # type: ignore
    enhancement = enhance_problem_statement(problem_statement)
    results = [get_problem_type(problem_statement, enhancement) for _ in range(3)]
    create_count = results.count(PROBLEM_TYPE_CREATE)
    fix_count = results.count(PROBLEM_TYPE_FIX)
    if create_count > fix_count:
        return PROBLEM_TYPE_CREATE, enhancement
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
    
    print(f"Final patch: {result}")
    return result if result else ""
def process_create_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id, _current_tool_manager 
    def get_files_to_modify(problem_statement: str) -> str:
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
                [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME],
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
    total_timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 150
    problem_statement = input_dict.get("problem_statement", "")
    tool_manager = EnhancedToolManager()
    _current_tool_manager = tool_manager
    initial_structure = get_files_to_modify(problem_statement)
    s_time = time.time()
    initial_solution = None
    BASIC_APPROACH_RETRY = 20
    for attempt in range(BASIC_APPROACH_RETRY):
        os.system("git reset --hard")
        initial_solution, _ = basic_approach(
            initial_structure, problem_statement
        )
        print(f"Initial solution in process_create_task for {attempt}: {initial_solution}")
        if initial_solution is not None:
            break
        time.sleep(1)
    if initial_solution is not None:
        os.system("git reset --hard")
        extract_and_write_files(initial_solution)
        patch = tool_manager.get_final_git_patch()
        return patch
    
    elapsed_time = time.time() - s_time
    return fix_task_solve_workflow(problem_statement, timeout=total_timeout - elapsed_time - 60, run_id_1=run_id, enhancement=enhancement, should_review=False)
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
def clean_code_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response
def process_fix_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    logs = []
    patch_text = ""
    try:
        patch_text = fix_task_solve_workflow(
            problem_text, timeout=timeout - 60, run_id_1=run_id, enhancement=enhancement, should_review=True
        )
        os.system("git reset --hard")
    except Exception as e:
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logs.append(error_info)
    return patch_text
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
    GENERATE_TESTCASES_PROMPT = textwrap.dedent("""
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
    """)
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
            if not testcases or not testcases.strip():
                retry += 1
                continue
            lines = testcases.split("\n")
            if not lines or len(lines) == 0:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_response})
                test_generation_messages.append({
                    "role": "user",
                    "content": f"Include file name in the response. example:\n```python\ntest_a.py\n{{content}}\n\ntest_b.py\n{{content}}\n```\n```javascript\ntest_a.js\n{{content}}\n\ntest_b.js\n{{content}}\n```",
                })
                continue
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_single_testset: {e}")
            time.sleep(1)
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
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.error(f"ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)
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
            logger.warning(
                f"[{log_prefix}] Usage exceeded limit: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
            )
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
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        if cot.is_thought_repeated():
            logger.info(
                f"[TEMPERATURE] Thought repeated {cot.repeated_thoughts} times"
            )
            last_thought = cot.thoughts[-1]
            messages.append({
                "role": "user",
                "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                    previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                ),
            })
            temperature = 0.5
            if cot.repeated_thoughts >= 2:
                model_idx = (cot.repeated_thoughts - 2) % len(models)
                selected_model = models[model_idx]
        else:
            temperature = 0.0
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
            logger.error(f"[{log_prefix}] Inference error: {e}")
            is_timeout_error = "Agent execution timeout" in str(e)
            if is_timeout_error:
                cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
                return tool_manager.get_final_git_patch()
        tool_names_list = (next_tool_name if isinstance(next_tool_name, list) else [next_tool_name])
        tool_args_list = (next_tool_args if isinstance(next_tool_args, list) else [next_tool_args])
        
        logger.info(f"[{log_prefix}] Used model: {selected_model}, Inference time: {inference_duration:.2f}s")
        logger.info(f"[{log_prefix}] Next thought: {next_thought}\n\n")
        logger.info(f"[{log_prefix}] About to execute {len(tool_names_list)} tool call(s): {tool_names_list}\n")
        logger.info(f"[{log_prefix}] Tool arguments: {json.dumps(tool_args_list, indent=4)}\n\n")
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
                error_traceback = traceback.format_exc()
                error_msg = (
                    f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                )
                all_observations.append(error_msg)
                all_successful = False
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
        if finish_tool_name in tool_names_list:
            if finish_tool_name == "finish_find_files_to_fix":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        return obs
            elif finish_tool_name == "finish":
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        if obs != "finish":
                            break
                        return tool_manager.get_final_git_patch()
    return tool_manager.get_final_git_patch()
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
                return json.loads(
                    result.replace("```json", "").replace("```", "").strip()
                )
            except Exception as e:
                retry += 1
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
                file_content = (
                    file_content.rstrip() + "\n"
                    if file_content.strip()
                    else file_content
                )
                f.write(file_content)
            created_files.append(path)
    filename_set = set(file_names)
    for fname in file_names:
        filename_set.add(fname.split("/")[-1])
    for line in initial_solution.split("\n"):
        stripped = line.strip()
        if stripped in filename_set:
            write_file()
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