from __future__ import annotations
import json
import os
import requests
import subprocess
import sys
import textwrap
import time
import traceback
import re
import inspect
import random
import logging
import threading
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional
from json import JSONDecodeError
from enum import Enum
from uuid import uuid4
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
run_id = agent_start_time = _current_tool_manager = None
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy") 
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200")) 
PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX = "CREATE", "FIX"
MAX_FIX_TASK_STEPS, LATEST_OBSERVATIONS_TO_KEEP = 200, 20
SUMMARIZE_BATCH_SIZE, MAX_SUMMARY_RANGES = 6, 6
GLM_MODEL_NAME, GLM_OLD_MODEL_NAME = "zai-org/GLM-4.6-FP8", "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME = "moonshotai/Kimi-K2-Instruct", "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [model for model in [GLM_MODEL_NAME, GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]
FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
Now let's start.
```
{problem_statement}
```
""")
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
FORMAT_PROMPT = textwrap.dedent("""
**CRITICAL: You can make MULTIPLE tool calls in ONE response for efficiency!**
## Response Formats
### Format 1: Multiple Tool Calls (RECOMMENDED for efficiency)
next_thought: [Your detailed reasoning]
tool_call_1:
    tool_name: [exact tool name]
    tool_args: {{valid JSON}}
tool_call_2:
    tool_name: [exact tool name]
    tool_args: {{valid JSON}}
tool_call_3:
    tool_name: [exact tool name]
    tool_args: {{valid JSON}}
### Format 2: Single Tool Call (Legacy, less efficient)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {{valid JSON}}
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
    tool_args: {{"file_path": "abcd.py", "search": "old_code", "replace": "fixed_code"}}
tool_call_2:
    tool_name: run_code
    tool_args: {{"content": "test_content", "file_path": "file.js"}}
✅ **Good - Batch Multiple Searches**:
{multiple_search_guide}
❌ **Bad - One tool per response (too slow)**:
Response 1:
next_thought: Let me edit the file
next_tool_name: apply_code_edit
next_tool_args: {{"file_path": "aaa.py", ...}}
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
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.
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
13. If you find that the error while running the run_code tool due to missing dependencies, do not try to solve it as you don't have any internet access.
## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase
- Prefer using `get_file_content` to read the code you need to modify.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.
You have access to the following tools:-
{tools_docs}
Here is the problem statement:
{problem_statement}
{format_prompt}
""")
CREATE_TASK_SYSTEM_PROMPT = textwrap.dedent(
"""
    # Role
    You are a senior bug-fix engineer working on an open-source repository.
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
   - If problem is complex: Use `decompose_complex_problem` to break into sub-problems (steps 1-3)
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
   - If uncertain about change: Use `assess_change_confidence` to evaluate before implementing
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
   - Use `select_thinking_mode` to calibrate approach (especially at step 15, 20+)
    ## 8. Final Reflection and Additional Testing
    - Reflect carefully on the original intent of the user and the problem statement.
    - Think about potential edge cases or scenarios that may not be covered by existing tests.
    - Write additional tests that would need to pass to fully validate the correctness of your solution.
    - Run these new tests and ensure they all pass.
    - Be aware that there are additional hidden tests that must also pass for the solution to be successful.
    - Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.
    ## At Decision Points
   - Use `select_thinking_mode` after investigation, before major changes, or when stuck
   - Helps optimize step efficiency based on remaining budget
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
    {search_guide}
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
    - If stuck in a pattern, CHANGE your approach (different files, different strategy, or rollback)
    - Be specific about what you'll learn from your next action that you don't already know
    **Critical**: These checkpoints exist to prevent wasted effort. Take them seriously and be willing to pivot when not making progress.
    - **MANDATORY**: At meta-cognitive checkpoints (step 15, 20+), use `select_thinking_mode` to calibrate your approach
      - Especially critical at step 20+ to optimize remaining step budget
      - Example: `select_thinking_mode(current_situation="<situation>", step_number=22, recent_outcomes="<outcome>")`
    # Cognitive Tools for Knowledge Persistence
    You have access to powerful cognitive tools designed to preserve knowledge across rollbacks and prevent retry loops:
# Adaptive Reasoning Tools (MANDATORY USAGE)
You have access to three adaptive tools that MUST be used at specific points in your workflow:
## 1. decompose_complex_problem - Problem Breakdown (MANDATORY FOR COMPLEX PROBLEMS)
**MANDATORY when**: Problem statement >300 words OR involves multiple files/components
**When to use**: Early (steps 1-3) when problem seems complex or multi-faceted
**Purpose**: Break complex problem into 3-5 ordered sub-problems with clear roadmap
**Benefits**: Makes overwhelming problems manageable; provides clear checkpoints; identifies dependencies
**You MUST use this tool if**: Problem statement >300 words, multiple files/components, architecture changes
**Example**: Multi-file refactor, architecture changes, complex bugs affecting multiple systems
## 2. assess_change_confidence - Risk Assessment (MANDATORY FOR NON-TRIVIAL CHANGES)
**MANDATORY when**: About to implement any non-trivial code change (not just formatting/comments)
**When to use**: Before implementing uncertain or risky code changes
**Purpose**: Get objective assessment of proposed change (correctness, risk, edge cases)
**Benefits**: Prevents wasted steps on low-confidence approaches; identifies gaps in reasoning
**You MUST use this tool for**: Logic changes, algorithm modifications, touching core code, multi-file impact
**Skip ONLY for**: Pure formatting, comments, simple variable renames
**Example**: Major logic changes, algorithm modifications, critical path code
## 3. select_thinking_mode - Efficiency Optimization (MANDATORY AT CHECKPOINTS)
**MANDATORY when**: At meta-cognitive checkpoints (step 15, 20+) to calibrate approach
**When to use**: At decision points to calibrate thinking depth
**Purpose**: Adapt thinking depth (FAST/NORMAL/DEEP) based on situation and step budget
**Benefits**: Saves steps on simple tasks; allocates deep thinking to complex decisions; step budget awareness
**Modes**:
  - FAST: Simple reads, searches, trivial edits (when steps running low)
  - NORMAL: Standard code changes, testing, hypothesis formation
  - DEEP: Complex decisions, stuck situations, critical bugs (use sparingly)
**You MUST use this tool**: At step 15, step 20+, or when deciding how to proceed after investigation
**Example**: At step 22, use to assess if you need FAST mode for remaining steps
**ENFORCEMENT - When You MUST Use These Tools**:
1. **decompose_complex_problem**: REQUIRED if problem >300 words or multi-file. Use in steps 1-3.
2. **assess_change_confidence**: REQUIRED before ANY non-trivial code change. Do not skip.
3. **select_thinking_mode**: REQUIRED at meta-cognitive checkpoints (step 15, 20+). Optimizes remaining budget.
These are NOT optional aids - you MUST use them when their trigger conditions are met.
# CREATE Task Workflow (MANDATORY FOR NEW FEATURE DEVELOPMENT)
**If you are building NEW functionality (not fixing existing bugs), you MUST follow this CREATE workflow:**
## Step 1: Design API Contract (MANDATORY - Steps 1-3)
**You MUST use `design_api_contract` before writing ANY implementation code**
- Call early: After understanding requirements, BEFORE coding
- Purpose: Define function signatures, class structures, data formats, edge cases
- Example: `design_api_contract(feature_description="<description>", requirements="<requirement01>,<requirement02>, <requirement03>")`
- Output guides entire implementation - prevents architectural mistakes
## Step 2: Validate Design Completeness (MANDATORY - Step 3-5)
**You MUST use `validate_design_completeness` after initial design**
- Call after: You have an API design or implementation plan
- Purpose: Verify design covers ALL requirements (catches missing features early)
- Example: `validate_design_completeness(proposed_design="<your API design>", original_requirements="<full problem statement>")`
- If validation shows gaps: STOP and revise design before implementing
## Step 3: Plan Incremental Build (MANDATORY - Step 4-6)
**You MUST use `plan_incremental_build` before starting implementation**
- Call after: Design is validated as complete
- Purpose: Break implementation into 3-5 testable phases (prevents big-bang failures)
- Example: `plan_incremental_build(api_design="<your design>", step_budget_remaining=25)`
- Follow the phases in order - test each phase before moving to next
## Step 4: Implement Phase by Phase
- Implement Phase 1 (simplest/core functionality first)
- Test Phase 1 thoroughly before proceeding
- Implement Phase 2, test it
- Continue until all phases complete
- Use `assess_change_confidence` before major code changes (already mandatory)
## Step 5: Final Verification
- Verify ALL requirements covered (use `validate_design_completeness` again if needed)
- Test edge cases thoroughly
- Call `finish` when complete
**CRITICAL ENFORCEMENT:**
- You MUST call `design_api_contract` in steps 1-3 for CREATE tasks
- You MUST call `validate_design_completeness` after design (step 3-5)
- You MUST call `plan_incremental_build` before implementation (step 4-6)
- Skipping these tools will lead to failed CREATE tasks
**Why this matters**: CREATE tasks fail when you:
1. Start coding without clear design → architectural mistakes
2. Miss requirements → hidden tests fail
3. Build everything at once → hard to debug, wasted steps
    ## Hypothesis Tracking
    **Purpose**: Track theories about the bug to avoid retesting rejected hypotheses, especially after rollbacks.
    **Tools**:
    - **create_hypothesis(description, evidence)**: Log a theory when you form one
    - Use when: You have a theory but need to investigate further
    - Example: "Function fails on edge case" based on "test_edge_case fails with unexpected error"
    - **test_hypothesis(hypothesis_id, outcome, findings)**: Record test results
    - Use when: You've tested a hypothesis through code changes or investigation
    - Outcomes: "confirmed", "rejected", "inconclusive"
    - Example: After fixing issue, mark hypothesis #1 as "confirmed" with "fix resolves test failure"
    - **list_hypotheses()**: Review all hypotheses and their status
    - Use when: During meta-cognitive checkpoints, after rollbacks, or when stuck
    - Shows: Which theories confirmed/rejected/untested
    ## Strategy Memory
    **Purpose**: Remember what approaches you've tried, even after rolling back changes.
    **Tools**:
    - **log_strategy(approach, reasoning)**: Record planned approach BEFORE implementing
    - Use when: About to make significant code changes
    - Example: "Update function in <file> at line <N>" because "<reference relevant hypothesis for reasoning>"
    - **mark_strategy_outcome(strategy_id, success, reason)**: Record whether it worked
    - Use when: After testing the strategy (tests pass/fail)
    - Example: Mark strategy #1 as failed: "Tests passed but broke edge case in rare input scenario"
    - **list_attempted_strategies()**: Review all strategies and outcomes
    - Use when: After rollbacks (to see what doesn't work), during reflection, or when choosing next approach
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
    5. **After Rollbacks**:
    - IMMEDIATELY use `list_attempted_strategies` to see what you tried
    - IMMEDIATELY use `list_hypotheses` to see what you learned
    - This prevents retry loops since file state resets but cognitive state persists
    **Critical**: These tools create institutional memory that survives rollbacks. Use them consistently to avoid wasting effort.
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
                timeout=45,
            )
            if result.returncode > 1: 
                error_msg = result.stderr.strip() or "Unknown error" 
                return f"Error: Grep command failed with return code {result.returncode}: {error_msg}" 
            output = result.stdout 
        except Exception as e: 
            return f"Error: Failed to execute grep command: {e}" 
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
            pattern = re.escape(term)
            match_lines = [i for i, line in enumerate(lines) if pattern in line]
            if not match_lines:
                return f"'{term}' not found in file '{filepath}'"
            context = 20
            seen = set()
            chunks = [] 
            for ln in match_lines: 
                start = max(0, ln - context)
                end = min(len(lines), ln + context + 1)
                rkey = (start, end)
                if rkey in seen:
                    continue
                seen.add(rkey)
                chunk = lines[start:end]
                chunks.append(f"(lines {start+1}-{end}):\n" + "\n".join(chunk))
            result = "\n\n".join(chunks)
            return Utils.limit_strings(result, n=max_output_lines)
        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output
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
    def inference(cls, messages: list[dict], model: str, run_id: str = str(uuid4()), temperature: float = 0.0) -> dict:
        models = [model] if isinstance(model, str) else model 
        cleaned_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"system", "user", "assistant", "tool"}
            and (m.get("role") != "assistant" or m.get("content", "").strip())
        ]
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")
        result = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
        return result
    @classmethod
    def make_request(cls, messages: list, model: str, attempt: int = 0, temperature: float = 0.0, timeout: int = 150, tool_mode: str = "none", tool_docs: list = []) -> str:
        global run_id, DEFAULT_TIMEOUT, agent_start_time
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference" 
        timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT))) - 120 
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
                resp = requests.post(url, json=request_data, timeout=(30, timeout), headers=headers)
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
    def is_http_response(cls, raw_text: str):
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text: 
            return False, cls.ErrorType.TIMEOUT.name 
        if "HTTP ERROR: Request failed for model" in raw_text: 
            return False, cls.ErrorType.NETWORK_ERROR.name 
        return True, None 
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
    def _request_next_action_with_retry(cls, messages: dict, models: list[str], max_retries: int = 3, base_delay: float = 1.0, temperature: float = 0.0) -> str:
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
                raw_text, _ = cls.make_request(messages, model=current_model, temperature=temperature)
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
                    EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True),
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
            except Exception:
                retry += 1
                remaining = [model for model in AGENT_MODELS if model != selected_model]
                if remaining:
                    selected_model = random.choice(remaining)
                time.sleep(1)
        try:
            response = response.replace("```json", "").strip("```")
            return json.loads(response)
        except Exception:
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
                raise JSONDecodeError("Invalid JSON", json_string, 0)  
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
class EnhancedCOT:
    def __init__(self, latest_observations_to_keep=5, summarize_batch_size=10, do_summarize = True): 
        self.thoughts: list[EnhancedCOT.Action] = [] 
        self.latest_observations_to_keep = latest_observations_to_keep 
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size 
        self.summaries: dict[tuple[int, int], str] = {} 
        self.summarized_ranges: list[tuple[int, int]] = [] 
        self.do_summarize = do_summarize
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
            return 
        oldest_unsummarized = 0
        for start, end in sorted(self.summarized_ranges):
            if start <= oldest_unsummarized < end:
                oldest_unsummarized = end
            elif start > oldest_unsummarized:
                break  
        if oldest_unsummarized >= cutoff_idx:
            return  
        summarize_start = oldest_unsummarized
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
    def add_action(
        self, action: EnhancedCOT.Action
    ) -> bool:  # don't add if thought is repeated
        self.thoughts.append(action)
        # Check if we need to summarize older messages
        # Only check when we have enough messages to potentially summarize
        total_thoughts = len(self.thoughts)
        if not self.do_summarize:
            return True
        
        if (
            total_thoughts
            >= self.latest_observations_to_keep + self.summarize_batch_size
        ):
            self._check_and_summarize_if_needed()
        return True
    def _summarize_messages_batch(self, start_idx: int, end_idx: int) -> Optional[str]:
        """Summarize a batch of messages using LLM."""
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if thought.is_deleted:
                continue
            assistant_part = f"next_thought: {thought.next_thought}\n"
            assistant_part += f"next_tool_name: {thought.next_tool_name}\n"
            assistant_part += f"next_tool_args: {thought.next_tool_args}\n"
            if isinstance(thought.observation, (list, tuple)):
                try:
                    obs_render = json.dumps(
                        list(thought.observation), ensure_ascii=False
                    )
                except Exception:
                    obs_render = str(thought.observation)
            else:
                obs_render = str(thought.observation) if thought.observation else ""
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
        conversation_text = ""
        for i, part in enumerate(conversation_parts, 1):
            conversation_text += f"\n--- Step {i} ---\n"
            conversation_text += f"Assistant: {part['assistant']}\n"
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
    def to_str(self):
        messages = []
        last_summary_range = None
        if self.do_summarize:
            if self.summarized_ranges:
                allowed_summary_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:])
            else:
                allowed_summary_ranges = set()
        for i, thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                if self.do_summarize:
                    summary = self._get_summary_for_index(i)
                    if summary:
                        found_range = False
                        for (start, end), summ in self.summaries.items():
                            if start <= i < end:
                                current_range = (start, end)
                                if current_range not in allowed_summary_ranges:
                                    found_range = True
                                    break
                                if current_range != last_summary_range:
                                    messages.append(
                                        {
                                            "role": "system",
                                            "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]",
                                        }
                                    )
                                    last_summary_range = current_range
                                found_range = True
                                break  
                        if found_range:
                            continue
                assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                
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
          
                if thought.is_error is None or i == len(self.thoughts) - 1:
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
    class Action:
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation, is_error: bool = False, raw_response: str = None, total_attempts: int = 0, inference_error_counter: dict = None, request_data: list = None):
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
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None, limit: int = 1000, add_line_numbers: bool = False, search_in_file_callback=None) -> str:
        
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
                result = add_line_numbers_to_content(content, start_idx + 1) if add_line_numbers else content
            else:
                content = f.read()
                result = add_line_numbers_to_content(content, 1) if add_line_numbers else content
        return Utils.limit_strings(result, n=limit) if limit != -1 else result
    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
        ignore = {"__pycache__", "node_modules", "venv"}
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
        self.file_system_manager = self.search_manager = None
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None, limit: int = 1000, add_line_numbers: bool = False, structural_truncation: bool = False) -> str:
        search_callback = lambda fp, st: self.search_manager.search_in_file(fp, st)
        return self.file_system_manager.get_file_content(file_path=file_path, search_start_line=search_start_line, search_end_line=search_end_line, search_term=search_term, limit=limit, add_line_numbers=add_line_numbers, search_in_file_callback=search_callback)
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
    def __init__(self, file_ops: FileOperationsUtil = None):
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
            for i, line in enumerate(lines):
                if line.strip(): 
                    chunks.append((f"Line {i+1}:", line.strip())) 
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
        use_range = (search_start_line is not None or search_end_line is not None)
        if use_range:
            start_idx = max(0, (search_start_line if search_start_line is not None else 1) - 1)
            end_idx = min(len(lines), search_end_line if search_end_line is not None else len(lines))
            range_lines = lines[start_idx:end_idx]
            range_content = "\n".join(range_lines)
            occurrences_in_range = range_content.count(search)
            if occurrences_in_range == 0:
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
    def get_tool_args_for_tool(self, tool_name: str, required_only: bool = False) -> list[str]: 
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
    def __init__(self, available_tools: Optional[list[str]] = [], runner_hint: str | None = None, runner_mode_hint: str | None = None, initial_checkpoint=None, problem_statement: str = None, should_review: bool = True):
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
        
        # Initialize hypothesis and strategy tracking
        self.hypothesis_counter = 0
        self.hypotheses = []
        self.strategy_counter = 0
        self.strategies = []
        
        if should_review:
            self.is_reviewed = False
            self.file_by_file_reviewed = False
        else:
            self.is_reviewed = True
            self.file_by_file_reviewed = True
        os.makedirs(self.observation_dir, exist_ok=True)
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
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
            overwrite: If True, will overwrite the file if it exists. If False and file exists, returns an error. (default: False)
        Returns:
            Status message indicating success or error.
        """
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Set overwrite=True to overwrite."
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True) 
            self.file_ops.save(file_path, content) 
            return f"File '{file_path}' created successfully."
        except Exception as e:
            return f"Error creating file '{file_path}': {e}"
    @EnhancedToolManager.tool
    def run_code(self, file_path: str, run_command: List[str], content: str = None) -> str:
        """
        Runs code. Executes the code with run_command.
        Arguments:
            file_path: Code file path to run, relative to current directory (e.g., "file.py", "file.js")
            run_command: Command to execute the code (e.g., ["python", "file.py"], ["node", "file.js"])
            content: Content of the file to run (optional)
        Output:
            Execution result or error message
        """
        # Load file content
        if content is None:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                return f"Error: Could not read file '{file_path}': {e}"
        version_fix_needed = file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz"))
        fix_missing = version_fix_needed and VERSION_COMPATIBILITY_FIX not in content
        original_content = content if fix_missing else None
        if fix_missing:
            content = f"{VERSION_COMPATIBILITY_FIX}\n\n{content}"
        try:
            self.file_ops.save(file_path, content)
            self.generated_test_files.append(file_path)
        except Exception as e:
            return f"Error: Failed to save file '{file_path}': {e}"
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
        if fix_missing and original_content is not None:
            try:
                self.file_ops.save(file_path, original_content)
            except Exception as e:
                exec_result = (exec_result or "") + f"\nWarning: Failed to restore original file after execution: {e}"
        return exec_result
    @EnhancedToolManager.tool
    def run_bash(self, bash_command: List[str]) -> str:
        """
        Executes a bash command.
        Arguments:
            bash_command: list of command line arguments, e.g., ["ls", "-l"]
        Returns:
            Standard output or error output of the command.
        """
        try:
            logger.info(f"Running bash command in run_bash: {bash_command}")
            result = subprocess.run(
                bash_command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )
            if result.returncode != 0:
                return f"Error running bash command: {result.stderr}\n{result.stdout}" 
            return f"{result.stdout}\n{result.stderr}" 
        except Exception as e:
            return f"Error: {e}"
    @EnhancedToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self.file_ops.get_file_content(file_path, search_start_line, search_end_line, search_term, add_line_numbers=True, limit=1000)
    @EnhancedToolManager.tool
    def create_hypothesis(self, description: str, evidence: str) -> str:
        """Log a hypothesis about the bug's root cause.
        Use this tool when you form a theory about what's causing the issue. This helps track
        which theories you've already considered, especially important after rollbacks when
        you might lose track of what you've tried.
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
    def mark_strategy_outcome(
        self, strategy_id: int, success: bool, reason: str
    ) -> str:
        """Record whether a strategy worked.
        After attempting a strategy, record the outcome. This is crucial for institutional memory,
        especially when using rollbacks - you'll know what you already tried even after reverting changes.
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
    def list_hypotheses(self) -> str:
        """View all hypotheses with their test status.
        Use this to review what theories you've already considered and tested. Especially useful:
        - After a rollback (to see what you learned before rolling back)
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
        output.append(
            f"Summary: {len(confirmed)} confirmed, {len(rejected)} rejected, {len(inconclusive)} inconclusive, {len(untested)} untested\n"
        )
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
        Use this BEFORE making significant code changes to log your planned approach. This creates
        a history that persists across rollbacks, preventing you from retrying failed strategies.
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
    def list_attempted_strategies(self) -> str:
        """View all strategies tried, with outcomes.
        Use this to review what approaches you've already attempted. Critical for:
        - Avoiding retry loops (especially after rollbacks)
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
        output.append(
            f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} pending\n"
        )
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
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        return self.search_manager.search_in_file(file_path=file_path, search_term=search_term)
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
        try:
            obs_files = sorted(
                [
                    f
                    for f in os.listdir(self.observation_dir)
                    if f.startswith("observation_")
                ],
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
                lines = content.split("\n")
                failed = []
                for line in lines:
                    if "FAILED" in line or "failed" in line.lower():
                        failed.append(line.strip())
                if failed:
                    output.append(f"Found {len(failed)} failed test reference(s):\n")
                    for line in failed[:20]: 
                        output.append(f"  {line}")
                    if len(failed) > 20:
                        output.append(f"\n... and {len(failed) - 20} more")
                else:
                    output.append("No failed tests found in observation")
            elif query_type == "warnings":
                lines = content.split("\n")
                warnings = [
                    line.strip()
                    for line in lines
                    if "warning" in line.lower() or "warn" in line.lower()
                ]
                if warnings:
                    output.append(f"Found {len(warnings)} warning(s):\n")
                    for warning in warnings[:10]:
                        output.append(f"  {warning}") 
                    if len(warnings) > 10:
                        output.append(f"\n... and {len(warnings) - 10} more")
                else: 
                    output.append("No warnings found in observation")
            elif query_type == "summary":
                lines = content.split("\n")
                output.append(f"Total lines: {len(lines)}")
                output.append(f"Total characters: {len(content)}")
                error_count = sum(1 for line in lines if "error" in line.lower())
                warning_count = sum(1 for line in lines if "warning" in line.lower())
                passed_count = sum(
                    1 for line in lines if "PASSED" in line or "passed" in line.lower()
                )
                failed_count = sum(
                    1 for line in lines if "FAILED" in line or "failed" in line.lower()
                )
                output.append(f"\nKeyword counts:")
                output.append(f"  Errors: {error_count}")
                output.append(f"  Warnings: {warning_count}")
                output.append(f"  Passed: {passed_count}") 
                output.append(f"  Failed: {failed_count}") 
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
    @EnhancedToolManager.tool
    def think(self, thought: str) -> str:
        """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective. Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests.
        Arguments:
            thought: Your thoughts.
        Output:
            Confirmation that the thought has been logged.
        """
        return "ok"
    @EnhancedToolManager.tool
    def assess_change_confidence(self, proposed_change: str, file_path: str, reasoning: str) -> str:
        """Evaluate confidence in a proposed code change before applying it.
        This tool helps prevent low-confidence changes that waste steps. Use it when you're uncertain
        about a change to get objective assessment before implementing.
        Arguments:
            proposed_change: Description of what you plan to change
            file_path: The file you plan to modify
            reasoning: Why you think this change will work
        Output:
            Confidence assessment with recommendation: proceed / investigate_more / try_different_approach
        """
        if not proposed_change or not proposed_change.strip():
            return "Error: proposed_change cannot be empty"
        if not file_path or not file_path.strip(): 
            return "Error: file_path cannot be empty" 
        if not reasoning or not reasoning.strip():
            return "Error: reasoning cannot be empty - explain why you think this will work"
        if len(proposed_change) > 2000:
            proposed_change = proposed_change[:2000] + "\n... [truncated]"
        if len(reasoning) > 2000:
            reasoning = reasoning[:2000] + "\n... [truncated]"
        file_context = ""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    if len(content) > 3000:
                        file_context = content[:3000] + "\n... [truncated]"
                    else:
                        file_context = content
        except Exception:
            file_context = "Unable to read file"
        assessment_prompt = f"""You are helping an AI coding agent assess confidence in a proposed code change.
PROPOSED CHANGE:
{proposed_change}
FILE TO MODIFY: {file_path}
REASONING:
{reasoning}
FILE CONTEXT:
{file_context}
Assess the confidence in this proposed change:
1. **Correctness Likelihood** (0-10): How likely is this change to actually fix the problem?
2. **Risk of Breaking Things** (0-10): How likely is this to introduce new bugs?
3. **Edge Case Coverage** (0-10): How well does this handle edge cases?
Based on this assessment, provide ONE of three recommendations:
- CONFIDENCE: HIGH (score >= 7 on all metrics) - Proceed with this change
- CONFIDENCE: MEDIUM - INVESTIGATE MORE (score 4-6 on any metric)  
- CONFIDENCE: LOW - TRY DIFFERENT APPROACH (score < 4 on any metric)
Include scores and specific guidance."""
        try:
            messages = [{"role": "user", "content": assessment_prompt}]
            for model in [GLM_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]:
                try:
                    response, _ = EnhancedNetwork.make_request(messages, model=model, temperature=0.15)
                    if response:
                        return f"🎲 CHANGE CONFIDENCE ASSESSMENT\n{'='*80}\n{response}\n{'='*80}\n\nIMPORTANT: If confidence is MEDIUM or LOW, follow the recommendations before proceeding."
                except Exception:
                    continue 
            return "Error: Failed to generate confidence assessment after trying multiple models"
        except Exception as e: 
            return f"Error generating confidence assessment: {str(e)}"
    @EnhancedToolManager.tool
    def decompose_complex_problem(self, problem_statement: str, code_context: str) -> str:
        """Break down a complex problem into ordered sub-problems with clear roadmap.
        Use this tool when the problem seems complex or multi-faceted.
        Arguments:
            problem_statement: The full problem description
            code_context: Relevant code snippets or file names you've discovered
        Output:
            Structured decomposition with 3-5 sub-problems, dependencies, and roadmap.
        """
        if not problem_statement or not problem_statement.strip():
            return "Error: problem_statement cannot be empty"
        # Limit lengths
        if len(problem_statement) > 5000:
            problem_statement = problem_statement[:5000] + "\n... [truncated]"
        if len(code_context) > 3000:
            code_context = code_context[:3000] + "\n... [truncated]"
        decomposition_prompt = f"""You are helping an AI coding agent break down a complex problem into manageable sub-problems.
PROBLEM STATEMENT:
{problem_statement}
CODE CONTEXT:
{code_context}
Break this into 3-5 ordered sub-problems. For EACH sub-problem, provide:
1. **Sub-Problem Title**: Clear, concise description
2. **Description**: What needs to be accomplished
3. **Success Criteria**: How to know this sub-problem is solved
4. **Estimated Complexity** (Simple/Moderate/Complex)
5. **Dependencies**: Which other sub-problems must be solved first
Then provide:
**RECOMMENDED ORDER**: In what sequence should these be tackled?
**OVERALL STRATEGY**: High-level approach
Be specific and actionable."""
        try:
            messages = [{"role": "user", "content": decomposition_prompt}]
            for model in [GLM_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]:
                try:
                    response, _ = EnhancedNetwork.make_request(messages, model=model, temperature=0.2)
                    if response:
                        return f"📋 PROBLEM DECOMPOSITION\n{'='*80}\n{response}\n{'='*80}\n\nIMPORTANT: Use this roadmap to guide your approach."
                except Exception:
                    continue
            return "Error: Failed to generate problem decomposition after trying multiple models"
        except Exception as e:
            return f"Error generating problem decomposition: {str(e)}"
    @EnhancedToolManager.tool
    def select_thinking_mode(self, current_situation: str, step_number: int, recent_outcomes: str) -> str:
        """Select optimal thinking depth for current situation to balance thoroughness with step efficiency.
        Arguments:
            current_situation: What you're about to do
            step_number: Current step count
            recent_outcomes: Brief summary of last 2-3 actions and their results
        Output:
            Recommended thinking mode (FAST/NORMAL/DEEP) with specific guidance.
        """
        if not current_situation or not current_situation.strip():
            return "Error: current_situation cannot be empty"
        # Limit lengths
        if len(current_situation) > 1500:
            current_situation = current_situation[:1500] + "\n... [truncated]"
        if len(recent_outcomes) > 2000:
            recent_outcomes = recent_outcomes[:2000] + "\n... [truncated]"
        mode_selection_prompt = f"""You are helping an AI coding agent select the optimal thinking depth.
CURRENT SITUATION:
{current_situation}
STEP NUMBER: {step_number} (Target: ~15 steps, Max: 30 steps)
RECENT OUTCOMES:
{recent_outcomes}
Select the appropriate thinking mode:
**FAST MODE**: Minimal reasoning, quick execution (simple operations)
**NORMAL MODE**: Standard reasoning depth (standard coding tasks)
**DEEP MODE**: Extended analysis (complex decisions, stuck situations)
Recommend:
```
RECOMMENDED MODE: [FAST/NORMAL/DEEP]
REASONING: [Why this mode is appropriate]
SPECIFIC GUIDANCE: [What to focus on]
STEP BUDGET ALERT: [If steps > 20, warn about remaining budget]
```"""
        try:
            messages = [{"role": "user", "content": mode_selection_prompt}]
            for model in [GLM_MODEL_NAME, KIMI_MODEL_NAME, QWEN_MODEL_NAME]:
                try:
                    response, _ = EnhancedNetwork.make_request(messages, model=model, temperature=0.1)
                    if response:
                        return f"⚡ THINKING MODE SELECTION\n{'='*80}\n{response}\n{'='*80}\n\nIMPORTANT: Adapt your thinking depth accordingly."
                except Exception:
                    continue
            return "Error: Failed to generate thinking mode selection after trying multiple models"
        except Exception as e:
            return f"Error generating thinking mode selection: {str(e)}"
    @EnhancedToolManager.tool
    def design_api_contract(self, feature_description: str, requirements: str) -> str:
        """Design function signatures, class interfaces, and module boundaries BEFORE implementation.
        CRITICAL for CREATE tasks - defines the API contract before writing code.
        Arguments:
            feature_description: What feature/functionality you're building
            requirements: Key requirements and constraints from problem statement
        Output:
            Structured API design with function signatures, data structures, edge cases
        """
        if len(feature_description) > 3000:
            feature_description = feature_description[:3000] + "...(truncated)"
        if len(requirements) > 2000:
            requirements = requirements[:2000] + "...(truncated)"
        prompt = f"""You are an expert software architect designing APIs for new functionality.
FEATURE TO BUILD:
{feature_description}
KEY REQUIREMENTS:
{requirements}
Design a clear API contract that includes:
1. **Function/Class Signatures**: Define all public functions/classes needed
2. **Data Structures**: Define key data structures
3. **Module Organization**: How should code be organized
4. **Edge Case Handling**: What edge cases must the API handle
5. **Integration Points**: How new code integrates with existing codebase
Provide a clear, implementable API design."""
        reasoning_models = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME] 
        for model in reasoning_models: 
            try:
                messages = [{"role": "user", "content": prompt}]
                response, _ = EnhancedNetwork.make_request(messages, model=model, temperature=0.15)
                return f"API Design (using {model}):\n\n{response}"
            except Exception:
                continue
        return "Error: Could not generate API design with available models"
    @EnhancedToolManager.tool
    def validate_design_completeness(self, proposed_design: str, original_requirements: str) -> str:
        """Verify that planned design addresses ALL requirements from problem statement.
        CRITICAL for CREATE tasks - catches missing features before implementation.
        Arguments:
            proposed_design: Your API design or implementation plan
            original_requirements: Full problem statement/requirements
        Output:
            Requirement coverage analysis with gaps and missing components
        """
        if len(proposed_design) > 4000:
            proposed_design = proposed_design[:4000] + "...(truncated)"
        if len(original_requirements) > 3000:
            original_requirements = original_requirements[:3000] + "...(truncated)"
        prompt = f"""You are a requirements validation expert. Analyze if the proposed design covers ALL requirements.
ORIGINAL REQUIREMENTS:
{original_requirements}
PROPOSED DESIGN:
{proposed_design}
Perform a thorough requirement coverage analysis:
1. **Extract ALL Requirements**: List every requirement from the original
2. **Coverage Check**: For each requirement, verify if design addresses it
3. **Gap Analysis**: Identify missing/incomplete requirements
4. **Edge Case Coverage**: Check if design handles edge cases
Format your analysis with coverage summary and critical gaps."""
        reasoning_models = [KIMI_MODEL_NAME, GLM_MODEL_NAME, QWEN_MODEL_NAME]
        for model in reasoning_models:
            try:
                messages = [{"role": "user", "content": prompt}]
                response, _ = EnhancedNetwork.make_request(messages, model=model, temperature=0.1)
                return f"Design Validation (using {model}):\n\n{response}"
            except Exception:
                continue
        return "Error: Could not validate design completeness with available models"
    @EnhancedToolManager.tool
    def plan_incremental_build(self, api_design: str, step_budget_remaining: int) -> str:
        """Break implementation into 3-5 testable increments (simplest → full features).
        CRITICAL for CREATE tasks - enables early validation and manages step budget.
        Arguments:
            api_design: Your API design or feature description
            step_budget_remaining: Estimated steps remaining (target ~15, max 30)
        Output:
            Phased implementation plan with test criteria for each phase
        """
        if len(api_design) > 4000:
            api_design = api_design[:4000] + "...(truncated)"
        prompt = f"""You are an expert at incremental software development. Create a phased build plan.
API/FEATURE TO BUILD:
{api_design}
STEP BUDGET: ~{step_budget_remaining} steps remaining (target 15, max 30)
Create a 3-5 phase incremental build plan. Each phase should:
- Build on previous phase
- Be independently testable
- Add measurable functionality
- Fit within step budget
Format each phase with Goal, Implementation tasks, Test Criteria, and Estimated Steps.
Phases should go from simple → complex."""
        reasoning_models = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME]
        for model in reasoning_models:
            try:
                messages = [{"role": "user", "content": prompt}]
                response, _ = EnhancedNetwork.make_request(messages, model=model, temperature=0.2)
                return f"Incremental Build Plan (using {model}):\n\n{response}"
            except Exception: 
                continue 
        return "Error: Could not generate incremental build plan with available models"
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
            ls = subprocess.run(["git", "ls-files", "-m", "-o", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()
            to_add = [f for f in ls if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)
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
def llm_select_run_command_for_file(file_path: str) -> list:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception:
        return []
    prompt = (
        "I'd like you to respond with the command to run this file. Use built-in approach to run the file.\nMake your command as simple as possible.\n"
        "```\n"
        f"{file_path}\n"
        f"{file_content}\n"
        "```\n"
        "You must respond in JSON format:\n"
        "```\n"
        '{\n    "command": ["bbb", "aaa.js"]\n}\n'
        "```"
    )
    for _ in range(10):
        try:
            messages = [{"role": "user", "content": prompt}]
            raw_text, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            json_result = json.loads(cleaned)
            return json_result.get("command", [])
        except Exception:
            time.sleep(1)
    return []
def clean_code_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response
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
def generate_initial_solution(
    problem_statement: str, initial_structure: str, temperature: float = 0.7
) -> str:
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
            time.sleep(2)
    return ""
def create_task_solve_workflow(problem_statement, enhancement, timeout):
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "list_directory_structure",
            "search_in_file",
            "apply_code_edit",
            "run_code",
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "analyze_saved_observation",
            "assess_change_confidence",
            "decompose_complex_problem",
            "select_thinking_mode",
            "design_api_contract",
            "validate_design_completeness",
            "plan_incremental_build",
            "think",
            "finish",
        ],
        problem_statement=problem_statement,
    )
    multiple_search_guide = """
next_thought: I need to find all references to the function in file
tool_call_1:
    tool_name: search_in_file
    tool_args: {{"search_term": "function problematic_func"}}
tool_call_2:
    tool_name: search_in_file
    tool_args: {{"search_term": "problematic_func("}}
tool_call_3:
    tool_name: get_file_content
    tool_args: {{"file_path": "abcd.js"}}
"""
    search_guide = "- Use `search_in_file` to find all occurrences of an issue before fixing."
    system_prompt = CREATE_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT.format(multiple_search_guide = multiple_search_guide),
        search_guide = search_guide
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
    cot = EnhancedCOT(latest_observations_to_keep = LATEST_OBSERVATIONS_TO_KEEP, summarize_batch_size = SUMMARIZE_BATCH_SIZE, do_summarize = True)
    return execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        MAX_FIX_TASK_STEPS,
        timeout,
        [GLM_OLD_MODEL_NAME, QWEN_MODEL_NAME],            
        log_prefix="CREATE_MAIN_AGENT",
    )
def get_modify_contents(problem_statement) -> str:
    all_files = []
    excluded_dirs = {'.git', '__pycache__', 'node_modules', '.pytest_cache', '.venv', 'venv', '.tox'}
    
    for r, dirs, files in os.walk("."):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        
        for f in files:
            file_path = os.path.join(r, f)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    # Only include files with actual content
                    if content.strip():
                        all_files.append({
                            'name': f,
                            'path': file_path,
                            'content': content
                        })
            except Exception:
                pass
    
    # Use LLM to determine the primary language and select relevant skeleton files
    LANGUAGE_AND_FILES_DETECTION_PROMPT = textwrap.dedent("""
    You are an expert code analyzer. Analyze ALL the provided files and their content to:
    1. Determine if this is primarily which language code generation project
    2. Identify which files are relevant skeleton/stub files that need to be implemented
    
    Analyze the content carefully:
    1. Look at the problem statement - what language does it suggest?
    2. Examine the actual code syntax in the files (even if they don't have standard extensions)
    3. Check for language-specific patterns (imports, syntax, keywords)
    4. Consider which language has more substantial implementation code
    5. Identify stub files or skeleton code that needs to be implemented
    6. Select ONLY the files that are relevant to the problem statement (exclude test files, config files, etc. unless specifically needed)
    
    You must respond only in JSON format with the language and list of file paths:
    ```json
    {
        "language": "language_name",
        "skeleton_files": ["./path/to/file1.py", "./path/to/file2.py"]
    }
    ```
    
    IMPORTANT: 
    - You must respond only in JSON format.
    - Use the exact file paths as shown in the file list
    - Only include files that are relevant to implementing the solution
    - Exclude test files, documentation, and configuration files unless they are essential
    - If no files are relevant, return an empty list
    """)
    
    context = f"Problem Statement:\n{problem_statement}\n\n"
    context += f"Found {len(all_files)} files in the repository.\n\n"
    context += "Available Files:\n"
    
    for i, file_info in enumerate(all_files):
        file_preview = file_info['content'][:500]  # First 500 chars
        context += f"\nFile #{i+1}: {file_info['path']}\n"
        context += f"Content preview:\n{file_preview}\n"
        if len(file_info['content']) > 500:
            context += "...(truncated)\n"
        context += "---\n"
    
    logger.info(f"Language and file detection context prepared with {len(all_files)} files")
    
    language = ""
    selected_file_names = []
    
    retry = 0
    while retry < 5:
        try:
            messages = [
                {"role": "system", "content": LANGUAGE_AND_FILES_DETECTION_PROMPT},
                {"role": "user", "content": context}
            ]
            response, _ = EnhancedNetwork.make_request(messages, model=KIMI_MODEL_NAME, temperature=0.0)
            logger.info(f"Language and files detection response: {response}")
            # Parse JSON response
            response_clean = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(response_clean)
            
            language = result.get("language", "").lower()
            
            selected_file_names = result.get("skeleton_files", [])
            
            if not selected_file_names:
                logger.warning("LLM returned empty file list")
            
            logger.info(f"LLM detected language: {language}, selected files: {selected_file_names}")
            break
        except Exception as e:
            logger.warning(f"Error using LLM for language/file detection (attempt {retry+1}): {e}")
            retry += 1
            time.sleep(2)
    
    skeleton_files = []
    skeleton_file_names = []
    file_map = {f['name']: f for f in all_files}
    file_path_map = {f['path']: f for f in all_files}
    
    for selected_name in selected_file_names:
        file_info = None
        if selected_name in file_path_map:
            file_info = file_path_map[selected_name]
        elif selected_name in file_map:
            file_info = file_map[selected_name]
        else:
            for f in all_files:
                if selected_name in f['path'] or selected_name in f['name'] or f['path'] in selected_name or f['name'] in selected_name:
                    file_info = f
                    break
        
        if file_info:
            skeleton_files.append(f"{file_info['path']}\n{{\n{file_info['content']}\n}}")
            skeleton_file_names.append(file_info['path'])
            logger.info(f"Added file to skeleton: {file_info['path']}")
        else:
            logger.warning(f"Selected file '{selected_name}' not found in all_files")
    
    skeleton = "\n\n".join(skeleton_files) if skeleton_files else ""
    logger.info(f"Built skeleton with {len(skeleton_files)} files for {language}: {skeleton_file_names}")
    return skeleton
def process_create_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id
    print("Processing create task")
    total_timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    problem_statement = input_dict.get("problem_statement", "")
    tool_manager = EnhancedToolManager()
    initial_structure = get_modify_contents(problem_statement)    
    if len(initial_structure) > 0:
        initial_solution, _ = single_process_create_task(
            problem_statement, initial_structure
        )
        if initial_solution is not None:        
            os.system("git reset --hard")
            extract_and_write_files(initial_solution)
            patch = tool_manager.get_final_git_patch()
            return patch
    elapsed_time = time.time() - agent_start_time
    return create_task_solve_workflow(problem_statement, enhancement, total_timeout - elapsed_time - 60)
def is_all_tests_passed(output: str) -> bool:
    prompt = (
        "Check the test output and tell me if all the tests passed successfully or there is any failure or error.\n"
        "This is the output:\n"
        "```\n"
        f"{output}\n"
        "```\n"
        'Return only "true" or "false".'
    )
    for _ in range(9):
        try:
            result, _ = EnhancedNetwork.make_request(
                messages=[{"role": "user", "content": prompt}],
                model=QWEN_MODEL_NAME
            )
            return result.lower() == "true"
        except Exception as e:
            logger.error("[IS_ALL_TESTS_PASSED] Exception: %s", e)
            time.sleep(1)
    return False
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
    selected_model = KIMI_MODEL_NAME
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
def process_fix_task(input_dict: Dict[str, Any], enhancement: str):
    global run_id
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    logs = []
    patch_text = ""
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
            logger.info(f"Result of running command: {result}")
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
            other_models = [model for model in AGENT_MODELS if model != selected_model]
            if other_models:
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return ""
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
            other_models = [model for model in AGENT_MODELS if model != selected_model]
            selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
    return ""
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
    logs: List[str] = [f"cwd: {os.getcwd()}"]
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought, next_tool_name, next_tool_args = None, None, None
    modified_files, files_with_syntax_errors = set(), set()
    model_len = len(models)
    for step in range(n_max_steps):
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
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
        cost_usage = EnhancedNetwork.get_cost_usage()
        logger.info(
            "=" * 40 + f"[{log_prefix}] Step {step}" + "=" * 40
        )
        logger.info(
            f"[{log_prefix}] Elapsed time: {elapsed_time}/{timeout} seconds, Usage: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
        )
        if cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", 0):
            logger.warning(
                f"[{log_prefix}] Usage exceeded limit: {cost_usage.get('used_cost_usd', 0)}/ {cost_usage.get('max_cost_usd', 0)} USD"
            )
            break
        selected_model = models[(cot.repeated_thoughts - 2) % model_len] if (cot.is_thought_repeated() and cot.repeated_thoughts >= 2) else models[0]
        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        temperature = 0.5 if cot.is_thought_repeated() else 0.0
        if cot.is_thought_repeated():
            logger.info(f"[TEMPERATURE] Thought repeated {cot.repeated_thoughts} times")
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
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
            ) = EnhancedNetwork.inference(
                messages,
                model=models_to_try,
                run_id=run_id,
                temperature=temperature,
            )
            selected_model = used_model
            inference_duration = time.time() - inference_start_time
        except Exception as e:
            logger.error(f"[{log_prefix}] Inference error: {e}")
            if "Agent execution timeout" in str(e):
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
            inference_duration = 0
        tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]
        logger.info(
            f"[{log_prefix}] Used model: {selected_model}, Inference time: {inference_duration:.2f}s"
        )
        logger.info(f"[{log_prefix}] Next thought: {next_thought}\n\n")
        logger.info(
            f"[{log_prefix}] About to execute {len(tool_names_list)} tool call(s): {tool_names_list}\n"
        )
        logger.info(
            f"[{log_prefix}] Tool arguments: {json.dumps(tool_args_list, indent=4)}\n\n"
        )
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
        for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
            try:
                tool_name = tool_name.replace('"', "").replace("'", "") if isinstance(tool_name, str) else tool_name
                if tool_args:
                    observation = tool_manager.get_tool(tool_name)(**tool_args)
                else:
                    observation = tool_manager.get_tool(tool_name)()
                if (
                    tool_name == "apply_code_edit"
                    and tool_args
                    and "file_path" in tool_args
                ):
                    file_path = tool_args["file_path"]
                    obs_lower = str(observation).lower()
                    if "ok, code edit applied successfully" in obs_lower:
                        modified_files.add(file_path)
                    elif "syntax error" in obs_lower:
                        files_with_syntax_errors.add(file_path)
                # Observation token length checks and truncations/offloading
                estimated_tokens = Utils.count_tokens(str(observation))
                if estimated_tokens > reject_observation_token_threshold:
                    observation = (
                        f"Error: Tool output from '{tool_name}' exceeded token limit "
                        f"({estimated_tokens} tokens > {reject_observation_token_threshold} tokens limit). "
                        "The response is too large to process. Please use more specific queries, "
                        "target smaller file ranges, or break the request into smaller operations."
                    )
                elif estimated_tokens > save_observation_to_file_token_threshold:
                    observation_path = tool_manager._save_large_observation(
                        str(observation), tool_name
                    )
                    observation = (
                        f"Tool output from `{tool_name}` exceeded token limit "
                        f"({estimated_tokens} tokens > {save_observation_to_file_token_threshold} tokens limit). "
                        f"The full output has been saved to: {observation_path}. "
                        "You can read this file using the get_file_content tool if needed."
                    )
                all_observations.append(observation)
            except EnhancedToolManager.Error as e:
                all_successful = False
                error_msg = f"Tool {idx+1} ({tool_name}) error: {e.message}"
                all_observations.append(error_msg)
            except Exception as e:
                all_successful = False
                import traceback
                error_traceback = traceback.format_exc()
                error_msg = f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                all_observations.append(error_msg)
        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = (
                "\n\n--- Tool Call Results ---\n"
                + "\n\n".join(
                    f"Tool {i+1} ({tool_names_list[i]}):\n{obs}"
                    for i, obs in enumerate(all_observations)
                )
            )
        logger.info(f"[{log_prefix}] Combined observation: {combined_observation}\n\n")
        cot.add_action(
            EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
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
            for name, obs in zip(tool_names_list, all_observations):
                if name == finish_tool_name:
                    if finish_tool_name == "finish_find_files_to_fix":
                        return obs
                    elif finish_tool_name == "finish":
                        if obs == "finish":
                            return tool_manager.get_final_git_patch()
    return tool_manager.get_final_git_patch()
def single_process_create_task(
    problem_statement: str, initial_structure: str
) -> tuple[str, str] | tuple[None, None]:
    BASIC_APPROACH_RETRY = 7
    for i in range(BASIC_APPROACH_RETRY):
        os.system("git reset --hard")
        initial_solution, test_cases = basic_approach(
            initial_structure, problem_statement
        )
        if initial_solution is not None:
            return (initial_solution, test_cases)
        time.sleep(2)
    return (None, None)
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
                result, _ = EnhancedNetwork.make_request(messages=[{"role": "user", "content": file_names_prompt}], model=selected_model)
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
            current_file = next((f for f in file_names if f == stripped or f.endswith("/" + stripped) or f.split("/")[-1] == stripped), stripped)
            current_file, content = current_file, []
        elif current_file:
            content.append(line)
    write_file()
    return created_files
def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str, 
    enhancement: str, 
    n_max_steps=MAX_FIX_TASK_STEPS,
    initial_checkpoint=None,
    should_review: bool = True
) -> tuple[str, list, list]:
    global run_id, _current_tool_manager
    run_id = run_id_1
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "create_new_file",
            "list_directory_structure",
            "get_file_content",
            "search_in_all_files_content",
            "apply_code_edit",
            "run_code",
            "run_bash",
            "finish",
        ],
        initial_checkpoint=initial_checkpoint,
        problem_statement=problem_statement,
        should_review=should_review,
    )
    _current_tool_manager = tool_manager
    logger.info(
        "Starting main agent execution... Enhancement: %s", enhancement
    )
    logger.info("Available tools: %s", tool_manager.available_tools)
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT.format(multiple_search_guide = ""),
    )
    enhanced_problem = (
        f"{problem_statement}\n\n---\n\n# Enhanced Problem Analysis\n\n{enhancement}"
        if enhancement else problem_statement
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
        log_prefix="FIX_MAIN_AGENT"
    )
    
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
        enhancement = ""
        try:
            global _current_tool_manager
            
            _current_tool_manager = EnhancedToolManager()
            problem_type, enhancement = check_problem_type(input_dict.get("problem_statement"))
            logger.info(f"Problem type: {problem_type}")
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