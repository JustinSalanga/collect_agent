from __future__ import annotations
import re
import subprocess
import sys
import requests
import json
import inspect
import traceback
import random
import textwrap
import time
import functools
import os
import uuid
from pathlib import Path
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, cast, Callable
from enum import Enum
from pydantic import BaseModel, ValidationError
PROBLEM_TYPE_FIX = "FIX"
PROBLEM_TYPE_CREATE = "CREATE"
RUN_ID = os.getenv("EVALUATION_RUN_ID", str(uuid.uuid4()))
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1500"))
MAX_FIX_TASK_STEPS = 250
class ToolCall(BaseModel):
    tool_name: str
    tool_args: dict
class Model(BaseModel):
    name: str
    timeout: int
class LlmMessage(BaseModel):
    class Role(Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
    role: LlmMessage.Role
    content: str
    @classmethod
    def system(cls, content: str) -> LlmMessage:
        return cls(role=cls.Role.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> LlmMessage:
        return cls(role=cls.Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> LlmMessage:
        return cls(role=cls.Role.ASSISTANT, content=content)
    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}
GLM_MODEL_4_6 = Model(name="zai-org/GLM-4.6-FP8", timeout=150)
GLM_MODEL_4_5 = Model(name="zai-org/GLM-4.5-FP8", timeout=150)
QWEN_MODEL = Model(name="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", timeout=50)
KIMI_MODEL = Model(name="moonshotai/Kimi-K2-Instruct", timeout=40)
DEEPSEEK_MODEL = Model(name="deepseek-ai/DeepSeek-V3-0324", timeout=50)
AGENT_MODELS=[QWEN_MODEL, GLM_MODEL_4_6, GLM_MODEL_4_5, KIMI_MODEL, DEEPSEEK_MODEL]
FORMAT_PROMPT = textwrap.dedent("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚡ RESPONSE FORMAT (MUST READ FIRST!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **REQUIRED format for EVERY response**:
thought: [Your reasoning]
tool_call_1:
    tool_name: [tool name]
    tool_args: {{[JSON]}}
tool_call_2:
    tool_name: [tool name]
    tool_args: {{[JSON]}}
2. **Rules**:
- Start with `thought:` (NOT `next_thought` or `thought_1`)
- MUST have at least ONE tool_call
- Number sequentially: tool_call_1, tool_call_2, tool_call_3...
3. **✅ CORRECT Example**:
thought: I need to read and search multiple files (first 300 lines)
tool_call_1:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path1]"}}
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "grep -rn '[error_search_pattern]' [search_directory] | head -100"}}
Note: Use head -300 for first 300 lines. For further reading, use sed -n '301,600p' [file_path] for lines 301-600.
4. **❌ WRONG Examples**:
- No tool calls:
thought: I understand the problem
(Missing tool_call_1, tool_call_2...) ← INVALID
Wrong field names:
next_thought: Let me analyze
tool_call_1: ...
- Malformed JSON:
tool_call_1:
    tool_name: bash
    tool_args: {{command: "[test_command]"}}  ← INVALID (Missing quotes)
- No thought field:
tool_call_1:
    tool_name: bash
    tool_args: {{"command": "[test_command]"}} ← INVALID
""")
FORMAT_PROMPT_V1 = textwrap.dedent("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚡ RESPONSE FORMAT (MUST READ FIRST!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **REQUIRED format for EVERY response**:
thought: [Your reasoning]
tool_call_1:
    tool_name: [tool name]
    tool_args: {{[JSON]}}
2. **Rules**:
- Start with `thought:` (NOT `next_thought` or `thought_1`)
- MUST have ONly ONE tool_call
- tool_call_1
3. **✅ CORRECT Example**:
thought: I need to read and search files (first 300 lines)
tool_call_1:
    tool_name: view_file
    tool_args: {{"path": "aaa/xyz"}}
4. **❌ WRONG Examples**:
- No tool calls:
thought: I understand the problem
(Missing tool_call_1...) ← INVALID
Wrong field names:
next_thought: Let me analyze
tool_call_1: ...
- Malformed JSON:
tool_call_1:
    tool_name: view_file
    tool_args: {{path: "aaa/xyz"}}  ← INVALID (Missing quotes)
- No thought field:
tool_call_1:
    tool_name: view_file
    tool_args: {{"path": "aaa/xyz"}} ← INVALID
""")
FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 MISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Goal**: Fix the bug described in <problem_statement> with minimal code changes.
**Rules**:
- Make minimal changes to non-test files only
- DO NOT modify existing test files
- Use RELATIVE paths from working directory: {working_directory}
- DO NOT use absolute path
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 9-STEP WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Follow these steps systematically to resolve the issue:
1. **Understand the Problem and Explore Repository**
- Carefully read the issue and think critically about what is required.
- **Think hard about a plan to solve it before coding** - don't rush into implementation.
- Explore the repo to familiarize yourself with its structure.
- Find all issue files and relevant test files using `bash` tool with `find` and `grep` commands.
    - **Important**: The issue is newly found after publication, so there are no direct test files handling this specific issue.
    - **However**, Relevant test files are the source of truth - they show the expected behavior and can help you understand how the code should work.
    - All relevant test files will be used for regression testing after the fix to ensure your changes don't break existing functionality.
2. **Reproduce the Error**
- Create a script to reproduce the error and execute it with `[runner] [file_path]` using the bash tool.
- Confirm the error occurs as described in the problem statement.
- This validates that you understand the issue correctly before attempting a fix.
3. **Investigate Codebase and Identify Root Cause**
- Explore relevant files and directories related to the issue using `bash` tool with `find` and `grep` commands.
- Search for key functions, classes, or variables related to the issue using `grep -rn` to search across the codebase.
- **Find all relevant test files**: Use `bash` tool with `find` and `grep` commands to locate test files that might be related to the functionality being fixed. Search for files containing relevant functionality names, class names, function names, or keywords from the problem statement. For example: `find . -type f \\( -name "*[functionality_name]*" -o -name "*[class_name]*" -o -name "*[module_name]*" \\)` or `grep -rn "[functionality_name]\\|[class_name]\\|[module_name]\\|[relevant_keyword]" . | head -300`. Even if no direct tests handle the exact issue, relevant tests are the source of truth for expected behavior.
- **Analyze expected behavior from tests**: Read relevant test files using `bash` tool (e.g., `head -300 [some_test_file]`) to understand:
    - What the expected behavior should be
    - What edge cases are covered
    - What the correct output/behavior looks like
    - How the code should handle different scenarios
- Use the **sequential_thinking** tool to break down your analysis into clear, logical steps:
    - Step 1: What is the current behavior (from problem statement and code inspection)?
    - Step 2: What is the expected behavior (from relevant test files)?
    - Step 3: What is the gap between current and expected behavior?
    - Step 4: What code changes are needed to bridge this gap?
    - Continue with additional steps as needed (set totalThoughts to at least 5, up to 25 for complex issues)
- Use the **sequential_thinking** tool to analyze 5-7 different possible sources of the problem, then distill those down to 1-2 most likely sources.
- Identify the root cause of the problem (not just symptoms) through systematic sequential analysis.
- Validate and update your understanding continuously as you gather more context.
4. **Develop a Detailed Plan**
- Use the **sequential_thinking** tool to outline a specific, simple, and verifiable sequence of steps to fix the problem.
- Break down the fix into small, incremental changes.
- Before planning changes, read the relevant file contents using `bash` tool (e.g., `head -300 [file_path]`) to ensure complete context.
5. **Implement the Fix**
- Before editing, always read the relevant file contents or section using `bash` tool to ensure complete context.
- Choose fix location carefully: Examine error traces to understand the execution flow, then fix where you have the most context to properly resolve the issue.
- Edit the sourcecode of the repo using `str_replace_in_file` tool to resolve the issue.
- Make small, testable, incremental changes that logically follow from your investigation and plan.
- If a patch is not applied correctly, attempt to reapply it.
6. **Test the Fix**
- Rerun your reproduction script from Step 2 using `bash` tool and confirm that the error is fixed.
- Run tests frequently using the `bash` tool with test commands.
- After each change, verify correctness by running relevant tests immediately.
- Find **All** related test files from the repo (Step 1) using `bash` tool with `find` and `grep`, and run them to make sure that your fix doesn't break anything else.
- Ensure all tests pass before proceeding.
7. **Debug if Needed**
- If tests fail, analyze failures to understand why.
- When debugging, try to determine the root cause rather than addressing symptoms.
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening.
- To test hypotheses, you can also add test statements or functions.
- Revisit your assumptions if unexpected behavior occurs.
- Revise your patch based on debugging findings and return to Step 6.
**If same fix fails repeatedly (3+ attempts):**
- STOP and reconsider your approach entirely
- Check if you're fixing at the right location (where error occurs vs where data is created)
- Consider fixing at the error point if it has enough context to resolve the issue
- Try the opposite strategy (prevent at source ↔ handle at destination)
8. **Final Verification**
- Confirm the root cause is fixed.
- Review your solution for logic correctness and robustness.
- Think about edge cases and make sure your fix handles them as well.
9. **Final Reflection and Additional Testing**
- Reflect carefully on the original intent of the user and the problem statement.
- Think about potential edge cases or scenarios that may not be covered by existing tests.
- Write additional tests if needed using `create_file` tool to capture important behaviors or edge cases.
- Run these new tests using `bash` tool and ensure they all pass.
- Be aware that there are additional hidden tests that must also pass for the solution to be successful.
- Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.
GUIDE FOR HOW TO USE **sequential_thinking** TOOL:
- Your thinking should be thorough and so it's fine if it's very long. Set totalThoughts to at least 5, but setting it up to 25 is fine as well. You'll need more total thoughts when you are considering multiple possible solutions or root causes for an issue.
- Use this tool as much as you find necessary to improve the quality of your answers.
- You can run bash commands (like tests, a reproduction script, or 'grep'/'find' to find relevant context) in between thoughts.
- The **sequential_thinking** tool can help you break down complex problems, analyze issues step-by-step, and ensure a thorough approach to problem-solving.        
- Don't hesitate to use it multiple times throughout your thought process to enhance the depth and accuracy of your solutions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📐 CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Essential Guidelines**
- You are working in a codebase with other engineers and many different components. Be careful that changes you make in one component don't break other components. 
- When designing changes, implement them as a senior software engineer would. This means following best practices.
- When possible, choose the simpler solution.
- Use your `bash` tool to set up any necessary environment variables, such as those needed to run tests.
- You should run relevant tests to verify that your changes work.
- **Fix Location Principle**: When an error occurs, examine the full execution path. Fix at the location that has sufficient context to resolve the issue - this is often where the error manifests, not necessarily where the problematic data was created. Avoid modifying complex logic solely to satisfy constraints elsewhere when those constraints can handle or transform the data themselves.
2. **Multi-file awareness**
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `find and grep` to enumerate matches across the codebase and to drill into each file; iterate until no applicable occurrences remain.
- **Think about dependencies**: When you modify code that generates other files, you MUST regenerate those files and include them in the patch.
- **How to regenerate**: After modifying code that generates files, run code that uses/imports the modified code to trigger regeneration, then include the regenerated files in the patch.
- **Search for ALL related files**: Look for files with similar names, in the same directory, or that might be generated from your changes.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.
3. **File Operations + Immediate Execution**
File edit MUST include execution tool_call in SAME response.
✅ CORRECT:
thought: Edit and test file
tool_call_1:
    tool_name: str_replace_in_file
    tool_args: {{"path": "fix_file_path", "old_str": "old", "new_str": "new"}}
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "[runner] [fix_file_path]" or "[runner] [test_runner] [fix_file_path]"}}
❌ WRONG:
Response 1: Edit file only
Response 2: Execute file ← TOO LATE!
4. **Multi-Tool Call Batching**
Batch independent operations together.
Examples to batch:
• Multiple file reads → tool_call_1, 2, 3
• Multiple searches → tool_call_1, 2, 3
• Multiple test runs → tool_call_1, 2, 3
✅ CORRECT:
thought: Read multiple files together (first 300 lines each)
tool_call_1:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path1]"}}
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path2]"}}
tool_call_3:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path3]"}}
Note: Use head -300 for first 300 lines. For further reading, use sed -n '301,600p' [file_path] for lines 301-600.
❌ WRONG: Reading one file per response
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏗 TOOL USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Core Principle: Batch ALL independent operations in ONE response**
1. **File Edit + Immediate Execution**
thought: Fix bug and verify immediately
tool_call_1:
    tool_name: str_replace_in_file
    tool_args: {{"path": "[fix_file_path]", "old_str": "old_code", "new_str": "new_code"}}
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "[runner] [fix_file_path]"}}
tool_call_3:
    tool_name: bash
    tool_args: {{"command": "[runner] [test_runner] [fix_file_path]"}}       
2. **Batch File Reads (with 300 line limit)**
thought: Read implementation and related files (first 300 lines, use sed for further reading if needed)
tool_call_1:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path1]"}}
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path2]"}}
tool_call_3:
    tool_name: bash
    tool_args: {{"command": "head -300 [file_path3]"}}
Note: Use head -300 for first 300 lines (or all if file is smaller). If you need more, use sed -n '301,600p' [file_path] for lines 301-600, sed -n '601,900p' for lines 601-900, etc.
3. **Batch Search Operations**
thought: Locate files and search for patterns
tool_call_1:
    tool_name: bash
    tool_args: {{"command": "find . -name '[some_file_extension]' -type f"}}
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "grep -rn '[some_search_pattern]' ."}}
tool_call_3:
    tool_name: bash
    tool_args: {{"command": "find . -type f \\( -name '[some_file_extension]' -o -name '[some_file_extension]' -o -name '[some_file_extension]' \\)"}}
4. **Batch Testing**
thought: Run all related tests to check for regressions
tool_call_1:
    tool_name: bash
    tool_args: {{"command": "[runner] [test_runner] [test_file_path]"}}       
tool_call_2:
    tool_name: bash
    tool_args: {{"command": "[runner] [test_runner] [test_file_path]"}}
tool_call_3:
    tool_name: bash
    tool_args: {{"command": "[runner] [test_runner] [test_file_path]"}}
💡 Key Patterns:
• File edits → ALWAYS include execution in SAME response
• Independent operations → ALWAYS batch together
• Multiple searches → Batch all grep/find commands
• Multiple tests → Batch all test runs together
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧪 TEST RUNNING GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Purpose**
CRITICAL: Many projects have test runners with essential setup. Running tests directly may MISS setup and FAIL. You MUST detect and use the project's test runner if it exists.
2. **How to Detect Test Infrastructure**
Search for test-related files and patterns:
- Look for test directories, test runner and entry files 
- Check existing test files to see how they're invoked, look for test configuration files, check if tests exist at all
3. **How to Run Tests**
- Custom test runner found: Use the project's test runner (e.g., `[runner] [test_runner] [test_file_path]`)
- Standard test framework found: Use appropriate command
- No tests found: Verify the fix manually or check if tests should be created
4. All fails, then you must use appropriate test solutions for the project, for example - creating independent scripts.      
5. **Batch Multiple Test Runs (MANDATORY)**
ALWAYS batch multiple test runs using multi-tool calls in ONE response. Do NOT run tests one-by-one.
✅ CORRECT:
thought: Run all related tests
tool_call_1: bash("<test_command_1>")
tool_call_2: bash("<test_command_2>")
tool_call_3: bash("<test_command_3>")
❌ FORBIDDEN: Running one test per response
CRITICAL: ALL tests MUST pass. If any fails, fix is incorrect. Return to Step 6 with failure details.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📝 Problem Statement (<problem_statement>)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{problem_statement}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔨 Available Tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{available_tools}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚠ CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT:
- Install/update packages
- Run git commands
- Modify existing test files
MUST:
- Preserve backward compatibility
- Use multi-tool calls when possible
{output_format}
""")
DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent("""
# Do not repeat the same tool call with the same arguments.
1. **You're not allowed to repeat the same tool call with the same arguments.**
2. **Your previous response:**
{previous_response}
3. **TRY TO USE SOMETHING DIFFERENT!**
""")
CRITICAL_RULES_INJECTION = textwrap.dedent("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS (Check EVERY turn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Response Format** - Always respond with the restricted format:
    thought: [Your thinking process here]
    tool_call_1:
        tool_name: [tool_name]
        tool_args: {[valid_json]}
    tool_call_2:
        tool_name: [tool_name]
        tool_args: {[valid_json]}
    ...
2. **Problem + Related Tests** - Issue is new (no direct test), analyze related test patterns to infer expected behavior
3. **Multi-File Awareness** - Search entire codebase before fixing, fix all occurrences
4. **BATCH Tool Calls** - Use 3-5 tool calls in ONE response (grep/find/cat/str_replace_in_file)
    * Batching reduces turns and saves budget - don't call tools one-by-one
    * Example: Search ~2-3 patterns at once, fix ~2-3 files at once, read ~2-3 files at once
5. **Test Strategy** - Discover test runner (explore test-related entry points, scripts, configs), use original if possible
6. **Test Every Edit** - Edit + test in SAME turn (mandatory)
7. **NEVER Update Dependencies** - Do NOT modify project environment version, dependencies, packages, or config files. Only fix bugs in source code.
8. **Regression Verification**:
    * check all relevant test files to check for breaks and Ensure no side-effect or signature changes
""")
INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# 🚀 Now let's start. Follow 9-STEP WORKFLOW to fix the bugs in <problem_statement>
""")
class Utils:
    @classmethod
    def set_env_for_agent(cls, repo_dir: str):
        repo_dir = os.path.abspath(repo_dir)
        if not os.path.exists(repo_dir):
            raise FileNotFoundError(f"Repository directory does not exist: {repo_dir}")
        os.chdir(repo_dir)
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        pythonpath = os.environ.get("PYTHONPATH", "")
        if current_dir not in pythonpath:
            os.environ["PYTHONPATH"] = f"{pythonpath}:{current_dir}" if pythonpath else current_dir
        try:
            if not os.path.exists(".git"):
                subprocess.run(["git", "init"], check=True, timeout=10, capture_output=True)
                subprocess.run(["git", "config", "--global", "--add", "safe.directory", current_dir], timeout=5, capture_output=True)
                subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], timeout=5, capture_output=True)
                subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], timeout=5, capture_output=True)
                subprocess.run(["git", "add", "."], timeout=30, capture_output=True)
                subprocess.run(["git", "commit", "-m", "Initial commit"], timeout=30, capture_output=True, check=False)
            else:
                subprocess.run(["git", "config", "--global", "--add", "safe.directory", current_dir], timeout=5, capture_output=True)
        except Exception as e:
            print(f"Git initialization failed (non-critical): {e}")
    
    @classmethod
    def clean_code_response(cls, response: str) -> str:
        response = response.strip()
        response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
        response = response.removesuffix("```").strip()
        return response
    @classmethod
    def count_tokens(cls, messages: list[LlmMessage]) -> int:
        import re
        text = " ".join(m.content for m in messages)
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
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
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                fixed_json: dict | None = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)
    @classmethod
    def delete_files(cls, file_paths: list[str]):
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                pass
    @classmethod
    def get_code_with_line_numbers(cls, path: str, view_range: List[int] | None = None) -> str:
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                raise ValueError(f"No file found at the path {path}")
            if file_path.is_dir():
                if view_range:
                    raise ValueError("view_range parameter is not allowed when path points to a directory")
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            file_lines = file_content.split("\n")
            total_lines = len(file_lines)
            init_line = 1
            if view_range:
                if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                    raise ValueError("Invalid view_range. It should be a list of two integers")
                start, end = view_range
                if start < 1 or start > total_lines:
                    raise ValueError(f"Invalid view_range start {start}. Should be within [1, {total_lines}]")
                init_line = start
                if end == -1:
                    file_content = "\n".join(file_lines[start - 1:])
                else:
                    if end < start:
                        raise ValueError(f"Invalid view_range end {end}")
                    elif end > total_lines:
                        end = total_lines
                    file_content = "\n".join(file_lines[start - 1:end])
            numbered_content = "\n".join([
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate[str](file_content.split("\n"))
            ])
            return numbered_content
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
class EnhancedNetwork:
    
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
        CONTEXT_OVERFLOW = 10
    
    HTTP_ERROR_CONFIG = {
        500: ("Internal Server Error", 1),
        502: ("Bad Gateway", 1),
        503: ("Service Unavailable", 1),
        504: ("Gateway Time-out", 1),
        401: ("Unauthorized", 1),
        422: ("Unsupported Entity", 1),
        403: ("Forbidden", 1),
        429: ("Rate Limit Exceeded", 1),
    }
    
    class ContextOverflowError(RuntimeError):
        def __init__(self, status_code: int, message: str):
            self.status_code = status_code
            super().__init__(message)
    class GlobalTimeoutError(RuntimeError):
        def __init__(self, elapsed: float, timeout: float, message: str):
            self.elapsed = elapsed
            self.timeout = timeout
            super().__init__(message)
    @classmethod
    def get_cost_usage(cls) -> dict:
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/usage?evaluation_run_id={RUN_ID}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            usage_info = response.json()
            if isinstance(usage_info, dict):
                return usage_info
            else:
                print(f"get_cost_usage returned non-dict: {type(usage_info)}, returning default")
                return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {"used_cost_usd": 0, "max_cost_usd": float("inf")}
    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}
    @classmethod
    def make_request(
        cls, 
        messages: list[LlmMessage],
        model: Model = QWEN_MODEL,
        tool_mode: str = "none",
        tools: list = [],
        attempt: int = 1, 
        non_cycle_retries: int = 1,
        temperature: float = 0.0,
        check_timeout: Callable | None = None,
    ) -> tuple[str, list, Model]:
       
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        headers = {"Content-Type": "application/json"}
        
        cycle_model_list = [model]
        for m in AGENT_MODELS:
            if m != model:
                cycle_model_list.append(m)
        
        start_time = time.time()
        print(f"📡 [RUN_ID]: {RUN_ID} | Attempts: {attempt} | Model: {model.name} | Temperature: {temperature}")
        
        last_error = None
        last_error_type = "Unknown"
        if tool_mode == "required" and (tools is None or len(tools) == 0):
            raise ValueError("tools is not provided when tool_mode is required")
        
        for i in range(attempt):
            if check_timeout is not None:
                check_timeout()
            if i == 0 or last_error_type not in ["Request timeout", "HTTP error"]:
                model_idx = 0
            else:
                if i < non_cycle_retries:
                    model_idx = 0
                else:
                    model_idx = (i - non_cycle_retries + 1) % len(cycle_model_list)
            current_model = cycle_model_list[model_idx]
            request_data = {
                "evaluation_run_id": RUN_ID,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "model": current_model.name,
            }
            if tool_mode == "required" and len(tools) > 0:
                request_data["tools"] = tools
                request_data["tool_mode"] = "required"
            print(f"   ├── 🔂 Starting Request | Attempt: {i + 1}/{attempt} | Model: {current_model.name} | Timeout: {current_model.timeout}s")
            attempt_start = time.time()
            try:
                response = requests.post(
                    url,
                    json=request_data,
                    proxies={'http': '', 'https': ''},
                    timeout=(30, current_model.timeout),
                    headers=headers
                )
                response.raise_for_status()
                response_json = response.json()
                try:
                    content = response_json["content"]
                    tool_calls = response_json["tool_calls"]
                    
                    if (tool_mode == "none" and (content is None or content == "")):
                        print(response_json)
                        raise RuntimeError(f"Invalid Response, cannot find content from the llm response even though tool_mode is none")
                    elif (tool_mode == "required" and (tool_calls is None or len(tool_calls) == 0)):
                        print(response_json)
                        raise RuntimeError(f"Invalid Response, cannot find tool_calls from the llm response even though tool_mode is required")       
                
                    attempt_elapsed = time.time() - attempt_start
                    total_elapsed = time.time() - start_time
                    print(f"   ├── ✅ Request  Success | Attempt: {i + 1}/{attempt} | Model: {current_model.name} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                    return content, tool_calls, current_model
                except Exception as e:
                    last_error = e
                    last_error_type = "Response parsing error"
                    attempt_elapsed = time.time() - attempt_start
                    total_elapsed = time.time() - start_time
                    print(f"   ├── ❌ Invalid response for model {current_model.name}: {e} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                    sleep_time = 5 + min(i * 2, 10)
                    print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with same model...")
                    time.sleep(sleep_time)
                
            except JSONDecodeError as e:
                last_error = e
                last_error_type = "JSON parsing error"
                attempt_elapsed = time.time() - attempt_start
                total_elapsed = time.time() - start_time
                print(f"   ├── ❌ Invalid JSON response for model {current_model.name}: {e} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                sleep_time = 5 + min(i * 2, 10)
                print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with same model...")
                time.sleep(sleep_time)
                
            except requests.exceptions.Timeout as e:
                last_error = e
                last_error_type = "Request timeout"
                attempt_elapsed = time.time() - attempt_start
                total_elapsed = time.time() - start_time
                print(f"   ├── ❌ Request timeout after {current_model.timeout}s for model {current_model.name} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                _, sleep_time = cls.HTTP_ERROR_CONFIG.get(504, ("Timeout", 60))
                next_model_msg = "next model" if (i + 1) >= non_cycle_retries else "same model"
                print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with {next_model_msg}...")
                time.sleep(sleep_time)
                
            except requests.exceptions.ConnectionError as e:
                last_error = e
                last_error_type = "Connection error"
                attempt_elapsed = time.time() - attempt_start
                total_elapsed = time.time() - start_time
                print(f"   ├── ❌ Connection error for model {current_model.name}: {e} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                sleep_time = 10 + min(i * 5, 20)
                print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with same model...")
                time.sleep(sleep_time)
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code
                
                if status_code in [400, 413, 414, 431]:
                    error_msg = f"Context overflow error (HTTP {status_code}): Request too large. Status codes indicate the request exceeds context limits."
                    attempt_elapsed = time.time() - attempt_start
                    total_elapsed = time.time() - start_time
                    raise cls.ContextOverflowError(status_code, error_msg)
                
                error_title, sleep_time = cls.HTTP_ERROR_CONFIG.get(
                    status_code, ("HTTP Error", 10)
                )
                last_error_type = "HTTP error"
                attempt_elapsed = time.time() - attempt_start
                total_elapsed = time.time() - start_time
                print(f"   ├── ❌ HTTP {status_code} {error_title} for model {current_model.name} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                next_model_msg = "next model" if (i + 1) >= non_cycle_retries else "same model"
                print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with {next_model_msg}...")
                time.sleep(sleep_time)
                
            except requests.exceptions.RequestException as e:
                last_error = e
                last_error_type = "Request error"
                attempt_elapsed = time.time() - attempt_start
                total_elapsed = time.time() - start_time
                print(f"   ├── ❌ Request error for model {current_model.name}: {e} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                sleep_time = 10 + min(i * 5, 20)
                print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with same model...")
                time.sleep(sleep_time)
                
            except Exception as e:
                last_error = e
                last_error_type = f"Unexpected error ({type(e).__name__})"
                attempt_elapsed = time.time() - attempt_start
                total_elapsed = time.time() - start_time
                print(f"   ├── ❌ Unexpected error for model {current_model.name}: {e} | Attempt: {i + 1}/{attempt} | Timeout: {attempt_elapsed:.0f}s/{current_model.timeout}s | Total: {total_elapsed:.0f}s")
                sleep_time = 10
                print(f"   ├── 💤 Sleeping {sleep_time}s before retrying with same model...")
                time.sleep(sleep_time)
        
        tried_models = [model] * non_cycle_retries + [m for m in cycle_model_list if m != model][:max(0, attempt - non_cycle_retries)]
        raise RuntimeError(
            f"Failed after {attempt} attempts. Last error: {last_error_type} - {last_error}. "
            f"Tried models: {[m.name for m in tried_models[:attempt]]}"
        )
    @classmethod
    def _update_error_counter(cls, error_counter: dict[str, int], error_body: str) -> None:
        error_patterns = {
            cls.ErrorType.UNKNOWN.name: ["500", "Internal Server Error"],
            cls.ErrorType.NETWORK_ERROR.name: ["502", "Bad Gateway", "NETWORK_ERROR", "Connection error", "Connection refused", "Network unreachable", "Request error","network error", "network unreachable"],
            cls.ErrorType.RESOURCE_EXHAUSTED.name: ["503", "Service Unavailable", "RESOURCE_EXHAUSTED"],
            cls.ErrorType.TIMEOUT.name: ["504", "Gateway Time-out", "Gateway Timeout", "TIMEOUT", "Request timeout"],
            cls.ErrorType.INVALID_RESPONSE_FORMAT.name: ["422", "Unprocessable Entity", "Invalid JSON", "Invalid response", "Invalid parse_response"],
            cls.ErrorType.AUTHENTICATION_ERROR.name: ["401", "Unauthorized", "403", "Forbidden", "AUTHENTICATION_ERROR"],
            cls.ErrorType.RATE_LIMIT_EXCEEDED.name: ["429", "Rate Limit Exceeded", "RATE_LIMIT_EXCEEDED"],
            cls.ErrorType.RESERVED_TOKEN_PRESENT.name: ["RESERVED_TOKEN_PRESENT"],
            cls.ErrorType.EMPTY_RESPONSE.name: ["EMPTY_RESPONSE"],
        }
        
        error_map = {
            pattern: error_type
            for error_type, patterns in error_patterns.items()
            for pattern in patterns
        }
        
        for key, error_type in error_map.items():
            if key in error_body:
                error_counter[error_type] += 1
                return
        error_counter[cls.ErrorType.UNKNOWN.name] += 1
    
    @classmethod
    def _should_add_error_context(cls, error_body: str) -> bool:
        skip_errors = {"RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", "EMPTY_RESPONSE", "TIMEOUT", "AUTHENTICATION_ERROR"}
        return not any(err in error_body for err in skip_errors)
    
    @classmethod
    def inference(
        cls,
        messages: list[LlmMessage],
        model: Model = QWEN_MODEL,
        temperature: float = 0.0,
        max_attemps: int = 2,
        check_timeout: Callable | None = None,
    ) -> tuple[str, list[ToolCall], str, int, dict[str, int], Model]:
        error_counter = cls.get_error_counter()
        thought = "Processing request"
        tool_calls: list[ToolCall] = []
        raw_text = 'not defined'
        total_attempts = 0
        total_tokens = Utils.count_tokens(messages)
        print(f"   ├── 🛅 Input token size: {total_tokens} tokens (model={model.name}, messages={len(messages)})")
        for attempt in range(max_attemps):
            try:
                total_attempts += 1
                try:
                    raw_text, _, success_model = cls.make_request(
                        messages,
                        model,
                        attempt=5,
                        non_cycle_retries=2,
                        temperature=temperature,
                        check_timeout=check_timeout
                    )
                    start_time = time.time()
                    thought, tool_calls, parse_error = cls.parse_response(raw_text)
                    if parse_error:
                        raise ValueError(parse_error)
                    if not thought:
                        thought = "Processing request"
                    print(f"   └── ✅ Parsing  Success | Elapsed: {(time.time() - start_time):.2f}s | Model: {success_model.name}")
                    return thought, tool_calls, raw_text, total_attempts, error_counter, success_model
                
                except cls.GlobalTimeoutError:
                    raise
                except cls.ContextOverflowError as e:
                    print(f"[INFERENCE] ContextOverflowError in make_request (attempt {attempt + 1}), breaking retry loop: {e}")
                    raise
                except (ValueError, RuntimeError) as e:
                    error_body = str(e)
                    cls._update_error_counter(error_counter, error_body)
                    if attempt < max_attemps - 1:
                        print(f"Request error (attempt {attempt + 1}/{max_attemps}): {error_body[:200]} | Sleeping 10s before retry")
                        time.sleep(10)
                        continue
                    else:
                        raise RuntimeError(f"Failed after {max_attemps} attempts: {error_body}")
            except cls.GlobalTimeoutError:
                raise
            except cls.ContextOverflowError:
                raise
            except Exception as e:
                error_body = str(e)
                cls._update_error_counter(error_counter, error_body)
                if attempt < max_attemps - 1:
                    print(f"Error (attempt {attempt + 1}/{max_attemps}): {error_body[:200]} | Sleeping 10s before retry")
                    time.sleep(10)
                    continue
                raise RuntimeError(f"Failed after {max_attemps} attempts: {error_body}")
        raise RuntimeError(f"Inference failed: exhausted all {max_attemps} attempts without success")
    
    @classmethod
    def fix_json_string_with_llm(cls, json_string: str) -> dict | None:
        messages = [
            LlmMessage.system("Fix the json string sent by the user. Reply only with the json string and nothing else."),
            LlmMessage.user(json_string)
        ]
        try:
            response, _, _ = cls.make_request(messages, model=DEEPSEEK_MODEL, attempt=2)
            response = response.replace('```json', '').strip('```')
            return json.loads(response)
        except Exception:
            return None
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        pattern = ''
        for i, k in enumerate[str](arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'
        match=re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub(r"(^|\s)[\'\"]*thought[\'\"]*:", r"\1thought:", text_resp, flags=re.IGNORECASE | re.MULTILINE)
        text_resp = re.sub(r"(^|\s)[\'\"]*tool_call_(\d+)[\'\"]*:", r"\1tool_call_\2:", text_resp, flags=re.IGNORECASE | re.MULTILINE)
        text_resp = re.sub(r"(^|\s)[\'\"]*tool_name[\'\"]*:", r"\1tool_name:", text_resp, flags=re.IGNORECASE | re.MULTILINE)
        text_resp = re.sub(r"(^|\s)[\'\"]*tool_args[\'\"]*:", r"\1tool_args:", text_resp, flags=re.IGNORECASE | re.MULTILINE)
        text_resp = re.sub(r"(^|\s)[\'\"]*observation[\'\"]*:", r"\1observation:", text_resp, flags=re.IGNORECASE | re.MULTILINE)
        return text_resp
    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        if not isinstance(start_pos, int):
            return None
        if not text:
            return None
        if start_pos < 0 or start_pos >= len(text):
            return None
        if text[start_pos] != '{':
            return None
        MAX_DEPTH = 10000
        MAX_LENGTH = 1000000  # 1MB max for a single JSON object
        brace_depth = 0
        i = start_pos
        while i < len(text) and i - start_pos < MAX_LENGTH:
            if text[i] == '{':
                brace_depth += 1
                if brace_depth > MAX_DEPTH:
                    return None  # Too deeply nested
            elif text[i] == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_pos:i+1]
                if brace_depth < 0:
                    return None
            i += 1
        return None
    
    @classmethod
    def _parse_json_forgiving(cls, json_str: str) -> dict | None:
        if not json_str:
            return None
        if len(json_str) > 1000000:
            return None
        json_str = json_str.strip()
        if not json_str:
            return None
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        try:
            fixed = re.sub(r',\s*}', '}', json_str)
            fixed = re.sub(r',\s*]', ']', fixed)
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError, TypeError, re.error):
            pass
        try:
            if not re.search(r'[^a-zA-Z0-9_\[\]{}:,\s"\'\.\-]', json_str):
                if len(json_str) < 10000:
                    result = eval(json_str, {"__builtins__": {}}, {})
                    if isinstance(result, dict):
                        return result
        except (SyntaxError, NameError, TypeError, ValueError, Exception):
            pass
        try:
            if json_str.startswith('{'):
                pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', json_str)
                if pairs and len(pairs) > 0:
                    return dict(pairs)
                pairs = re.findall(r"'([^']+)'\s*:\s*'([^']*)'", json_str)
                if pairs and len(pairs) > 0:
                    return dict(pairs)
                pairs = re.findall(r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']*)["\']', json_str)
                if pairs and len(pairs) > 0:
                    return dict(pairs)
        except (re.error, ValueError, TypeError, Exception):
            pass
        return None
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        if not block:
            return None
        if len(block) > 100000:
            block = block[:100000]
        tool_name: str | None = None
        tool_args: dict = {}
        tool_args_found: bool = False
        tool_args_pos: int = -1
        args_patterns_balanced = [
            r'tool_args\s*:\s*(\{)',
            r'args\s*:\s*(\{)',
            r'arguments\s*:\s*(\{)',
        ]
        args_patterns_greedy = [
            r'tool_args\s*:\s*(\{.*?\})',
            r'args\s*:\s*(\{.*?\})',
            r'arguments\s*:\s*(\{.*?\})',
        ]
        try:
            for pattern in args_patterns_balanced:
                try:
                    match = re.search(pattern, block, re.IGNORECASE)
                    if match:
                        tool_args_pos = match.start()
                        break
                except (re.error, Exception):
                    continue
        except Exception:
            pass
        name_patterns = [
            r'tool_name\s*:\s*["\']?([^"\'\n\s]+(?:\s+[^"\'\n\s]+)*?)(?=["\']?\s*(?:tool_args|$))',
            r'tool_name\s*:\s*["\']?([^"\'\n]+)["\']?',
            r'name\s*:\s*["\']?([^"\'\n\s]+(?:\s+[^"\'\n\s]+)*?)(?=["\']?\s*(?:tool_args|$))',
        ]
        all_name_matches: list[tuple[int, str]] = []
        try:
            for pattern in name_patterns:
                try:
                    for match in re.finditer(pattern, block, re.IGNORECASE):
                        try:
                            name = match.group(1)
                            name = name.strip().strip('"').strip("'")
                            name = re.sub(r'\s*tool_args.*$', '', name).strip()
                            if name and len(name) < 100 and len(name) > 0:
                                all_name_matches.append((match.start(), name))
                        except (AttributeError, IndexError, Exception):
                            continue
                except (re.error, Exception):
                    continue
        except Exception:
            pass
        if tool_args_pos >= 0 and all_name_matches:
            best_match = None
            best_distance = float('inf')
            for pos, name in all_name_matches:
                if pos < tool_args_pos:
                    distance = tool_args_pos - pos
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name
            if best_match is None:
                for pos, name in all_name_matches:
                    distance = pos - tool_args_pos
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name
            if best_match:
                tool_name = best_match
        elif all_name_matches:
            tool_name = all_name_matches[-1][1]
        for pattern in args_patterns_balanced:
            try:
                match = re.search(pattern, block, re.IGNORECASE)
                if match:
                    brace_start = match.end() - 1
                    args_str = cls._extract_balanced_braces(block, brace_start)
                    if args_str:
                        parsed = cls._parse_json_forgiving(args_str)
                        if isinstance(parsed, dict):
                            tool_args_found = True
                            tool_args = parsed
                            break
            except (re.error, Exception):
                continue
        
        if not tool_args_found:
            for pattern in args_patterns_greedy:
                try:
                    match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
                    if match:
                        args_str = match.group(1)
                        parsed = cls._parse_json_forgiving(args_str)
                        if isinstance(parsed, dict):
                            tool_args_found = True
                            tool_args = parsed
                            break
                except (re.error, Exception):
                    continue
        
        if not tool_args_found:
            if re.search(r'tool_args\s*:', block, re.IGNORECASE):
                args_match = re.search(r'tool_args\s*:\s*\{', block, re.IGNORECASE)
                if args_match:
                    brace_start = args_match.end() - 1
                    if cls._extract_balanced_braces(block, brace_start):
                        pairs = re.findall(r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']+)["\']', block)
                        for key, value in pairs:
                            if key not in ['tool_name', 'name']:
                                tool_args[key] = value
                                tool_args_found = True
                if not tool_args_found:
                    if re.search(r'tool_args\s*:\s*\{\s*\}', block, re.IGNORECASE):
                        tool_args_found = True
        if tool_name and len(tool_name) > 0:
            return {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "_tool_args_found": tool_args_found
            }
        return None
    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, list[ToolCall], str]:
        if not isinstance(text_resp, str):
            return "", [], "Invalid response. Input is not a string"
        text_resp = text_resp.replace('\x00', '').replace('\r', '\n')
        MAX_INPUT_SIZE = 10 * 1024 * 1024
        if len(text_resp) > MAX_INPUT_SIZE:
            text_resp = text_resp[:MAX_INPUT_SIZE]
            return "", [], "Invalid response. Input too large (truncated)"
        text_resp = text_resp.strip()
        if not text_resp:
            return "", [], "Invalid response. Empty input"
        try:
            if "observation:" in text_resp.lower():
                text_resp = re.split(r'observation\s*:', text_resp, flags=re.IGNORECASE)[0].strip()
            text_resp = cls.sanitise_text_resp(text_resp)
        except Exception as e:
            return "", [], f"Error sanitizing input: {str(e)}"
        thought: str = ""
        thought_patterns = [
            r'thought\s*:\s*(.*?)(?=\n(?:tool_call|$))',
            r'thought\s*:\s*(.*?)(?=\ntool_call)',
            r'thought\s*:\s*(.*)',
        ]
        for pattern in thought_patterns:
            try:
                match = re.search(pattern, text_resp, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1)
                    thought = extracted.strip()
                    if thought and len(thought) > 2 and len(thought) < 100000:
                        break
            except (AttributeError, IndexError, re.error):
                continue
        tool_calls: list[dict] = []
        tool_call_positions: list[tuple[int, int, str]] = []
        try:
            tool_call_header_pattern = r'tool_call_(\d+)\s*:'
            for match in re.finditer(tool_call_header_pattern, text_resp, re.IGNORECASE):
                start = match.start()
                end = match.end()
                call_num = match.group(1)
                if start >= 0 and end > start:
                    tool_call_positions.append((start, end, call_num))
        except (re.error, AttributeError, IndexError):
            pass
        covered_positions: set[tuple[int, int]] = set()
        for i, position in enumerate(tool_call_positions):
            if not isinstance(position, tuple) or len(position) != 3:
                continue
            start, header_end, call_num = position
            if header_end < start or header_end > len(text_resp):
                continue
            block_end = len(text_resp)
            if i + 1 < len(tool_call_positions):
                next_start = tool_call_positions[i + 1][0]
                if next_start > header_end:
                    block_end = next_start
            else:
                try:
                    remaining_text = text_resp[header_end:min(header_end + 10000, len(text_resp))]
                    standalone_match = re.search(r'^tool_name\s*:', remaining_text, re.MULTILINE | re.IGNORECASE)
                    if standalone_match:
                        block_end = header_end + standalone_match.start()
                except (re.error, Exception):
                    pass
            
            if block_end > header_end and block_end <= len(text_resp):
                covered_positions.add((start, block_end))
                block = text_resp[header_end:block_end].strip()
                if block:
                    call_dict = cls._extract_tool_call_from_block(block)
                    if call_dict and "tool_name" in call_dict:
                        tool_calls.append(call_dict)
        name_patterns = [
            r'tool_name\s*:\s*["\']?([^"\'\n\s]+(?:\s+[^"\'\n\s]+)*?)(?=["\']?\s*(?:tool_args|$))',
            r'tool_name\s*:\s*["\']?([^"\'\n]+)["\']?',
            r'"tool_name"\s*:\s*"([^"]+)"',
            r"'tool_name'\s*:\s*'([^']+)'",
            r'tool_name\s*=\s*["\']([^"\']+)["\']',
        ]
        for pattern in name_patterns:
            try:
                for name_match in re.finditer(pattern, text_resp, re.IGNORECASE):
                    name_start = name_match.start()
                    if name_start < 0:
                        continue
                    is_inside = False
                    for start, end in covered_positions:
                        if start <= name_start < end:
                            is_inside = True
                            break
                    if is_inside:
                        continue
                    tool_name_raw = name_match.group(1)
                    tool_name = tool_name_raw.strip().strip('"').strip("'")
                    tool_name = re.sub(r'\s*tool_args.*$', '', tool_name).strip()
                    if not tool_name or len(tool_name) > 100:
                        continue
                    start_pos = name_match.end()
                    end_pos = min(start_pos + 1000, len(text_resp))
                    extended_context = text_resp[start_pos:end_pos]
                    tool_args: dict = {}
                    tool_args_found: bool = False
                    args_patterns_balanced = [r'tool_args\s*:\s*(\{)', r'args\s*:\s*(\{)', r'arguments\s*:\s*(\{)']
                    args_patterns_greedy = [r'tool_args\s*:\s*(\{.*?\})', r'args\s*:\s*(\{.*?\})', r'arguments\s*:\s*(\{.*?\})']
                    for args_pattern in args_patterns_balanced:
                        try:
                            args_match = re.search(args_pattern, extended_context, re.IGNORECASE)
                            if args_match:
                                brace_start = args_match.end() - 1
                                args_str = cls._extract_balanced_braces(extended_context, brace_start)
                                if args_str:
                                    parsed = cls._parse_json_forgiving(args_str)
                                    if isinstance(parsed, dict):
                                        tool_args_found = True
                                        tool_args = parsed
                                        break
                        except (re.error, Exception):
                            continue
                    if not tool_args_found:
                        for args_pattern in args_patterns_greedy:
                            try:
                                args_match = re.search(args_pattern, extended_context, re.DOTALL | re.IGNORECASE)
                                if args_match:
                                    args_str = args_match.group(1)
                                    parsed = cls._parse_json_forgiving(args_str)
                                    if isinstance(parsed, dict):
                                        tool_args_found = True
                                        tool_args = parsed
                                        break
                            except (re.error, Exception):
                                continue
                    if not tool_args_found:
                        if re.search(r'tool_args\s*:', extended_context, re.IGNORECASE):
                            if not tool_args:
                                tool_args = cls._extract_key_value_pairs_from_context(extended_context)
                                if tool_args:
                                    tool_args_found = True
                            if not tool_args_found and re.search(r'tool_args\s*:\s*\{\s*\}', extended_context, re.IGNORECASE):
                                tool_args_found = True
                    tool_calls.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "_tool_args_found": tool_args_found
                    })
            except (re.error, AttributeError, IndexError, Exception):
                continue
        
        if not tool_calls:
            brace_depth = 0
            start_pos = -1
            objects: list[str] = []
            MAX_DEPTH = 10000
            MAX_OBJECTS = 1000
            for i, char in enumerate(text_resp):
                if i > 1000000:
                    break
                if char == '{':
                    if brace_depth == 0:
                        start_pos = i
                    brace_depth += 1
                    if brace_depth > MAX_DEPTH:
                        break
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and start_pos != -1:
                        obj_str = text_resp[start_pos:i+1]
                        if len(obj_str) < 100000 and ('tool' in obj_str.lower() or 'name' in obj_str.lower()):
                            objects.append(obj_str)
                            if len(objects) >= MAX_OBJECTS:
                                break
                        start_pos = -1
                    if brace_depth < 0:
                        break
            for obj_str in objects[:MAX_OBJECTS]:
                call_dict = cls._extract_from_broken_json_object(obj_str)
                if call_dict and "tool_name" in call_dict:
                    tool_calls.append(call_dict)
        validated_calls: list[ToolCall] = []
        seen: set[tuple[str, str]] = set()
        skipped_calls: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                skipped_calls.append("Invalid call format")
                continue
            tool_name_raw = call.get("tool_name")
            if not tool_name_raw or not isinstance(tool_name_raw, str):
                skipped_calls.append("Missing tool_name")
                continue
            tool_name = tool_name_raw.strip()
            tool_name = ''.join(c for c in tool_name if c.isprintable() or c in ' \t\n').strip()
            if not tool_name or len(tool_name) > 100:
                skipped_calls.append("Invalid tool_name")
                continue
            tool_args_found = call.pop("_tool_args_found", None)
            tool_args_raw = call.get("tool_args")
            if not isinstance(tool_args_raw, dict):
                if isinstance(tool_args_raw, (list, tuple)) and len(tool_args_raw) == 0:
                    tool_args = {}
                    tool_args_found = True if tool_args_found is None else tool_args_found
                else:
                    skipped_calls.append("Invalid tool_args type")
                    continue
            else:
                tool_args = tool_args_raw
            
                if tool_args_found is None:
                    tool_args_found = True
            if tool_args_found is False and not tool_args:
                skipped_calls.append("Missing tool_args")
                continue
            try:
                key = (tool_name, json.dumps(tool_args, sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    validated_calls.append(ToolCall(tool_name=tool_name, tool_args=tool_args))
            except ValidationError as e:
                skipped_calls.append(f"ToolCall validation failed: {str(e)[:100]}")
                continue
            except (TypeError, ValueError) as e:
                skipped_calls.append("tool_args not JSON serializable")
                continue
            except Exception as e:
                skipped_calls.append(f"ToolCall construction failed: {type(e).__name__}")
                continue
        if validated_calls:
            if skipped_calls:
                print(f"Parse warning: Skipped {len(skipped_calls)} invalid tool call(s): {', '.join(skipped_calls[:5])}")
            return thought, validated_calls, ""
        if skipped_calls:
            error_msg = f"No valid tool calls found. Skipped {len(skipped_calls)} invalid call(s): {', '.join(skipped_calls[:3])}"
        else:
            error_msg = "No valid tool calls found in response. No tool_name patterns detected."
        try:
            safe_resp = text_resp[:500].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            print(f"Parse error: {error_msg}\nResponse: {safe_resp}")
        except Exception:
            print(f"Parse error: {error_msg}\nResponse: <encoding error>")
        return thought, [], error_msg
    
    @classmethod
    def _extract_key_value_pairs_from_context(cls, context: str) -> dict:
        result: dict = {}
        if not context:
            return result
        if len(context) > 100000:
            context = context[:100000]
        try:
            pattern1 = r'"([^"]+)"\s*:\s*"([^"]*)"'
            try:
                for match in re.finditer(pattern1, context):
                    try:
                        key = match.group(1)
                        value = match.group(2)
                        if key not in ['tool_name', 'name']:
                            result[key] = value
                    except (AttributeError, IndexError):
                        continue
            except (re.error, Exception):
                pass
            pattern2 = r"'([^']+)'\s*:\s*'([^']*)'"
            try:
                for match in re.finditer(pattern2, context):
                    try:
                        key = match.group(1)
                        value = match.group(2)
                        # regex group results are always str, no need to check isinstance
                        if key not in ['tool_name', 'name']:
                            result[key] = value
                    except (AttributeError, IndexError):
                        continue
            except (re.error, Exception):
                pass
        except Exception:
            pass
        return result
    
    @classmethod
    def _extract_from_broken_json_object(cls, json_str: str) -> dict | None:
        if not json_str:
            return None
        if len(json_str) > 100000:
            json_str = json_str[:100000]
        tool_name: str | None = None
        tool_args: dict = {}
        name_patterns = [
            r'"tool_name"\s*:\s*"([^"]+)"',
            r"'tool_name'\s*:\s*'([^']+)'",
            r'tool_name\s*:\s*["\']?([^"\'\s,}]+)["\']?',
        ]
        try:
            for pattern in name_patterns:
                try:
                    match = re.search(pattern, json_str, re.IGNORECASE)
                    if match:
                        try:
                            extracted = match.group(1)
                            # regex group(1) is always str, no need to check isinstance
                            tool_name = extracted.strip().strip('"').strip("'")
                            if tool_name and len(tool_name) < 100 and len(tool_name) > 0:
                                break
                        except (AttributeError, IndexError):
                            continue
                except (re.error, Exception):
                    continue
        except Exception:
            pass
        tool_args_found: bool = False
        args_patterns = [
            r'"tool_args"\s*:\s*(\{.*?\})',
            r"'tool_args'\s*:\s*(\{.*?\})",
            r'tool_args\s*:\s*(\{.*?\})',
        ]
        try:
            for pattern in args_patterns:
                try:
                    match = re.search(pattern, json_str, re.DOTALL | re.IGNORECASE)
                    if match:
                        try:
                            args_str = match.group(1)
                            parsed = cls._parse_json_forgiving(args_str)
                            if isinstance(parsed, dict):
                                tool_args_found = True
                                tool_args = parsed
                                break
                        except (AttributeError, IndexError):
                            continue
                except (re.error, Exception):
                    continue
        except Exception:
            pass
        if not tool_args_found:
            try:
                if re.search(r'tool_args\s*:', json_str, re.IGNORECASE):
                    if not tool_args:
                        try:
                            tool_args = cls._extract_key_value_pairs_from_context(json_str)
                            if isinstance(tool_args, dict):
                                tool_args.pop('tool_name', None)
                                tool_args.pop('name', None)
                                if tool_args:
                                    tool_args_found = True
                        except Exception:
                            pass
                    
                    if not tool_args_found:
                        try:
                            if re.search(r'tool_args\s*:\s*\{\s*\}', json_str, re.IGNORECASE):
                                tool_args_found = True
                        except Exception:
                            pass
            except Exception:
                pass
        if tool_name and len(tool_name) > 0:  # tool_name is already validated as str in extraction logic
            return {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "_tool_args_found": tool_args_found
            }
        return None
class ProblemParser:
    @classmethod
    def enhance_problem_statement(cls, problem_statement: str) -> str:
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
        for _ in range(2):
            try:
                messages = [
                    LlmMessage.system(ENHANCEMENT_PROMPT),
                    LlmMessage.user(f"Problem Statement:\n\n{problem_statement}"),
                ]
                enhanced, _, _ = EnhancedNetwork.make_request(
                    messages, model=QWEN_MODEL, attempt=5
                )
                return enhanced
            except Exception as e:
                print(f"Error: {e}")
        return ""
    @classmethod
    def get_problem_type(cls, problem_statement: str, enhancement: str) -> str:
        system_prompt = textwrap.dedent("""
        You are a helpful Problem Classifier to find a Task Name from PROJECT DESCRIPTION and project structure.
        Classify development tasks as either:
        - FIX: If the PROJECT DESCRIPTION is about fixing a bug, creating a new functionality or improving the existing codebase.
        - CREATE: If the PROJECT DESCRIPTION is about creating a new functionality from scratch.
        Output ONLY: "CREATE" or "FIX"
        """)
        instance_prompt = f"{problem_statement}\n# Enhanced Problem: \n{enhancement}"
        for _ in range(5):
            try:
                messages = [
                    LlmMessage.system(system_prompt),
                    LlmMessage.user(instance_prompt)
                ]
                response, _, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL)
                if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                    continue
                else:
                    return response
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(2)
        return PROBLEM_TYPE_FIX
    @classmethod   
    def analyze_problem(cls, problem_statement: str) -> tuple[str, str]:
        try:
            enhancement = cls.enhance_problem_statement(problem_statement)
            problem_type = cls.get_problem_type(problem_statement, enhancement)
            return problem_type, enhancement
        except Exception as e:
            print(f"Error in analyze_problem: {e}")
            return PROBLEM_TYPE_FIX, ""
class EnhancedCOT:
    class Action:
        def __init__(
            self, 
            thought: str, 
            tool_calls: list[ToolCall], 
            observation: str,
            is_error: bool = False,
            raw_response: str | None = None,
            total_attempts: int = 0,
            inference_error_counter: dict | None = None,
            request_data: list = None
        ):
            self.thought = thought
            self.tool_calls = tool_calls
            self.observation = observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter or {}
            self.request_data = request_data or []
            self.is_deleted = False
    def __init__(self, latest_observations_to_keep=10, summarize_batch_size=10):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        self.summaries: dict[tuple[int, int], str] = {}
        self.summarized_ranges: list[tuple[int, int]] = []
        self.max_summary_ranges_to_keep = 6  # Keep last 6 summary ranges for context
    def add_action(self, action: 'EnhancedCOT.Action') -> bool:
        self.thoughts.append(action)
        total_thoughts = len(self.thoughts)
        if total_thoughts >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True
    def _check_and_summarize_if_needed(self):
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
            return  # All messages before cutoff are already summarized or being kept
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
    
    def _summarize_messages_batch(self, start_idx: int, end_idx: int) -> Optional[str]:
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if thought.is_deleted:
                continue
            tool_calls_str = ""
            for j, tool_call in enumerate(thought.tool_calls, 1):
                tool_calls_str += f"tool_call_{j}: {tool_call.tool_name}({tool_call.tool_args})\n"
            assistant_part = f"thought: {thought.thought}\n{tool_calls_str}"
            obs_render = thought.observation if thought.observation else ""
            if len(obs_render) > 3000:
                obs_render = obs_render[:3000] + "... [truncated]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append({
                "assistant": assistant_part,
                "user": user_part,
                "is_error": thought.is_error
            })
        
        if not conversation_parts:
            return None
        conversation_text = ""
        for i, part in enumerate(conversation_parts, 1):
            conversation_text += f"\n--- Step {start_idx + i} ---\n"
            conversation_text += f"Assistant: {part['assistant']}\n"
            conversation_text += f"User: {part['user']}\n"
            if part['is_error']:
                conversation_text += "[Error occurred]\n"
        summarization_prompt = f"""You are summarizing a bug-fixing conversation history between an AI agent and its environment.
Summarize the following conversation steps concisely, focusing on:
1. Key actions taken (files read/edited, tests run, tools used)
2. Important findings or errors encountered
3. Progress made toward understanding/fixing the problem
4. Critical decisions or changes in approach
Keep the summary concise (1-2 sentences per step) but preserve important details like file names, test results, and key insights.
Conversation to summarize:
{conversation_text}
Provide a concise summary:"""
        try:
            messages = [
                LlmMessage.system("You are a helpful assistant that summarizes technical conversation history concisely while preserving important details."),
                LlmMessage.user(summarization_prompt)
            ]
            response, _, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL, temperature=0.0, attempt=2)
            return response.strip()
        except Exception as e:
            return None
    
    def _get_summary_for_index(self, idx: int) -> Optional[str]:
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None
    
    def to_str(self) -> list[LlmMessage]:
        messages: list[LlmMessage] = []
        last_summary_range = None
        allowed_summary_ranges = set(self.summarized_ranges[-self.max_summary_ranges_to_keep:]) if self.summarized_ranges else set()
        for i, action in enumerate(self.thoughts):
            if action.is_deleted:
                continue
            tool_calls_lines = []
            for j, tool_call in enumerate(action.tool_calls, 1):
                tool_calls_lines.append(
                    f"tool_call_{j}:\n  tool_name: {tool_call.tool_name}\n  tool_args: {json.dumps(tool_call.tool_args, ensure_ascii=False)}"
                )
            tool_args_str = "\n".join(tool_calls_lines) if tool_calls_lines else ""
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                summary = self._get_summary_for_index(i)
                if summary:
                    current_range = None
                    for (start, end), _ in self.summaries.items():
                        if start <= i < end:
                            current_range = (start, end)
                            break
                    
                    if current_range and current_range in allowed_summary_ranges:
                        if current_range != last_summary_range:
                            start, end = current_range
                            messages.append(
                                LlmMessage.user(f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]")
                            )
                            last_summary_range = current_range
                    continue
                assistant_str = (
                    f"thought:{action.thought}\n"
                    f"{tool_args_str}\n"
                )
                _obs_len = len(action.observation.splitlines()) if action.observation else 0
                user_str = f"{'Error occurred. ' if action.is_error else ''}Output omitted ({_obs_len} lines)\n"
            else:
                if action.is_error is None or i == len(self.thoughts) - 1:
                    assistant_str = f"thought:{action.thought}\n{tool_args_str}"
                    obs_render = action.observation if action.observation else ""
                    user_str = obs_render
                else:
                    if self.thoughts[-1].is_error == None and action.is_error != None:
                        assistant_str = (
                            f"thought:{action.thought}\n"
                            f"{tool_args_str}")
                        _obs_len = len(action.observation.splitlines()) if action.observation else 0
                        user_str = (
                            f"Error occurred. Detailed output omitted "
                            f"({_obs_len} lines)\n"
                        )
                    else:
                        assistant_str = f"thought:{action.thought}\n{tool_args_str}"
                        obs_render = action.observation if action.observation else ""
                        user_str = obs_render
            messages.append(LlmMessage.assistant(assistant_str))
            messages.append(LlmMessage.user(user_str))
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == LlmMessage.Role.USER:
                    messages[i].content = messages[i].content + "\n\n" + CRITICAL_RULES_INJECTION
                    break
        return messages
    def is_thought_repeated(self) -> bool:
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        last_tool_calls = last.tool_calls
        prev_tool_calls = prev.tool_calls
        last_repr = [(tc.tool_name, json.dumps(tc.tool_args, sort_keys=True)) for tc in last_tool_calls]
        prev_repr = [(tc.tool_name, json.dumps(tc.tool_args, sort_keys=True)) for tc in prev_tool_calls]
        if last_repr == prev_repr:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False
class EnhancedToolManager:
    TOOL_LIST: dict[str, dict[str, str|dict]] = {}
    created_test_files: list[str] = []
    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_TOOL_NAME=11
            INVALID_FILE_PATH=12
            INVALID_TOOL_CALL=13
            IMPORT_ERROR=14
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message
    
    @staticmethod
    def tool(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'tool_invocations'):
                self.tool_invocations = {}
            if fn.__name__ not in self.tool_invocations:
                self.tool_invocations[fn.__name__] = 0
            self.tool_invocations[fn.__name__] += 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                if not hasattr(self, 'tool_failure'):
                    self.tool_failure = {}
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {}
                if isinstance(e.error_type, EnhancedToolManager.Error.ErrorType):
                    error_type_name = e.error_type.name
                else:
                    error_type_name = str(e.error_type)
                if error_type_name not in self.tool_failure[fn.__name__]:
                    self.tool_failure[fn.__name__][error_type_name] = 0
                self.tool_failure[fn.__name__][error_type_name] += 1
                return e.message
        setattr(wrapper, 'is_tool', True)
        return wrapper
    
    def __init__(self, **kwargs):
        pass
    @classmethod
    
    def tool_parsing(cls, fn: Any) -> dict[str, Any]:
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0].rstrip()
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1).lstrip()
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if "list" in type_hint.lower() or "List" in type_hint:
                item_type_match = re.search(r'list\[([^\]]+)\]|List\[([^\]]+)\]', type_hint, re.IGNORECASE)
                if item_type_match:
                    item_type = item_type_match.group(1) or item_type_match.group(2)
                    item_type = item_type.strip()
                    
                    if 'str' in item_type.lower():
                        items_json_type = "string"
                    elif 'int' in item_type.lower():
                        items_json_type = "integer"
                    elif 'float' in item_type.lower() or 'number' in item_type.lower():
                        items_json_type = "number"
                    elif 'bool' in item_type.lower():
                        items_json_type = "boolean"
                    else:
                        items_json_type = "string"
                    
                    properties[param.name] = {
                        "type": "array",
                        "items": {"type": items_json_type},
                        "description": param_description
                    }
                    continue
            if 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint and 'list' not in type_hint.lower():
                json_type = "integer"
            elif 'float' in type_hint or 'number' in type_hint.lower():
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        return tool_schemas
    def get_tool_docs(self) -> str:
        tool_docs = f"# Available Tools ({len(self.TOOL_LIST)} tools)\n\n"
        for idx, (tool_name, tool_metadata) in enumerate(self.TOOL_LIST.items(), 1):
            description = tool_metadata.get('description', '').strip()
            input_schema = tool_metadata.get('input_schema', {})
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
            tool_docs += f"## {idx}. {tool_name}\n\n"
            tool_docs += f"{description}\n\n"
            tool_docs += "### detailed parameter schema\n\n"
            tool_docs += "```json\n"
            tool_docs += json.dumps(input_schema, indent=1)
            tool_docs += "\n```\n\n"
            example_args = {}
            for param_name in required:
                param_info = properties.get(param_name, {})
                param_type = param_info.get('type', 'string')
                if param_type == "string":
                    example_args[param_name] = "..."
                elif param_type == "boolean":
                    example_args[param_name] = True
                elif param_type == "integer":
                    example_args[param_name] = random.randint(1, 10)
                elif param_type == "array":
                    example_args[param_name] = []
                else:
                    example_args[param_name] = "..."
            tool_docs += "### **Usage format**\n\n"
            tool_docs += "```\n"
            tool_docs += f"tool_call_1:\n"
            tool_docs += f"    tool_name: {tool_name}\n"
            tool_docs += "    tool_args: {{"
            args_parts = []
            for key, value in example_args.items():
                if isinstance(value, str):
                    args_parts.append(f'"{key}": "..."')
                elif isinstance(value, bool):
                    args_parts.append(f'"{key}": {str(value)}')
                elif isinstance(value, int):
                    args_parts.append(f'"{key}": {value}')
                elif isinstance(value, list):
                    args_parts.append(f'"{key}": []')
                else:
                    args_parts.append(f'"{key}": "..."')
            tool_docs += ", ".join(args_parts)
            tool_docs += "}}\n"
            tool_docs += "```\n\n"
        return tool_docs
    def get_tool(self, tool_name: str) -> Callable[..., Any]:
        tool_method = getattr(self, tool_name, None)
        if tool_method is None:
            raise ValueError(f"❌ Invalid Tool: tool '{tool_name}' function not found in the tool manager")
        elif not callable(tool_method):
            raise ValueError(f"❌ Invalid Tool: tool '{tool_name}' function is not a callable")
        return tool_method
    
    def validate_tool_args(self, tool_args: dict[str, Any], doc: dict[str, Any]) -> tuple[bool, str]:
        input_schema = cast(dict[str, Any], doc.get("input_schema", {}))
        if not input_schema or input_schema.get("type") != "object":
            return True, ""
        properties = cast(dict[str, Any], input_schema.get("properties", {}))
        required = cast(list[str], input_schema.get("required", []))
        errors = []
        for param_name in required:
            if param_name not in tool_args:
                errors.append(f"Required keyword argument '{param_name}' is missing")
            else:
                value = tool_args[param_name]
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    errors.append(f"Required keyword argument '{param_name}' cannot be empty string or null")
        all_properties = set[str](properties.keys())
        provided_args = set[str](tool_args.keys())
        unexpected_args = provided_args - all_properties
        if unexpected_args:
            for arg in sorted(unexpected_args):
                errors.append(f"Unexpected keyword argument '{arg}' is not in schema")
        for param_name, value in tool_args.items():
            if param_name not in properties:
                continue
            prop_schema = properties[param_name]
            expected_type = prop_schema.get("type", "string")
            actual_type = type(value).__name__
            type_valid = False
            if expected_type == "string":
                type_valid = isinstance(value, str)
            elif expected_type == "integer":
                type_valid = isinstance(value, int)
            elif expected_type == "number":
                type_valid = isinstance(value, (int, float))
            elif expected_type == "boolean":
                type_valid = isinstance(value, bool)
            elif expected_type == "array":
                type_valid = isinstance(value, list)
                if type_valid:
                    items_schema = prop_schema.get("items", {})
                    if items_schema:
                        item_type = items_schema.get("type", "string")
                        invalid_items = []
                        for i, item in enumerate[Any](value):
                            item_valid = False
                            if item_type == "string":
                                item_valid = isinstance(item, str)
                            elif item_type == "integer":
                                item_valid = isinstance(item, int)
                            elif item_type == "number":
                                item_valid = isinstance(item, (int, float))
                            elif item_type == "boolean":
                                item_valid = isinstance(item, bool)
                            else:
                                item_valid = True
                            if not item_valid:
                                invalid_items.append(i)
                        if invalid_items:
                            errors.append(
                                f"'{param_name}' is array, but items at indices {invalid_items[:5]}{'...' if len(invalid_items) > 5 else ''} "
                                f"are not {item_type} (got {type(value[invalid_items[0]]).__name__ if invalid_items else 'unknown'})"
                            )
            else:
                type_valid = True
            if not type_valid:
                errors.append(f"'{param_name}' is {expected_type}, but {actual_type} provided")
        if errors:
            error_msg = "❌ Invalid args:\n"
            for i, error in enumerate[Any](errors, 1):
                error_msg += f"{i}. {error}\n"
            error_msg += f"*check this tool's doc: {doc}"
            return False, error_msg.strip()
        return True, ""
    def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        try:
            tool = self.TOOL_LIST.get(tool_name, None)
            if tool is None:
                _available_tools = list[Any](self.TOOL_LIST.keys())
                raise ValueError(f"❌ Invalid Tool Name: tool '{tool_name}' not found in the whitelist. Only {len(_available_tools)} tools are whitelisted: {_available_tools}")
            tool_method = self.get_tool(tool_name)
            is_valid, error_msg = self.validate_tool_args(tool_args, tool)
            if not is_valid:
                raise ValueError(error_msg)
            result = tool_method(**tool_args) if tool_args else tool_method()
            return result
        except (Exception, EnhancedToolManager.Error) as e:
            return f"Error executing tool `{tool_name}`: {e}"
    def get_final_git_patch(self) -> str:
        Utils.delete_files(self.created_test_files)
        try:
            command = f"""
            shopt -s globstar
            cp .gitignore .gitignore.backup 2>/dev/null || true
            
            git add * 2>/dev/null || true
            git diff --cached > .patch.txt
            cat .patch.txt
            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            print("Generating git patch...")
            result = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            print(f"git patch stdout:\n{result.stdout}")
            print(f"git patch stderr:\n{result.stderr}")
            return result.stdout
        except Exception as e:
            print(f"Error generating git patch: {e}")
            return ""
class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(self, available_tools: Optional[list[str]] = []):
        self.thought_history: list[dict[str, Any]] = []
        self.branches: dict[str, list[dict[str, Any]]] = {}
        
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        
        if available_tools is not None and len(available_tools) > 0:
            missing_tools = []
            for tool_name in available_tools:
                if tool_name not in self.TOOL_LIST:
                    missing_tools.append(tool_name)
            if missing_tools:
                error_msg = f"❌ The following tools requested in 'available_tools' were not found in TOOL_LIST:\n"
                error_msg += f"  Missing tools: {', '.join(missing_tools)}\n"
                error_msg += f"  Tools in TOOL_LIST: {', '.join(sorted(self.TOOL_LIST.keys()))}\n"
                error_msg += f"  Please check:\n"
                error_msg += f"    1. Tool names are spelled correctly\n"
                error_msg += f"    2. Tools are properly decorated with @EnhancedToolManager.tool\n"
                error_msg += f"    3. Tools are defined in tool manager class itself or its parent classes"
                print(error_msg)
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_NAME, error_msg)
        self.tool_failure = {
            k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }
        self.tool_invocations = {
          k:0 for k in self.TOOL_LIST.keys()
        }
    
    @EnhancedToolManager.tool
    def bash(self, command: str, is_test_command: bool = False) -> str:
        """
Run commands in a bash shell with optional test result parsing.
State is persistent across command calls. Please avoid commands that may produce very large output.
WHEN TO USE:
- Exploring repository structure: ls, find, tree commands
- Finding files: find . -name 'pattern' -type f
- Searching code: grep -rn 'pattern' directory/
- Running commands: [runner] [file_path] [arguments], etc.
- Environment setup: export VAR=value, cd directory, source activate
CRITICAL: Do not try to install packages because you cannot access Internet
EFFICIENCY TIP: This tool is ideal for batch operations to reduce tool call overhead.
- Read multiple files at once: head -300 [file1_path] [file2_path] [file3_path]
- Search across multiple files: grep -n '[pattern]' [file1_path] [file2_path] [file3_path] | head -300
- Combine file operations: for f in [file1_path] [file2_path]; do echo "=== $f ==="; head -300 $f; done
- Chain operations: command1 && command2 or command1; command2
- For further reading after first 300 lines: sed -n '301,600p' [file_path] (lines 301-600), sed -n '601,900p' (lines 601-900), etc.
FILE READING GUIDELINE: Always use head -300 for first 300 lines, then use sed for further reading if needed.
- First 300 lines: head -300 [file_path] (reads first 300 lines, or all lines if file is smaller)
- Further reading (lines 301-600): sed -n '301,600p' [file_path]
- Further reading (lines 601-900): sed -n '601,900p' [file_path]
- No need to check file size first - head -300 works for any file size
- For search results: grep -n '[pattern]' [file_path] | head -300
- For command output: [command] | head -300
- Examples:
    * First 300 lines: head -300 file_path
    * Next 300 lines (301-600): sed -n '301,600p' file_path
    * Multiple files: head -300 file1_path file2_path (shows first 300 lines of each)
    * Search results: grep -r 'pattern' . | head -300 (limit to 300 lines)
SMART TEST PARSING: When running test commands, set is_test_command to true 
to get intelligent parsing showing only failed tests and summary.
Output: Command output (raw or parsed based on is_test_command flag)
Arguments:
    command: The bash command to run. Use this for batch operations to minimize tool calls.
    is_test_command: Set to true when running test commands to get parsed output showing only failed tests
"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=60
            )
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            output = output.strip()
            if not output:
                return "No output"
            if is_test_command:
                formatted_output = f"Test Command: {command}\n\n"
                formatted_output += output
                return formatted_output
            return output
        except subprocess.TimeoutExpired:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.TIMEOUT,
                f"Command timed out after 60 seconds"
            )
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR,
                f"Error executing command: {str(e)}"
            )
    @EnhancedToolManager.tool
    def create_file(self, path: str, file_text: str) -> str:
        """
Create a new file with the specified content. Cannot be used if the file already exists and is not empty.
Output: Success message with file path
Arguments:
    path: Path where the file should be created
    file_text: Content to write to the file
"""
        try:
            file_path = Path(path).resolve()
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                        f"File already exists and is not empty at: {path}. Cannot overwrite."
                    )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_text)
            self.created_test_files.append(str(file_path))
            return f"File created successfully at: {path}"
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR,
                f"Error creating file: {str(e)}"
            )
    @EnhancedToolManager.tool
    def str_replace_in_file(self, path: str, old_str: str, new_str: str = "") -> str:
        """
Replace exact text in a file. Searches for old_str in the entire file and replaces it with new_str.
The old_str must appear exactly ONCE in the file - if found zero times, returns error with full file content.
If found multiple times, returns error showing line numbers of all matches to prevent accidental mass replacement.
Works for both single-line and multi-line strings. Whitespace, indentation, and newlines must match exactly.
To replace multiple occurrences, call this function multiple times with more specific old_str that includes surrounding context.
WHEN TO USE:
- Fixing bugs: Replace incorrect code with corrected version
- Adding functionality: Replace existing method/function with enhanced version
- Updating values: Change configuration values, constants, default parameters
- Refactoring: Rename variables, update method signatures
EFFICIENCY TIP:
- Batch multiple str_replace_in_file calls in ONE response
- MUST follow with test execution in SAME response (test immediately after edit)
- Include 3-5 lines of context in old_str for uniqueness
- Use multi-line old_str for complex changes (preserves formatting)
Output: On success: Returns diff-like view showing 5 lines before and after the change with line numbers. Old lines marked with "-" (removed) and new lines marked with "+" (added), following standard git diff format. On error: Returns detailed error message. If old_str not found, includes full file content with line numbers to help identify the correct string.
Arguments:
    path: Path to the file to edit
    old_str: The exact string to replace. Must appear exactly once in the file. Can span multiple lines. Must match whitespace exactly.
        CRITICAL: Use ACTUAL newline characters, NOT escaped \\n strings!
        CORRECT (in JSON): "line1\nline2" → becomes actual newlines in Python
        WRONG: "line1\\nline2" → searches for literal backslash-n characters
        If old_str not found, read file first with: bash head -300 [path]
    new_str: The new string to replace with. Default is empty string which deletes old_str.
"""
        try:
            file_path = Path(path).resolve()
            
            if not file_path.exists():
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND,
                    f"File does not exist: {path}"
                )
            conversion_note = ""
            if old_str and "\\n" in old_str and "\n" not in old_str:
                # Convert escaped newlines to actual newlines
                old_str = old_str.replace("\\n", "\n")
                conversion_note = "Note: Your provided text had \\n (escaped newlines) instead of actual newlines, so I converted them and proceeded with the replacement.\n\n"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not old_str.strip():
                if content.strip():
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                        "old_str is empty but file is not empty"
                    )
                new_content = new_str
            else:
                occurrences = content.count(old_str)
                if occurrences == 0:
                    try:
                        # Limit to 50 lines instead of 300 to reduce token usage
                        file_content_50_lines = Utils.get_code_with_line_numbers(str(file_path), view_range=[1, 50])
                        total_lines = len(content.split('\n'))
                        error_msg = (
                            f"Error: old_str not found in {path}\n\n"
                            f"File content (first 50 lines of {total_lines} total, with line numbers):\n"
                            f"{file_content_50_lines}\n\n"
                            f"... (file has {total_lines} total lines)\n"
                            f"Please check the file content above and provide the exact string to replace.\n"
                            f"To read more lines, use: bash sed -n '51,100p' {path} (for lines 51-100), or bash sed -n '101,150p' {path} (for lines 101-150), etc."
                        )
                    except Exception as e:
                        error_msg = (
                            f"Error: old_str not found in {path}\n\n"
                            f"File exists but could not read content: {str(e)}"
                        )
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND,
                        error_msg
                    )
                elif occurrences > 1:
                    line_numbers = []
                    start_pos = 0
                    while True:
                        pos = content.find(old_str, start_pos)
                        if pos == -1:
                            break
                        text_before_match = content[:pos]
                        line_num = text_before_match.count("\n") + 1
                        line_numbers.append(line_num)
                        start_pos = pos + len(old_str)  # Move past entire match to find next one
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND,
                        f"Multiple occurrences ({occurrences}) of old_str found starting at lines {line_numbers}. Please ensure it is unique by providing more context in old_str."
                    )
                new_content = content.replace(old_str, new_str)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            old_lines = content.split("\n")
            new_lines = new_content.split("\n")
            if old_str in content:
                before_old = content.split(old_str)[0]
                old_start_line_idx = before_old.count("\n")
                old_str_line_count = old_str.count("\n") + (0 if old_str.endswith("\n") else 1)
                old_end_line_idx = old_start_line_idx + old_str_line_count - 1
            else:
                old_start_line_idx = 0
                old_end_line_idx = 0
            new_start_line_idx = old_start_line_idx
            new_str_line_count = new_str.count("\n") + (0 if new_str.endswith("\n") else 1)
            new_end_line_idx = new_start_line_idx + new_str_line_count - 1
            diff_lines = []
            context_start = max(0, old_start_line_idx - 5)
            for i in range(context_start, old_start_line_idx):
                if i < len(old_lines):
                    line_num = i + 1
                    diff_lines.append(f"{line_num:4}|   {old_lines[i]}")
            for i in range(old_start_line_idx, min(old_end_line_idx + 1, len(old_lines))):
                line_num = i + 1
                diff_lines.append(f"{line_num:4}| - {old_lines[i]}")
            for i in range(new_start_line_idx, min(new_end_line_idx + 1, len(new_lines))):
                line_num = i + 1
                diff_lines.append(f"{line_num:4}| + {new_lines[i]}")
            context_end = min(len(new_lines), new_end_line_idx + 1 + 5)
            for i in range(new_end_line_idx + 1, context_end):
                if i < len(new_lines):
                    line_num = i + 1
                    diff_lines.append(f"{line_num:4}|   {new_lines[i]}")
            diff_view = "\n".join(diff_lines)
            success_msg = f"File {path} edited successfully.\n\n"
            if conversion_note:
                success_msg += conversion_note
            success_msg += f"Changes (showing 5 lines before and after):\n{diff_view}\n\nReview and edit again if needed."
            return success_msg
        except EnhancedToolManager.Error:
            raise
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR,
                f"Error editing file: {str(e)}"
            )
    @EnhancedToolManager.tool
    def view_file(self, path: str, view_range: Optional[list[int]] = None) -> str:
        """
View the contents of a file or directory. If path is a file, displays the result of applying cat -n.
If path is a directory, lists non-hidden files and directories up to 2 levels deep.
Arguments:
    path: Path to file or directory
    view_range: Optional list of two integers [start_line, end_line] to show specific line range. Use [start, -1] to show from start to end of file
Output:
    File content with line numbers or directory listing
"""
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND,
                    f"The path {path} does not exist"
                )
            if file_path.is_dir():
                if view_range:
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                        "view_range parameter is not allowed when path points to a directory"
                    )
                result = subprocess.run(
                    ["find", str(file_path), "-maxdepth", "2", "-not", "-path", "*/.*"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    return f"Files and directories up to 2 levels deep in {path}:\n{result.stdout}"
                else:
                    return f"Error listing directory: {result.stderr}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            file_lines = file_content.split("\n")
            total_lines = len(file_lines)
            init_line = 1
            
            if view_range:
                if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                        "Invalid view_range. It should be a list of two integers"
                    )
                start, end = view_range
                if start < 1 or start > total_lines:
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                        f"Invalid view_range start {start}. Should be within [1, {total_lines}]"
                    )
                init_line = start
                if end == -1:
                    file_content = "\n".join(file_lines[start - 1:])
                else:
                    if end > total_lines or end < start:
                        raise EnhancedToolManager.Error(
                            EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                            f"Invalid view_range end {end}"
                        )
                    file_content = "\n".join(file_lines[start - 1:end])
            numbered_content = "\n".join([
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ])
            return f"Content of {path}:\n{numbered_content}\n\nTotal lines: {total_lines}"
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND,
                f"Error reading file: {str(e)}"
            )
    @EnhancedToolManager.tool
    def sequential_thinking(
        self, 
        thought: str, 
        thoughtNumber: int, 
        totalThoughts: int, 
        nextThoughtNeeded: bool,
        isRevision: bool = False,
        revisesThought: int | None = None,
        branchFromThought: int | None = None,
        branchId: str | None = None,
        needsMoreThoughts: bool = False
    ) -> str:
        """
A detailed tool for dynamic and reflective problem-solving through thoughts.
This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.
When to use this tool:
- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially
- Problems that require a multi-step solution
- Tasks that need to maintain context over multiple steps
- Situations where irrelevant information needs to be filtered out
Key features:
- You can adjust total_thoughts up or down as you progress
- You can question or revise previous thoughts
- You can add more thoughts even after reaching what seemed like the end
- You can express uncertainty and explore alternative approaches
- Not every thought needs to build linearly - you can branch or backtrack
- Generates a solution hypothesis
- Verifies the hypothesis based on the Chain of Thought steps
- Repeats the process until satisfied
- Provides a correct answer
Output: JSON response with thoughtNumber, totalThoughts, nextThoughtNeeded, branches, and thoughtHistoryLength
Arguments:
    thought: Your current thinking step (can include analysis, breaking down complex parts, questioning assumptions, exploring alternatives, revising previous thoughts, synthesizing information, making connections, identifying patterns, formulating hypotheses, planning next steps)
    thoughtNumber: The current thought number (1-based)
    totalThoughts: How many thoughts you plan to have (can be adjusted)
    nextThoughtNeeded: Set to true if you want to continue with another thought
    isRevision: Set to true if this thought revises a previous one
    revisesThought: If isRevision is true, specify which thought number this revises
    branchFromThought: If this thought branches from a previous one, specify the thought number
    branchId: A unique identifier for this branch
    needsMoreThoughts: Set to true if you need more thoughts after this one
"""
        try:
            if not thought or not isinstance(thought, str):
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                    "Invalid thought: must be a non-empty string"
                )
            if not isinstance(thoughtNumber, int) or thoughtNumber < 1:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                    "Invalid thoughtNumber: must be a positive integer"
                )
            if not isinstance(totalThoughts, int) or totalThoughts < 1:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                    "Invalid totalThoughts: must be a positive integer"
                )
            if not isinstance(nextThoughtNeeded, bool):
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                    "Invalid nextThoughtNeeded: must be a boolean"
                )
            thought_data = {
                "thought": thought,
                "thoughtNumber": thoughtNumber,
                "totalThoughts": totalThoughts,
                "nextThoughtNeeded": nextThoughtNeeded,
                "isRevision": isRevision,
                "revisesThought": revisesThought,
                "branchFromThought": branchFromThought,
                "branchId": branchId,
                "needsMoreThoughts": needsMoreThoughts,
            }
            if thoughtNumber > totalThoughts:
                thought_data["totalThoughts"] = thoughtNumber
                totalThoughts = thoughtNumber
            self.thought_history.append(thought_data)
            if branchFromThought and branchId:
                if branchId not in self.branches:
                    self.branches[branchId] = []
                self.branches[branchId].append(thought_data)
            response = {
                "thoughtNumber": thoughtNumber,
                "totalThoughts": totalThoughts,
                "nextThoughtNeeded": nextThoughtNeeded,
                "branches": list[str](self.branches.keys()),
                "thoughtHistoryLength": len(self.thought_history),
            }
            return json.dumps(response, indent=2)
        except EnhancedToolManager.Error:
            raise
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR,
                f"Error processing thought: {str(e)}"
            )
    @EnhancedToolManager.tool
    def finish(self, investigation_summary: str):
        """
Signals completion of the current workflow execution
Output: Nothing to return
Arguments:
    investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem. Use the following format:
        Problem: <problem_statement>
        Investigation: <investigation_summary>
        Solution: <your solution>
"""
        return "finish"
    @EnhancedToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
Lists the directory structure of the repository
Arguments:
    directory_path: the directory path to list (default: ".")
    max_depth: maximum depth to traverse (default: 1)
"""
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
        
        result_lines = tree(directory_path, "", 0, max_depth)
        return "\n".join(result_lines) if result_lines else f"{directory_path}/"
    @EnhancedToolManager.tool
    def finish_find_files_to_fix(self, files: List[str]):
        """
Signals completion of the file finding workflow execution
Output: Nothing to return
Arguments:
    files: The list of files to fix.
"""
        if not hasattr(self, 'files_to_fix'):
            self.files_to_fix = []
        self.files_to_fix = files
        return files
def execute_agent_workflow(
    tool_manager: FixTaskEnhancedToolManager,
    system_prompt,
    instance_prompt,
    *,
    finish_tool_name: str = "finish",
    timeout: int = DEFAULT_TIMEOUT,
    n_max_steps = MAX_FIX_TASK_STEPS,
    logger_prefix: str = "FIX_TASK",
    model: Model = QWEN_MODEL
) -> str:
    start_time = time.time()
    
    def check_timeout():
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"[{logger_prefix}] [CRITICAL] Workflow timeout reached: {elapsed_time:.2f}s > {timeout:.2f}s")
            cot.add_action(EnhancedCOT.Action(
                thought="global timeout reached",
                tool_calls=[],
                observation="",
                is_error=True,
                inference_error_counter={},
                request_data=[]
            ))
            raise EnhancedNetwork.GlobalTimeoutError(elapsed_time, float(timeout), f"Workflow execution exceeded timeout")
    
    cot = EnhancedCOT(latest_observations_to_keep=15, summarize_batch_size=5)
    
    for step in range(n_max_steps):
        elapsed_time = time.time() - start_time
        temperature = 0.0
        selected_model = model
        cost_usage = EnhancedNetwork.get_cost_usage()
        print(f"[{logger_prefix}] {step + 1}/{n_max_steps} | {elapsed_time:.1f}s/{timeout:.1f}s | ${cost_usage.get('used_cost_usd', 0):.4f}/${cost_usage.get('max_cost_usd', 0):.4f}")
        messages: list[LlmMessage] = [
            LlmMessage.system(system_prompt),
            LlmMessage.user(instance_prompt),
        ]
        messages.extend(cot.to_str())
        if cot.is_thought_repeated():
            last_action = cot.thoughts[-1] if cot.thoughts else None
            if last_action:
                prev_tool_calls = last_action.tool_calls
                prev_tool_calls_str = "\n".join([
                    f"tool_call_{i+1}:\n  tool_name: {call.tool_name}\n"
                    f"  tool_args: {json.dumps(call.tool_args, ensure_ascii=False)}"
                    for i, call in enumerate[ToolCall](prev_tool_calls)
                ])
                previous_response = f"thought: {last_action.thought}\n{prev_tool_calls_str}"
                messages.append(
                    LlmMessage.user(DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=previous_response))
                )
            print(f"[ADAPTIVE] Thought repeated {cot.repeated_thoughts} times")
            if cot.repeated_thoughts == 1:
                temperature = 0.2
            elif cot.repeated_thoughts == 2:
                temperature = 0.4
            elif cot.repeated_thoughts == 3:
                temperature = 0.6
            else:
                temperature = min(0.7 + (cot.repeated_thoughts - 3) * 0.05, 0.9)
            
            if cot.repeated_thoughts >= 2:
                model_idx = (cot.repeated_thoughts - 2) % len(AGENT_MODELS)
                selected_model = AGENT_MODELS[model_idx]
        elif step > 150 and step % 50 == 0:
            temperature = 0.3
        raw_text = ""
        total_attempts = 0
        error_counter = {}
        try:
            thought, tool_calls, raw_text, total_attempts, error_counter, success_model = EnhancedNetwork.inference(
                messages, 
                model=selected_model, 
                temperature=temperature,
                check_timeout=check_timeout
            )
            if selected_model != success_model and step < 1:
                continue
            observations = []
            has_finish = False
            has_error = False
            for i, tool_call in enumerate[ToolCall](tool_calls):
                tool_name = tool_call.tool_name
                tool_args = tool_call.tool_args
                if not tool_name:
                    observations.append(f"{i+1}. Error - missing tool_name")
                    has_error = True
                    continue
                tool_args_str = ", ".join([str(v) for k, v in tool_args.items()])
                tool_call_str = f"{tool_name}({tool_args_str})" if tool_args_str else f"{tool_name}()"
                try:
                    observation = tool_manager.execute_tool(tool_name, tool_args)
                    observations.append(f"\n{i+1}. success to run `{tool_call_str}`\n{observation}")
                    if tool_name == finish_tool_name:
                        has_finish = True
                except Exception as tool_error:
                    error_msg = f"\n{i+1}. fail to run `{tool_call_str}`\n{str(tool_error)}"
                    print(f"Tool execution failed: {tool_error}")
                    error_msg = f"{error_msg}\n{traceback.format_exc()}"
                    observations.append(error_msg)
                    has_error = True
            combined_observation = f"Executed {len(tool_calls)} function tools\n" + "\n".join(observations)
            print(f"[{logger_prefix}] [CRITICAL] {combined_observation}")
            cot.add_action(EnhancedCOT.Action(
                thought=thought, 
                tool_calls=tool_calls, 
                observation=combined_observation, 
                is_error=has_error, 
                raw_response=raw_text, 
                total_attempts=total_attempts, 
                inference_error_counter=error_counter, 
                request_data=messages
            ))
            if has_finish:
                print(f"[{logger_prefix}] Workflow called finish operation")
                break
        except EnhancedNetwork.GlobalTimeoutError as e:
            print(f"[{logger_prefix}] [CRITICAL] Workflow timeout reached: {e.elapsed:.2f}s > {e.timeout:.2f}s: {e}")
            break
        except EnhancedNetwork.ContextOverflowError as e:
            current_keep = cot.latest_observations_to_keep
            current_batch = cot.summarize_batch_size
            new_keep = max(5, int(current_keep * 0.8))
            new_batch = min(20, int(current_batch * 1.25))
            cot.latest_observations_to_keep = new_keep
            cot.summarize_batch_size = new_batch
            cot._check_and_summarize_if_needed()
            continue
        except Exception as e:
            error_msg = f"\n\nERROR: {repr(e)}\n{traceback.format_exc()}"
            print(f"[{logger_prefix}] [STEP {step + 1}] Inference error: {error_msg}")
            print(f"[{logger_prefix}] [STEP {step + 1}] Continuing to next step after error...")
            cot.add_action(EnhancedCOT.Action(
                thought=error_msg, 
                tool_calls=[], 
                observation="", 
                is_error=True, 
                raw_response=raw_text, 
                total_attempts=total_attempts, 
                inference_error_counter=error_counter, 
                request_data=messages
            ))
            continue
    else:
        cot.add_action(EnhancedCOT.Action(
            thought="global timeout reached", 
            tool_calls=[],
            observation="", 
            is_error=True
        ))
        print(f"[{logger_prefix}] [CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
def process_fix_task(
    problem_statement: str,
    timeout: int = DEFAULT_TIMEOUT - 200,
    n_max_steps: int = MAX_FIX_TASK_STEPS,
    enhancement: str = ""
) :
        patch_text = ""
        try:
            tool_manager = FixTaskEnhancedToolManager(
                available_tools = [
                    "bash",
                    "create_file",
                    "str_replace_in_file",
                    "sequential_thinking",
                    "finish"
                ]
            )
            enhanced_problem = problem_statement
            if enhancement:
                enhanced_problem = (
                    problem_statement
                    + f"\n\n---\n\n# Enhanced Problem Analysis\n\n{enhancement}"
                )
            current_working_dir = os.getcwd()
            system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
                problem_statement = enhanced_problem,
                available_tools=tool_manager.get_tool_docs(),
                output_format=FORMAT_PROMPT,
                working_directory=current_working_dir
            )
            instance_prompt = INSTANCE_PROMPT_TEMPLATE
            execute_agent_workflow(
                tool_manager,
                system_prompt,
                instance_prompt,
                timeout=timeout,
                n_max_steps=n_max_steps,
                model=QWEN_MODEL
            )
            patch_text = tool_manager.get_final_git_patch()
            print(f"workflow execution completed, patch length: {len(patch_text)}")
        except Exception as e:
            error_info = f"Error: {e}, {traceback.format_exc()}"
            print(f"[CRITICAL] Exception in task processing: {error_info}")
        print(f"[CRITICAL] patch: {patch_text}")
        return patch_text
def is_all_tests_passed(output: str) -> bool:
    check_all_tests_passed_prompt = """
    Check the test output and tell me if all the tests passed successfully or there is any failure or error.
    This is the output:
    ```
    {output}
    ```
    Return only "true" or "false".
    """
    for _ in range(2):
        try:
            result, _, _ = EnhancedNetwork.make_request(
                messages=[LlmMessage.user(check_all_tests_passed_prompt.format(output=output))],
                model=QWEN_MODEL,
            )
            print(f"[IS_ALL_TESTS_PASSED] Output:", output)
            print(f"[IS_ALL_TESTS_PASSED] Result:", result)
            if result.lower() == "true":
                return True
            else:
                return False
        except Exception as e:
            print(f"[IS_ALL_TESTS_PASSED] Exception: {e}")
            time.sleep(2)
    return False
def llm_select_run_command_for_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    for _ in range(5):
        try:
            prompt = textwrap.dedent(f"""
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
            """)
            messages = [LlmMessage.user(prompt)]
            raw_text, _, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL)
            json_result = json.loads(
                raw_text.replace("```json", "").replace("```", "").strip()
            )
            return json_result.get("command")
        except Exception as e:
            time.sleep(1)
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
    selected_model = GLM_MODEL_4_6
    while retry < 10:
        try:
            messages = [
                LlmMessage.system(EXTRACT_CONCEPTS_PROMPT),
                LlmMessage.user(f"Problem Statement:\n{problem_statement}"),
            ]
            response, _, _ = EnhancedNetwork.make_request(
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
        LlmMessage.system(GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT),
        LlmMessage.user(f"Problem Statement:\n{problem_statement}\n\nInitial structure:\n{initial_structure}\nGenerate the complete and correct implementation in files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"),
    ]
    selected_model = QWEN_MODEL
    print("[GENERATE_INITIAL_SOLUTION] Requesting code generation from model")
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(
                code_generation_messages, model=selected_model, temperature=temperature
            )
            if isinstance(result, tuple):
                code_response, _, _ = result
            else:
                code_response = result
            loop_check_messages = [
                LlmMessage.system(INFINITE_LOOP_CHECK_PROMPT),
                LlmMessage.user(f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final code."),
            ]
            result2 = EnhancedNetwork.make_request(
                loop_check_messages, model=selected_model
            )
            if isinstance(result2, tuple):
                loop_check_response, _, _ = result2
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
def get_files_to_modify(problem_statement: str, tool_manager: FixTaskEnhancedToolManager) -> str:
        system_prompt = textwrap.dedent(
            """
            You are a helpful assistant that finds the files to modify related to the problem statement.
            You must check the directory structure using `list_directory_structure` tool and then determine which files are needed for the problem statement.
            You must then use the `finish_find_files_to_fix` tool to signal the completion of the file finding workflow execution.
            You have access to the following tools:-
            {tools_docs}
            {format_prompt}
            """
        ).format(tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V1)
        instance_prompt = f"Problem Statement:\n{problem_statement}"
        try:
            execute_agent_workflow(
                tool_manager,
                system_prompt,
                instance_prompt,
                finish_tool_name="finish_find_files_to_fix",
                n_max_steps=20,
                logger_prefix="FIND_FILES_TO_MODIFY",
                model=QWEN_MODEL
            )
            contents = []
            for file_path in tool_manager.files_to_fix:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        contents.append(f"{file_path}\n{{\n{f.read()}\n}}")
                except Exception as e:
                    print(f"Error in get files to modify: Failed to open file {file_path}: {e}")
            return "\n\n".join(contents)
        except Exception as e:
            print(f"Error in get files to modify: {e}")
            print(f"Error in get files to modify: Failed to open file {file_path}: {e}")
            return ""
def process_create_task(problem_statement: str, enhancement: str):
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "view_file",
            "list_directory_structure",
            "finish_find_files_to_fix",
        ]
    )
    initial_structure = get_files_to_modify(problem_statement, tool_manager)
    print(initial_structure)
    s_time = time.time()
    initial_solution = None
    BASIC_APPROACH_RETRY = 10
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
    patch = process_fix_task(
        problem_statement,
        timeout=DEFAULT_TIMEOUT - int(elapsed_time) - 200,
        n_max_steps=40,
        enhancement=enhancement
    )
    return patch
def clean_code_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response
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
            print("🟡 run_command", run_command)
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
        LlmMessage.system(GENERATE_TESTCASES_PROMPT),
        LlmMessage.user(f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nInitial structure:\n{initial_structure}\n\nGenerate the complete and correct testcases.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```\n```javascript\ntest_a.js\ncontents of test_a.js\n\ntest_b.js\ncontents of test_b.js\n```"),
    ]
    selected_model = QWEN_MODEL
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(
                test_generation_messages, model=selected_model, temperature=temperature
            )
            if isinstance(result, tuple):
                testcode_response, _, _ = result
            else:
                testcode_response = result
            testcases = clean_code_response(testcode_response)
            if not testcases or not testcases.strip():
                retry += 1
                continue
            lines = testcases.split("\n")
            if not lines or len(lines) == 0:
                retry += 1
                test_generation_messages.append(LlmMessage.assistant(testcode_response))
                test_generation_messages.append(LlmMessage.user(f"Include file name in the response. example:\n```python\ntest_a.py\n{{content}}\n\ntest_b.py\n{{content}}\n```\n```javascript\ntest_a.js\n{{content}}\n\ntest_b.js\n{{content}}\n```"))
                continue
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_single_testset: {e}")
            time.sleep(1)
    return ""
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
                result, _, _ = EnhancedNetwork.make_request(
                    messages=[LlmMessage.user(file_names_prompt)],
                    model=QWEN_MODEL,
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
def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, RUN_ID
    problem_statement = str(input_dict.get("problem_statement", ""))
    Utils.set_env_for_agent(repo_dir)
    patch = ""
    problem_type, enhancement = ProblemParser.analyze_problem(problem_statement)
    try:
        if problem_type == PROBLEM_TYPE_FIX:
            patch = process_fix_task(problem_statement, enhancement=enhancement)
        else:
            patch = process_create_task(problem_statement, enhancement=enhancement)
    except Exception as e:
        print(f"Error in agent_main: {e}")
        patch = process_fix_task(problem_statement, enhancement=enhancement)
    finally:
        os.system("git reset --hard")
    print(patch)
    return patch