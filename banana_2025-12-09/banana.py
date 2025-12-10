from __future__ import annotations
import ast
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
import threading
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# ============= MODELS =============
GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
GLM_OLD_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [model for model in [GLM_OLD_MODEL_NAME, KIMI_MODEL_NAME, GLM_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]
# ==============GLOBAL VARIABLES ===============
run_id=None
START_TIME = time.time()
MODIFIED_SOLUTION_FILES_CONTENTS = []
MODIFIED_FILES_LIST = []
MODIFIED_FILES_CONTENTS = []
# ============= CONFIGS & CONSTANTS =============
MAX_FIX_TASK_STEPS = 200
LATEST_OBSERVATIONS_TO_KEEP = 15
SUMMARIZE_BATCH_SIZE = 5
MAX_SUMMARY_RANGES = 6
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
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://localhost:1234")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1300"))
PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"
# ================= PROMPTS ==================
STOP_INSTRUCTION=textwrap.dedent("""
# 🎯 RESPONSE REQUIREMENTS
- DO NOT generate `observation:` - it will be provided by the system
- Default format: next_thought: ... followed by next_tool_name and next_tool_args (single tool call)
- Use multiple tool calls (tool_call_1, tool_call_2, etc.) ONLY when searching multiple files at once for time efficiency
- Format: next_thought: ... followed by next_tool_name/next_tool_args OR tool_call_N blocks
""")
INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
Now let's start.
```
{problem_statement}
```
""")
# ============= FIX TASK PROMPTS =============
FORMAT_PROMPT_FIX_TASK = textwrap.dedent("""
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
""")
DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}
Try a different approach:
1. If you just searched, try reading the file instead
2. If you just edited, try running tests to verify
3. If tests failed, try a different fix approach
""")
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
review_prompt = textwrap.dedent("""
# 🔍 SOLUTION REVIEW PHASE
Before finalizing, you MUST conduct a comprehensive step-by-step review of your solution process:
1. **Root Causes Analysis**
    - Verify that you have identified and addressed the actual root cause(s) of the problem
    - Ensure you're not just treating symptoms but fixing the underlying issue
    - Check if there are any deeper architectural or design issues that need attention
    - Confirm that your fix addresses the root cause completely, not partially
2. **Edge Cases Verification**
    - Review all conditional branches and boundary conditions in your solution
    - Test edge cases: empty inputs, null values, maximum/minimum sizes, boundary values
    - Check for off-by-one errors, index out of bounds, division by zero, etc.
    - Verify handling of special characters, whitespace, and unusual input formats
    - Ensure your solution handles both typical and extreme input scenarios
3. **Untested Hypotheses**
    - Identify any assumptions you made during the solution process
    - Test each hypothesis explicitly - don't assume it's correct
    - Verify that your understanding of the problem matches the actual requirements
    - Check if there are alternative interpretations of the problem statement
    - Validate that your solution approach is sound, not just that it passes visible tests
4. **Solution Process Review**
    - Trace through your entire solution step by step
    - Verify each modification you made is necessary and correct
    - Check for any unintended side effects of your changes
    - Ensure all code paths are reachable and tested
    - Review error handling and exception cases
5. **Completeness Check**
    - Verify that all requirements from the problem statement are met
    - Check if there are any hidden requirements or implicit expectations
    - Ensure backward compatibility if required
    - Confirm that your solution doesn't break existing functionality
    - Validate that all test cases (both visible and hidden) would pass
6. **Code Quality Review**
    - Check for code smells, potential bugs, or logical errors
    - Verify proper error handling and edge case coverage
    - Ensure code is maintainable and follows best practices
    - Review for any performance issues or inefficiencies
**Action Required**: Go through each of these review points systematically. If you find any issues, fix them before calling finish again. Only call finish when you are confident the solution is complete, correct, and robust.
""")
# ============= COT Related Functions To Manage Chat History =============
class EnhancedCOT:
    class Action:          
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
    def __init__(self,latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        # Store summaries: key is (start_idx, end_idx) tuple, value is summary string
        self.summaries: dict[tuple[int, int], str] = {}
        # Track which indices have been summarized
        self.summarized_ranges: list[tuple[int, int]] = []
    def add_action(self, action: EnhancedCOT.Action) -> bool: # don't add if thought is repeated
        self.thoughts.append(action)
        # Check if we need to summarize older messages
        # Only check when we have enough messages to potentially summarize
        total_thoughts = len(self.thoughts)
        if total_thoughts >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True
    
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
            
            conversation_parts.append({
                "assistant": assistant_part,
                "user": user_part,
                "is_error": thought.is_error
            })
        
        if not conversation_parts:
            return None
        
        # Build the prompt for summarization
        conversation_text = ""
        for i, part in enumerate(conversation_parts, 1):
            conversation_text += f"\n--- Step {i} ---\n"
            conversation_text += f"Assistant: {part['assistant']}\n"
            # Observation already truncated to 2000 chars, show more context (up to 1500) for summarization
            user_obs = part['user']
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conversation_text += f"User: {user_obs}\n"
            if part['is_error']:
                conversation_text += "[Error occurred]\n"
        
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
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes conversation history concisely."},
                {"role": "user", "content": summarization_prompt}
            ]
            retry = 0
            while retry < 10:
                try:
                    response, _ = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME, temperature=0.0)                    
                    return response.strip()
                except Exception as e:
                    retry += 1                    
                    time.sleep(2)            
        except Exception as e:
            return None
        
        return None
    
    def _get_summary_for_index(self, idx: int) -> Optional[str]:
        """Get the summary for a given message index if it exists."""
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None
        
    def count_repeated_thoughts(self)->int:
        """
        Count the number of consecutive repeated thoughts at the end of the COT.
        Returns 0 if no repetition, or the count of consecutive repeated thoughts.
        """
        if len(self.thoughts) < 2:
            return 0
        
        # Get the last thought as reference
        last_thought = self.thoughts[-1]
        last_tool_name = last_thought.next_tool_name
        last_tool_args = last_thought.next_tool_args
        
        # Count backwards how many consecutive thoughts match the last one
        count = 0
        for i in range(len(self.thoughts) - 1, -1, -1):
            thought = self.thoughts[i]
            if thought.next_tool_name == last_tool_name and thought.next_tool_args == last_tool_args:
                count += 1
            else:
                break
        
        # Return count - 1 to get repetitions (excluding the first occurrence)
        # If count is 1, return 0 (no repetition); if count is 2+, return count - 1
        return max(0, count - 1)
    def is_thought_repeated(self)->bool:
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
    def to_str(self):
        messages=[]
        last_summary_range = None
        # Only include summaries for the last N summarized ranges to keep context bounded
        if self.summarized_ranges:
            allowed_summary_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:])
        else:
            allowed_summary_ranges = set()
        
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            
            if i<len(self.thoughts)-self.latest_observations_to_keep:
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
                                messages.append({
                                    "role": "system",
                                    "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"
                                })
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
                        obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                    except Exception:
                        obs_render = str(thought.observation)
                else:
                    obs_render = str(thought.observation) if thought.observation else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role":"assistant","content":assistant_str})
                messages.append({"role":"user","content":user_str})
                
            else:
                # Latest observations - always show full content
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
                messages.append({"role":"assistant","content":assistant_str})
                messages.append({"role":"user","content":user_str})
        return messages
# ============= Network Related Functions =============
class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
        INCOMPLETE_RESPONSE=10
    
    @classmethod 
    def is_http_response(cls, raw_text: str):
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'HTTP ERROR: Request failed for model' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name    
        return True, None
    
    @classmethod
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        
        stripped = raw_text.strip()
        has_next_thought = "next_thought" in raw_text.lower() or "<next_thought>" in raw_text.lower()
        has_next_tool_name = "next_tool_name" in raw_text.lower() or "<next_tool_name>" in raw_text.lower()
        has_next_tool_args = "next_tool_args" in raw_text.lower() or "<next_tool_args>" in raw_text.lower()
        
        # Valid endings: JSON format or XML-style tags
        valid_ending = (stripped.endswith("}") or stripped.endswith("}]") or 
                       stripped.endswith("</next_tool_args>") or stripped.endswith(">"))
        
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
            return False, cls.ErrorType.INCOMPLETE_RESPONSE.name
        if len(raw_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
                
        return cls.is_http_response(raw_text)
    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   
    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        selected_model = QWEN_MODEL_NAME
        retry = 0
        while retry < 10:
            try:
                response, _ = cls.make_request(messages, model=selected_model)                
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                retry += 1
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(2)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
        return None
    @classmethod
    def get_cost_usage(cls) -> dict:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/usage?evaluation_run_id={run_id if run_id else str(uuid4())}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            usage_info = response.json()
            return usage_info
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return f"ERROR: Error getting model info: {e}"
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0, timeout:int=180, tool_mode: str = "none", tool_docs: list = [])->str:
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
            "tools":  tool_docs
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        for i in range(attempts):
            try:
                response = requests.post(
                    url,
                    json=request_data,
                    timeout=(60, timeout),  # (connect timeout, read timeout)
                    headers=headers
                )
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
                    print(f"### Response parse error: {str(e)}")
                    raise RuntimeError(f"HTTP ERROR: Response Parse Error timeout for model {model} after {attempts} attempts")
                
                if (tool_mode == "none" and (raw_text is None or raw_text == "")) or (tool_mode != "none" and (tool_calls is None or len(tool_calls) == 0)):
                    raise RuntimeError(f"HTTP ERROR: NO RESPONSE FOUND Tool model {model} after {attempts} attempts")
                
                elapsed = time.time() - start_time
                return raw_text, tool_calls
                
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request timeout for model {model} after {attempts} attempts")
                time.sleep(5)
                continue
                
            except requests.exceptions.ConnectionError as e:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Connection error for model {model} after {attempts} attempts: {e}")
                time.sleep(10)
                continue
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                elapsed = time.time() - start_time
                error_msg = f"HTTP ERROR: HTTP ERROR {status_code} for model {model}"
                
                # Check for 504 Gateway Timeout specifically
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 504: Gateway Timeout for model {model} after {attempts} attempts: {e}")
                    time.sleep(10)
                    continue
                
                if i >= attempts - 1:
                    raise RuntimeError(f"{error_msg} after {attempts} attempts: {e}")
                time.sleep(10)
                continue
                
            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start_time
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request failed for model {model} after {attempts} attempts: {e}")
                time.sleep(10)
                continue
        # Fallback (should not reach here due to raises above)
        raise RuntimeError(f"HTTP ERROR: Failed to get response for model {model} after {attempts} attempts")
    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, 
                            models: List[str],
                            max_retries: int = 2, 
                            base_delay: float = 1.0,
                            temperature: float = 0.0) -> str:
        
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        current_model_idx = 0
        used_model = models[0] if models else None
        
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                current_model = models[current_model_idx] if current_model_idx < len(models) else models[-1]
                used_model = current_model
                logger.info(f"[RETRY] Attempt {attempt + 1}/{max_retries} using model: {current_model}")
                
                # index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text, _ = cls.make_request(messages,model=current_model, temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                
                # Check if it's a 504 error - switch to next model
                is_504_error = "504" in error_body or "HTTP ERROR 504" in error_body or "Gateway Timeout" in error_body
                
                if is_504_error and current_model_idx < len(models) - 1:
                    logger.warning(f"[MODEL SWITCH] 504 error detected with model {current_model}. Switching to next model...")
                    current_model_idx += 1
                    logger.info(f"[MODEL SWITCH] Switched to model: {models[current_model_idx]}")
                    # Don't count this as an attempt for the next model, so continue without incrementing
                    time.sleep(3)
                    continue
                
                if attempt < max_retries - 1:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in 3 seconds... (attempt {attempt + 1}/{max_retries})") 
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name]+=1
                    if "HTTP ERROR" not in error_body and "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body and "NETWORK_ERROR" not in error_body and "HTTP ERROR 429" not in raw_text and "INCOMPLETE_RESPONSE" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: " + error_body})
                        
                    time.sleep(random.uniform(2, 4))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages,used_model
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
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
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, next_tool_args: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''
        next_tool_args=next_tool_args.replace('```json','').strip('```')
        error_msg=''
        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg=f"Invalid JSON: {next_tool_args}"    
            try:
                tool_args_list = EnhancedToolManager.get_tool_args_for_tool(tool_name, required_only=True)
                # Check if get_tool_args_for_tool returned an error string instead of a list
                if isinstance(tool_args_list, str):
                    # Tool not found in class-level TOOL_LIST, can't parse malformed JSON
                    raise Exception(error_msg)
                next_tool_args = cls.parse_malformed_json(tool_args_list, next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args
    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), temperature:float=0.0) -> tuple[str, str | list, dict | list, str, int, dict, list, str]:
        """Prod inference with caching"""
        # Support both single model (str) and list of models
        models = [model] if isinstance(model, str) else model
        logger.debug(f"Inference called with models={models}, temperature={temperature}, messages={len(messages)}")
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
            logger.error("No valid messages after cleaning")
            raise RuntimeError("No valid messages to send to proxy.")
        # Calculate and print input token size
        total_tokens = sum(Utils.count_tokens(msg.get("content", "")) for msg in cleaned_msgs)
        print(f"[INFERENCE] Input token size: {total_tokens} tokens (models={models}, messages={len(cleaned_msgs)})")
        
        logger.debug(f"Cleaned messages: {len(cleaned_msgs)} messages")
        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages,used_model = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
        logger.debug(f"Inference completed: tool_name={next_tool_name}, total_attempts={total_attempts}, used_model={used_model}")
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages,used_model
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        """Sanitize response text by normalizing field names"""
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        text_resp=re.sub("[\'\"]*tool_call_[\'\"]*","tool_call_",text_resp)
        
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{re.escape(next_tool_name)}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)
        
        return text_resp
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
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    if start == -1:
                        start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        return text[start:i+1]
        
        return None
    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        """Extract tool_name and tool_args from a tool_call block"""
        tool_name_match = re.search(r'tool_name\s*:\s*(\S+)', block, re.IGNORECASE)
        if not tool_name_match:
            return None
        
        tool_name = tool_name_match.group(1).strip().strip('"').strip("'")
        
        # Find tool_args
        args_match = re.search(r'tool_args\s*:\s*(\{)', block, re.IGNORECASE)
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
            text_resp = re.split(r'observation\s*:', text_resp, flags=re.IGNORECASE)[0].strip()
        
        text_resp = cls.sanitise_text_resp(text_resp)
		
        if "Infrastructure is at maximum capacity" in text_resp:
            return None,None,None, "HTTP ERROR Maximum Capacity"
        
        if "No instances available" in text_resp:
            return None,None,None, "HTTP ERROR NO INSTANCES AVAILABLE"                
        
        next_thought = None
        thought_patterns = [
            r'next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))',
            r'next_thought\s*:\s*(.*?)(?=\ntool_call_)',
            r'next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)',
            r'next_thought\s*:\s*(.*)',
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
        tool_call_pattern = r'tool_call_(\d+)\s*:'
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
                    
                    # Parse args - if single tool, use it directly; if multiple, parse for first tool (args should be array)
                    if len(next_tool_names) == 1:
                        parsed_args = cls.parse_next_tool_args(next_tool_names[0], next_tool_args_raw)
                        if isinstance(parsed_args, list):
                            next_tool_args_list = parsed_args
                        else:
                            next_tool_args_list = [parsed_args]
                        return next_thought, next_tool_names[0], next_tool_args_list[0], error_msg
                    else:
                        # Multiple tools - parse args as array, then parse each element for corresponding tool
                        try:
                            parsed_args_raw = Utils.load_json(next_tool_args_raw.strip())
                            if isinstance(parsed_args_raw, list):
                                next_tool_args_list = []
                                for i, tool_name in enumerate(next_tool_names):
                                    if i < len(parsed_args_raw):
                                        # Parse args for this specific tool
                                        tool_args_str = json.dumps(parsed_args_raw[i]) if isinstance(parsed_args_raw[i], dict) else str(parsed_args_raw[i])
                                        parsed = cls.parse_next_tool_args(tool_name, tool_args_str)
                                        next_tool_args_list.append(parsed if isinstance(parsed, dict) else {})
                                    else:
                                        next_tool_args_list.append({})
                            else:
                                # Single args object for all tools - parse for first tool and replicate
                                parsed_args = cls.parse_next_tool_args(next_tool_names[0], next_tool_args_raw)
                                next_tool_args_list = [parsed_args if isinstance(parsed_args, dict) else {} for _ in next_tool_names]
                        except Exception:
                            # Fallback: try parsing as single args for first tool
                            parsed_args = cls.parse_next_tool_args(next_tool_names[0], next_tool_args_raw)
                            next_tool_args_list = [parsed_args if isinstance(parsed_args, dict) else {} for _ in next_tool_names]
                        return next_thought, next_tool_names, next_tool_args_list, error_msg
                        
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
# ============= Utils =============
class Utils:
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.info(f"unable to fix manually, trying with llm")
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)
                
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        with open("../failed_messages.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])
    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        import re
        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
        else:
            text = messages
        
        # Split into words and non-word tokens (punctuation, operators, etc.)
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        
        count = 0
        for token in tokens:
            if token.isspace():
                continue  # Whitespace is typically absorbed
            elif len(token) == 1:
                count += 1  # Single chars (punctuation, operators)
            else:
                count += max(1, (len(token) + 2) // 3)
        
        return count
class FileOperationsUtil:
    """
    Utility class providing common file operations used across managers.
    Handles file reading, writing.
    """
    
    def __init__(self, new_files_created: list):
        self.new_files_created = new_files_created
        self.file_system_manager = None
        self.search_manager = None
    
    def set_managers(self, file_system_manager, search_manager):
        """Set manager references after initialization to avoid circular dependencies."""
        self.file_system_manager = file_system_manager
        self.search_manager = search_manager
    
    def save(self, file_path: str, content: str) -> str:
        """
        Save content to file.
        
        Arguments:
            file_path: path to save the file
            content: content to write
        
        Returns:
            Success message
        """
        with open(file_path, "w") as file:
            file.write(content)
        self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"
    
    def get_file_content(self, file_path: str, search_start_line: int = None, 
                        search_end_line: int = None, search_term: str = None, 
                        limit: int = 1000, add_line_numbers: bool = False, 
                        structural_truncation: bool = False) -> str:
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
        search_callback = lambda fp, st: self.search_manager.search_in_file(
            fp, st
        )
        
        return self.file_system_manager.get_file_content(
            file_path=file_path,
            search_start_line=search_start_line,
            search_end_line=search_end_line,
            search_term=search_term,
            limit=limit,
            add_line_numbers=add_line_numbers,
            search_in_file_callback=search_callback
        )
class SearchManager:
    """
    Manages search operations across files and within specific files.
    Handles grep-based searches and pattern matching with context extraction.
    """
    
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
        if not cmd_stripped.startswith('grep'):
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
            '''
            Return the source code around matches showing ±20 lines of context.
            The final output is truncated with `limit_strings` to avoid excessive verbosity.
            '''
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_lines = f.read().splitlines()
            except Exception as e:
                logger.error(f"Error reading '{file_path}': {e}")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error reading '{file_path}': {e}")
            # Identify all lines that contain the search term.
            escaped_search_term = re.escape(search_term)
            match_lines = [idx + 1 for idx, line in enumerate(source_lines) if escaped_search_term in line]
            if not match_lines:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in file '{file_path}'")
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
                    context_src = "\n".join(source_lines[start_line - 1:end_line])
                    chunks.append(f"(lines {start_line}-{end_line}):\n{context_src}")
            
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)
        
        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return f"Search results are too long. Please refine your search term into more specific terms."
        else:
            return output
class FileSystemManager:
    """
    Manages file system navigation and reading operations.
    Handles directory listing, file content retrieval, and structural analysis.
    """
    
    def __init__(self):
        pass
    
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
                ignore = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.tox', '.venv', 'venv', '.mypy_cache', '.eggs', 'dist'}
                items = sorted(os.listdir(path))
                dirs = [i for i in items if not i.startswith('.') and i not in ignore and not i.endswith('.egg-info') and os.path.isdir(os.path.join(path, i))]
                files = [i for i in items if not i.startswith('.') and os.path.isfile(os.path.join(path, i))]
                entries = []
                for i, d in enumerate(dirs):
                    is_last = i == len(dirs) - 1 and not files
                    entries.append(f"{prefix}{'└── ' if is_last else '├── '}{d}/")
                    if current_depth < depth:
                        entries.extend(tree(os.path.join(path, d), prefix + ('    ' if is_last else '│   '), current_depth + 1))
                for i, f in enumerate(files):
                    entries.append(f"{prefix}{'└── ' if i == len(files) - 1 else '├── '}{f}")
                return entries
            except:
                return [f"{prefix}[Error]"]
        
        lines = tree(file_path, "", 0)
        return f"Directory structure (depth={depth}):\n{file_path}/\n" + "\n".join(lines) + f"\n\n{sum(1 for l in lines if l.rstrip().endswith('/'))}-dirs, {sum(1 for l in lines if not l.rstrip().endswith('/') and '[' not in l)}-files"
    
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, 
                        search_term: str = None, limit: int = 1000, add_line_numbers: bool = False, 
                        search_in_file_callback=None) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers.
        Works for all file types (Python, JavaScript, etc.).
        
        Arguments:
            file_path: filesystem path to target file
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
            limit: maximum output size in characters (-1 for unlimited)
            add_line_numbers: whether to add line numbers to output
            search_in_file_callback: callback to search within file
        
        Returns:
            File content as string
        """
        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            """Helper method to add line numbers to content."""
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return '\n'.join(numbered_lines)
        
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
class TestManager:
    """
    Manages test-related operations including running code, generating tests, and executing repository tests.
    Handles test file creation and test execution with various test runners.
    """
    
    def __init__(self, file_ops: 'FileOperationsUtil' = None):        
        self.file_ops = file_ops
    
    def _truncate_output(self, output: str, max_first_lines: int = 500, max_last_lines: int = 500) -> str:
        """Truncate long output to first N and last N lines with summary in middle."""
        lines = output.split('\n')
        total_lines = len(lines)
        
        if total_lines <= max_first_lines + max_last_lines:
            return output
        
        first_lines = lines[:max_first_lines]
        last_lines = lines[-max_last_lines:]
        omitted_lines = total_lines - max_first_lines - max_last_lines
        
        truncated = '\n'.join(first_lines)
        truncated += f"\n\n... ({omitted_lines} lines omitted) ...\n\n"
        truncated += '\n'.join(last_lines)
        
        return truncated
    
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
        logger.info(f"Running code from file: {file_path}")
        
        python_extensions = ('.py', '.pyw', '.pyx', '.pyi', '.pxd', '.pxi', '.pyz')
        if file_path.endswith(python_extensions):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
            
        self.file_ops.save(file_path, content)
        generated_test_files.append(file_path)
        
        try:
            run_command = get_run_command_for_file(file_path)
            logger.info(f"Running command in run_code: {run_command}")
            result = subprocess.run(run_command, capture_output=True, text=True, check=False, timeout=60)
        except ValueError as e:
            return f"Error: {e}"
            
        if result.returncode != 0:
            return f"Error running code: {result.stderr}"
        logger.debug(f"Code execution successful: {file_path}")
        observation = f"{result.stdout}\n"
        return observation
class CodeEditManager:
    """
    Manages code editing operations including targeted text replacement.
    Handles search/replace operations with similarity matching and error detection.
    """
    
    def __init__(self, file_ops: 'FileOperationsUtil' = None):
        self.file_ops = file_ops
    
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files.
        
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        
        Returns:
            Operation status - success confirmation with context or detailed error with guidance
        """
        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
            """Add context lines around a similar match for better understanding."""
            lines = original_content.split('\n')
            
            # Extract the actual content from the formatted match (remove the description part)
            match_lines = formatted_match.split('\n')
            if len(match_lines) < 2:
                return formatted_match
                
            # Skip the description line (e.g., "Lines 45-47: ..." or "Line 23: ...")
            actual_content_lines = match_lines[1:]
            actual_content = '\n'.join(actual_content_lines)
            
            # Find where this content appears in the original file
            best_match_start = -1
            best_similarity = 0
            
            # Search for the best matching position in the original content
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i:i + len(actual_content_lines)]
                candidate_content = '\n'.join(candidate_lines)
                
                import difflib
                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            
            if best_match_start == -1:
                return formatted_match  # Fallback to original if can't find position
            
            # Calculate context boundaries
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
            lines = original_content.split('\n')
            
            # Try different chunk sizes to find the best match
            chunks = []
            
            # Individual lines
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
            
            # Multi-line chunks (3-5 lines) for better context
            search_lines = search_string.split('\n')
            target_chunk_size = max(3, len(search_lines))
            
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i:i + target_chunk_size]
                chunk_content = '\n'.join(chunk_lines).strip()
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
                    # Find the position of the replacement and extract context
                    replace_pos = new_content.find(replace)
                    if replace_pos != -1:
                        lines = new_content.split('\n')
                        # Find which line number the replacement starts at
                        chars_so_far = 0
                        replace_line_start = 0
                        for i, line in enumerate(lines):
                            if chars_so_far + len(line) >= replace_pos:
                                replace_line_start = i
                                break
                            chars_so_far += len(line) + 1  # +1 for newline
                        
                        # Calculate how many lines the replacement spans
                        replace_lines_count = replace.count('\n') + 1
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
                    return f"Error: code edit failed in file {file_path}. {str(e)}"
            case num_hits:
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
# ============= Tools Definitions =============
class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}
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
    def tool(fn):
        def wrapper(self, *args, **kwargs):
            # Use .get() with default 0 to handle methods not in TOOL_LIST
            self.tool_invocations[fn.__name__] = self.tool_invocations.get(fn.__name__, 0) + 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                # Initialize tool_failure entry if not present
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {j:0 for j in self.Error.ErrorType.__members__}
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message
        # Preserve original function metadata
       
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True
        return wrapper
    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            # Special handling for list[str] / List[str] annotations so that the
            # generated JSON schema correctly represents an array of strings.
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
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
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        
        return tool_schemas
    @classmethod
    def get_tool_args_for_tool(cls, tool_name: str, required_only: bool = False) -> list[str] | str:
        if tool_name not in cls.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(cls.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return cls.TOOL_LIST[tool_name]['input_schema']['required']
    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])
    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        
        return tool_method
    
    def _save(self,file_path: str, content: str)->str:
        with open(file_path, "w") as file:
            file.write(content)
        # self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"
    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
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
            print("Generating git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            
            # output = output.stdout.decode("utf-8") + '\n' + output.stderr.decode("utf-8")
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"
class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(
        self,
        available_tools: Optional[list[str]] = [],
    ):
        self.new_files_created = []
        self.available_tools = available_tools
        self.generated_test_files = []
        self.observation_dir = ".observation"
        # Create observation directory if it doesn't exist
        os.makedirs(self.observation_dir, exist_ok=True)
        # Initialize file operations utility
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        # Initialize managers
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.test_manager = TestManager(
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
        Performs targeted text replacement within source files.
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
    def modify_test_case(self, file_path: str, search: str, replace: str) -> str:
        """
        Modifies test files or test cases when they are incorrect or need correction.
        Use this tool when you identify that a test file or specific test case is wrong and needs to be fixed.
        This tool uses the same underlying mechanism as apply_code_edit but is specifically intended for correcting test files.
        Arguments:
            file_path: path to the test file that needs modification
            search: exact text pattern in the test file to locate and replace (e.g., the incorrect test case code)
            replace: corrected test case code to substitute
        Output:
            Operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(
            file_path=file_path, search=search, replace=replace
        )
    
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
    @EnhancedToolManager.tool
    def run_shell_cmd(self, command: str) -> str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            command: A shell command to be run.
        Output:
            The stdout results of the command. Your working directory is the root of the project.
        '''
        if not command:
            return "Error: No command provided."
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=150
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command '{command}' timed out after 150 seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"
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
            logger.debug(f"Excluding files from patch: {exclude}")
            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.splitlines()
            # Filter and validate files before adding
            to_add = []
            for f in ls:
                # Skip empty strings
                if not f:
                    continue
                # Strip whitespace and newlines
                f = f.strip()
                if not f:
                    continue
                # Skip excluded files
                if f in exclude:
                    continue
                # Validate that it's a valid file path (not containing newlines or null bytes)
                # Note: tabs are valid in filenames on some systems, so we only check for truly invalid chars
                if '\n' in f or '\r' in f or '\0' in f:
                    logger.warning(f"Skipping invalid filename with control characters: {repr(f)}")
                    continue
                # Check if file actually exists and is a file (not a directory)
                if os.path.exists(f) and os.path.isfile(f):
                    to_add.append(f)
                elif not os.path.exists(f):
                    # File might be deleted or not yet created, skip it
                    logger.debug(f"Skipping non-existent file: {f}")
                    continue
                else:
                    # Path exists but is not a file (might be a directory)
                    logger.warning(f"Skipping non-file path: {f}")
                    continue
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            else:
                # No files to add, return empty patch
                logger.info("No files to add to patch")
                return ""
            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
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
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        return self.search_manager.search_in_file(
            file_path=file_path, search_term=search_term
        )
    
# ============= Fix Task Related Functions =============
def fix_task_solve_workflow(problem_statement: str, *, timeout: int,  enhancement: str):    
    print("[FIX_TASK_SOLVE_WORKFLOW] Initializing workflow")
    global run_id, START_TIME    
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE
    )
    enhanced_problem = problem_statement
    if enhancement:
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement        
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "create_new_file",
            "list_directory_structure",
            "get_file_content",
            "search_in_all_files_content",
            "apply_code_edit",
            "run_code",
            "run_shell_cmd",
            "finish",
            "think"
        ]
    )
    
    # Create initial system_prompt with tools for step 0
    system_prompt_0 = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX_TASK
    )
    instance_prompt_0 = INSTANCE_PROMPT_TEMPLATE.format(
        problem_statement=enhanced_problem
    )
    # Progress tracking for multi-file changes
    global MODIFIED_FILES_LIST, MODIFIED_FILES_CONTENTS
    MODIFIED_FILES_LIST = []  # Reset at start of workflow
    MODIFIED_FILES_CONTENTS = []
    def workflow(step: int, system_prompt: str, instance_prompt: str, finish_called_count: int = 0, step_prompt: str = None):
        """
        DFS-based recursive workflow function, try to do complete search.
        Returns True if terminating condition detected for both of timeout or finish tool call, False otherwise.
        """
        global MODIFIED_FILES_LIST, MODIFIED_FILES_CONTENTS
        
        # Base case: check timeout
        if time.time() - START_TIME > timeout:
            print("[FIX_TASK_SOLVE_WORKFLOW] Timeout reached")
            return True
        
        print(f"[FIX_TASK_SOLVE_WORKFLOW] Starting workflow step {step}")
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        if step_prompt is not None:
            print(f"[FIX_TASK_SOLVE_WORKFLOW] Step prompt: {step_prompt}")
            messages.append({"role": "user", "content": step_prompt})
        temperature = 0.0
        repeated_thoughts_count = cot.count_repeated_thoughts()
        if repeated_thoughts_count == 2:
            # Remove the last thought to backtrack and try a different approach
            if cot.thoughts:
                cot.thoughts.pop()
            return False
        if repeated_thoughts_count == 1:
            if cot.thoughts:
                last_thought = cot.thoughts[-1]
                messages.append(
                    {"role": "user", "content": MAKE_REAL_PROGRESS.format(
                        previous_response=f"next_thought:{last_thought.next_thought}\n next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    )}
                )
                temperature = 0.7
        tried_thoughts = []
        for retry in range(3):
            if time.time() - START_TIME > timeout:
                print("[FIX_TASK_SOLVE_WORKFLOW] Timeout reached")
                return True
            try:
                next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages, used_model = EnhancedNetwork.inference(
                    messages,
                    model=GLM_MODEL_NAME,
                    run_id=run_id,
                    temperature=temperature
                )
            except Exception as e:
                temperature=0.0
                logger.error(f"[FIX_TASK_SOLVE_WORKFLOW] Inference failed on retry {retry + 1}: {e}")
                continue
            if time.time() - START_TIME > timeout:
                logger.info("[FIX_TASK_SOLVE_WORKFLOW] Timeout reached")
                return True
            if next_thought is None:
                temperature=0.0
                continue
            if next_thought in tried_thoughts:
                temperature=0.7
                continue
            tried_thoughts.append(next_thought)
            tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
            tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]
            all_observations = []
            success_observation_count = 0
            error_observation_count = 0
            for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
                try:
                    if '"' in tool_name or "'" in tool_name:
                        tool_name = tool_name.replace('"', '').replace("'", "")
                    print("[FIX_TASK_SOLVE_WORKFLOW] Executing tool")
                    tool_func = tool_manager.get_tool(tool_name)
                    if isinstance(tool_func, str):
                        # get_tool returned an error string
                        observation = tool_func
                    elif tool_args:
                        observation = tool_func(**tool_args)
                    else:
                        observation = tool_func()
                    # Track file modifications for multi-file change awareness
                    if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
                        file_path = tool_args["file_path"]
                        if "ok, code edit applied successfully" in str(observation).lower():
                            if file_path not in MODIFIED_FILES_LIST:
                                MODIFIED_FILES_LIST.append(file_path)
                    all_observations.append(observation)
                    success_observation_count += 1
                except EnhancedToolManager.Error as e:
                    error_msg = f"Tool {idx + 1} ({tool_name}) error: {e.message}"
                    all_observations.append(error_msg)
                    error_observation_count += 1
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    error_msg = f"Tool {idx + 1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                    all_observations.append(error_msg)
                    error_observation_count += 1
            if error_observation_count >= success_observation_count:
                if temperature == 0.0:
                    temperature = 0.7
                else:
                    temperature = 0.0
                continue
            # Combine observations
            if len(all_observations) == 1:
                combined_observation = all_observations[0]
            else:
                combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                    [
                        f"Tool {i + 1} ({tool_names_list[i]}):\n{obs}"
                        for i, obs in enumerate(all_observations)
                    ]
                )
            show_transition(step, next_thought, next_tool_name, next_tool_args, combined_observation)
            # Add action to COT before checking finish (so finish action is recorded)
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,  # Keep original format (list or single)
                    next_tool_args=next_tool_args,
                    observation=combined_observation,
                    is_error=not error_observation_count == 0,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages
                )
            )
            
            # Check timeout condition before proceeding to next step
            if time.time() - START_TIME > timeout:
                print("[FIX_TASK_SOLVE_WORKFLOW] Timeout reached")
                return True           
            
            finish_called = any(name == "finish" for name in tool_names_list)
            if finish_called:
                # After finish: allow review tools, but limit to 2 finish calls total
                if finish_called_count >= 1:  # Second finish call (count was 1, becomes 2) - terminate
                    return True
                # First finish call: continue with review tools
                # Remove the last thought (finish action) before review phase
                if cot.thoughts:
                    cot.thoughts.pop()
                # Get the contents of the modified files
                modified_file_contents = get_files_content(MODIFIED_FILES_LIST)
                MODIFIED_FILES_CONTENTS.extend(modified_file_contents)
                print(f"[FIX_TASK_SOLVE_WORKFLOW] First finish call, Review mode started")
                is_finished = workflow(step + 1, system_prompt, instance_prompt, finish_called_count + 1, step_prompt=review_prompt)
            else:    
                # Recursive call to next step (DFS)
                is_finished = workflow(step + 1, system_prompt, instance_prompt, finish_called_count, step_prompt=None)
           
            if is_finished:
                return True
            if time.time() - START_TIME > timeout:
                print("[FIX_TASK_SOLVE_WORKFLOW] Timeout reached")
                return True
        return False
    
    workflow(0, system_prompt_0, instance_prompt_0, 0)
 
def process_fix_task(input_dict: Dict[str, Any], enhancement: str):
    problem_text = input_dict.get("problem_statement")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()
    fix_task_solve_workflow(problem_text, timeout=1300, enhancement=enhancement)
# =================== Other Functions ==================
def check_problem_type(problem_statement): # type: ignore
    def get_problem_type(problem_statement: str, enhancement: str) -> str:
        retry = 0
        PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
            '''
            You are a helpful Problem Classifier to find a Task Name from PROJECT DESCRIPTION and project structure.
            Classify development tasks as either:
            - FIX: If the PROJECT DESCRIPTION is about fixing a bug, creating a new functionality or improving the existing codebase.
            - CREATE: If the PROJECT DESCRIPTION is about creating a new functionality from scratch.
            Output ONLY: "CREATE" or "FIX"
            '''
        )
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                    {"role": "user", "content": f"{problem_statement}\n# Enhanced Problem: \n{enhancement}"}
                ]
                
                response, _ = EnhancedNetwork.make_request(messages, model=selected_model)              
                if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                    retry += 1
                else:
                    return response
            except Exception as e:
                logger.error(f"Error in get_problem_type: {e}")
                retry += 1
                other_models = [model for model in AGENT_MODELS if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]        
                time.sleep(2)     
        return PROBLEM_TYPE_FIX
    enhancement = enhance_problem_statement(problem_statement)
    return get_problem_type(problem_statement, enhancement), enhancement
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
            messages = [{"role": "system", "content": ENHANCEMENT_PROMPT}, {"role": "user", "content": f"Problem Statement:\n\n{problem_statement}"}]
            enhanced, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            return enhanced
        except Exception as e:
            retry += 1
            time.sleep(2)
 
    return ""
def get_misunderstanding_point(problem_statement: str, code_skeleton: str) -> str:
    """
    Analyzes the problem statement and code skeleton to identify potential misunderstanding points
    that could lead to implementation failures.
    
    Args:
        problem_statement: The problem description
        code_skeleton: The initial code structure/files provided
        
    Returns:
        A string containing the identified misunderstanding points and recommendations
    """
    MISUNDERSTANDING_ANALYSIS_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer and problem analyst. Your task is to identify potential misunderstanding points
        that could lead to implementation failures when solving the given problem.
        
        Analyze the problem statement and code skeleton to identify:
        
        1. **Ambiguous Requirements**:
           - Unclear specifications that could be interpreted multiple ways
           - Missing details that might lead to incorrect assumptions
           - Vague constraints or edge cases not explicitly mentioned
        
        2. **Common Misinterpretations**:
           - Typical mistakes developers make when reading similar problems
           - Easy-to-miss requirements or constraints
           - Subtle details that are often overlooked
        
        3. **Code Skeleton Analysis**:
           - Potential misunderstandings from the provided code structure
           - Function signatures that might be misinterpreted
           - Expected behavior implied by the skeleton that might conflict with requirements
           - Missing or incomplete hints in the skeleton
        
        4. **Implementation Pitfalls**:
           - Logic errors that are likely to occur
           - Edge cases that are easy to miss
           - Data structure or algorithm choices that might be incorrect
           - Boundary conditions that could be misunderstood
        
        5. **Critical Points to Clarify**:
           - Specific questions that should be answered before implementation
           - Assumptions that must be verified
           - Requirements that need explicit confirmation
        
        Format your response as markdown with clear section headers.
        Be specific and actionable. Focus on misunderstandings that would lead to test failures or incorrect implementations.
        """
    )
    
    retry = 0
    selected_model = random.choice([GLM_MODEL_NAME, KIMI_MODEL_NAME])
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": MISUNDERSTANDING_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": f"Problem Statement:\n{problem_statement}\n\nCode Skeleton:\n{code_skeleton}\n\nIdentify the potential misunderstanding points that could lead to implementation failures."
                }
            ]
            misunderstanding_analysis, _ = EnhancedNetwork.make_request(messages, model=selected_model, temperature=0.0)
            return misunderstanding_analysis
        except Exception as e:
            logger.error(f"Error in get_misunderstanding_point: {e}")
            retry += 1
            if retry < 10:
                # Try different model on retry
                other_models = [model for model in [GLM_MODEL_NAME, QWEN_MODEL_NAME] if model != selected_model]
                selected_model = other_models[random.randint(0, len(other_models) - 1)]
            time.sleep(random.uniform(2, 4))
    
    return ""
def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"
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
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.error(f"ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)
def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
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
    problem_type = None
    def run_task():
        nonlocal result, exception_occurred, problem_type
        try:
            problem_type, enhancement = check_problem_type(
                input_dict.get("problem_statement")
            )
            if problem_type == PROBLEM_TYPE_FIX:
                process_fix_task(input_dict, enhancement)
            else:
                process_create_task(input_dict, enhancement)
        except Exception as e:
            exception_occurred = e
            logger.error(f"Error in agent_main: {e}")
            try:
                time.sleep(1)
                process_fix_task(input_dict, enhancement)
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
    
    if problem_type == PROBLEM_TYPE_CREATE:
        print(f"[AGENT MAIN] Create task work flow successfully finished")
        print(f"[AGENT MAIN] Initialize repo")
        initialize_repo()
        print(f"[AGENT MAIN] Write files")
        write_files(MODIFIED_SOLUTION_FILES_CONTENTS)
        print(f"[AGENT MAIN] Get final git patch")
        result = EnhancedToolManager().get_final_git_patch()
        print(f"[AGENT MAIN] Final git patch: {result}")
    else:
        print(f"[AGENT MAIN] Fix task work flow successfully finished")
        if len(MODIFIED_FILES_CONTENTS) > 0:
            print(f"[AGENT MAIN] Write **MODIFIED_FILES_CONTENTS**")
            for file_path, content in MODIFIED_FILES_CONTENTS:
                print(f"[AGENT MAIN] Writing file: {file_path}")
            initialize_repo()
            write_files(MODIFIED_FILES_CONTENTS)
        else:
            print(f"[AGENT MAIN] No files to write")
            modified_file_contents = get_files_content(MODIFIED_FILES_LIST)
            initialize_repo()
            write_files(modified_file_contents)
        result = EnhancedToolManager().get_final_git_patch()
        print(f"[AGENT MAIN] final patch is {result}")
    
    return result if result else ""
# ============= Create Task Related Prompts =============
FORMAT_PROMPT_CREATE_TASK = textwrap.dedent("""
**Default: Use single tool call format. Use multiple tool calls ONLY when searching multiple files at once for time efficiency.**
## Response Formats
### Format 1: Single Tool Call (DEFAULT - Use this for most operations)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}
### Format 2: Multiple Tool Calls (ONLY for multi-file searches)
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
## When to Use Multiple Tool Calls
**ONLY use multiple tool calls when:**
- Searching multiple files at once (e.g., codebase_search on multiple files/directories simultaneously)
**Examples:**
✅ **Good - Multiple file searches (time efficient)**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: search_in_all_files_content
    tool_args: {"search_term": "function function_name"}
tool_call_2:
    tool_name: search_in_all_files_content
    tool_args: {"search_term": "function_name("}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "file_name.js"}
✅ **Good - Single tool call (default)**:
next_thought: I'll read this file to understand the code
next_tool_name: get_file_content
next_tool_args: {"file_path": "aaa.py"}
✅ **Good - Single tool to edit file**:
next_thought: I'll edit the file
next_tool_name: apply_code_edit
next_tool_args: {"file_path": "aaa.py", "search": "old_code", "replace": "new_code"}
✅ **Good - Single tool call to verify**:
next_thought: I'll run a command to verify the changes
next_tool_name: run_shell_cmd
next_tool_args: {"command": "your_command_here"}
## Critical Rules
- Default to single tool call format (next_tool_name, next_tool_args)
- Use multiple tool calls ONLY for parallel multi-file searches
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
""")
FORMAT_PROMPT_FIND_FILES_TO_FIX = textwrap.dedent("""
**Default: Use single tool call format. Use multiple tool calls ONLY when searching multiple files at once for time efficiency.**
## Response Formats
### Format 1: Single Tool Call (DEFAULT - Use this for most operations)
next_thought: [Your detailed reasoning]
next_tool_name: [exact tool name]
next_tool_args: {valid JSON}
### Format 2: Multiple Tool Calls (ONLY for multi-file searches)
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
## When to Use Multiple Tool Calls
**ONLY use multiple tool calls when:**
- Searching multiple files at once (e.g., codebase_search on multiple files/directories simultaneously)
**Examples:**
✅ **Good - Multiple file searches (time efficient)**:
next_thought: I need to find all references to the function
tool_call_1:
    tool_name: get_file_content
    tool_args: {"file_path": "aaa.py"}
tool_call_2:
    tool_name: get_file_content
    tool_args: {"file_path": "bbb.js"}
tool_call_3:
    tool_name: get_file_content
    tool_args: {"file_path": "ccc.js"}
✅ **Good - Single tool call (default)**:
next_thought: I'll read this file to understand the code
next_tool_name: get_file_content
next_tool_args: {"file_path": "aaa.py"}
## Critical Rules
- Default to single tool call format (next_tool_name, next_tool_args)
- Use multiple tool calls ONLY for parallel multi-file searches
- All JSON must be properly formatted with quotes
- Tool names must match exactly (case-sensitive)
""")
MAKE_REAL_PROGRESS=textwrap.dedent("""
You are making same mistakes.
Your previous response: 
{previous_response}
**Critical**:
1. Notice what you are going to do.
2. Find the reason the same mistake is repeated.
3. Don't make the same mistakes any more and make a real progress.
""")
CREATE_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
# Hey there! You're a Coding Assistant 🚀. You will be provided with a problem statement and need to create/fix code to implement the requirements.
**CRITICAL CONTEXT:**
This workflow involves an initial solution and test cases that were generated automatically. Both the source code files AND the test files may contain errors or incorrect implementations. The problem statement is the authoritative source of truth.
## Follow these steps to solve the problem:
1. **Understand the problem deeply** - Carefully read the problem statement and identify potential misunderstanding points, ambiguous requirements, and edge cases that might be overlooked.
2. **Review the initial solution** - Examine the automatically generated source code and test files. Both may contain errors.
3. **Identify edge cases and misunderstanding points** - Before coding, think about:
   - Boundary conditions (empty inputs, null values, maximum/minimum sizes)
   - Invalid inputs and error handling
   - Ambiguous requirements that could be misinterpreted
   - Common pitfalls and corner cases
   - Edge cases that might break the solution
4. **Fix source code** - Edit the source code to correctly implement the requirements, handling all identified edge cases.
5. **Fix test files if needed** - If test cases conflict with the problem statement, fix them using `modify_test_case` tool. If source code doesn't match requirements, fix the source code.
6. **Test thoroughly** - Run tests after each change. Test edge cases explicitly - empty inputs, boundary values, invalid inputs, and corner cases.
7. **Iterate until perfect** - Continue refining until all tests pass and edge cases are handled. Failing to test edge cases rigorously is the NUMBER ONE failure mode.
8. **Final validation** - Ensure the solution handles all edge cases, matches the problem statement exactly, and all tests pass. Be aware there are hidden tests that must also pass.
## Edge Cases and Misunderstanding Points (CRITICAL):
- **Always identify and handle edge cases first** before implementing the main logic:
  * Empty/null inputs (empty lists, None values, zero-length strings)
  * Boundary values (minimum/maximum sizes, first/last elements)
  * Invalid inputs (wrong types, out-of-range values, malformed data)
  * Special conditions (single element, all same values, already sorted/ordered)
  * Overflow/underflow conditions for numeric operations
- **Watch for common misunderstandings**:
  * Ambiguous requirements - clarify assumptions explicitly
  * Off-by-one errors in loops and indexing
  * Case sensitivity in string comparisons
  * Type coercion and implicit conversions
  * Default parameter behavior vs explicit values
- **Test edge cases explicitly** - Don't assume they're covered. Write test cases for each edge case you identify.
## Critical Requirements:
- Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
- Thoroughly check the entire codebase to ensure changes are exhaustive and don't break other functionality.
- If a test case fails several times consecutively, review it against the problem statement. If it's incorrect, fix the test file directly using `modify_test_case`.
- Do not create new files unless absolutely necessary.
- Always check both expected output mentioned in the problem statement AND the output in the most relevant test case.
- If you find errors while running the run_code tool due to missing dependencies, do not try to solve it as you don't have any internet access.
## Step Efficiency:
You have a limited step budget (target: 5 steps, maximum: 15 steps). Prioritize simpler, faster solutions and make forward progress with each step. Test frequently to catch issues early. Don't over-investigate - once you understand the issue, implement the fix.
You have access to the following tools:-
{tools_docs}
Here is the problem statement:
{problem_statement}
# Response Format Requirements
{format_prompt}
"""
)
# ============= Create Task Related Functions =============
def clean_code_response(response: str) -> str:
    """Clean code response by removing markdown code blocks for any language"""
    response = response.strip()
    response = re.sub(r"^```[\w-]*\n?", "", response, count=1)
    response = response.removesuffix("```").strip()
    return response
def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    def extract_file_names_using_llm(initial_solution: str) -> list:
        retry = 0
        while retry < 5:
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
                result, _ = EnhancedNetwork.make_request(messages=[{"role": "user", "content": file_names_prompt}], model=QWEN_MODEL_NAME)
                return json.loads(result.replace("```json", "").replace("```", "").strip())
            except Exception as e:
                retry += 1
                time.sleep(3)
        return []
    if not initial_solution.strip():
        return []
    
    file_names = extract_file_names_using_llm(initial_solution)
    # Ensure all file names are strings
    file_names = [str(f) for f in file_names if f]
    created_files = []
    current_file, content = None, []
    
    def write_file():
        if current_file and content:
            path = os.path.join(base_dir, current_file)
            dir_path = os.path.dirname(path)
            if dir_path:  # Only create directory if path has a directory component
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                file_content = '\n'.join(content)
                # Preserve structure, only strip trailing whitespace
                file_content = file_content.rstrip() + '\n' if file_content.strip() else file_content
                f.write(file_content)
            created_files.append(path)
            print(f"Created file: {path}")
    
    # Create a set for fast lookup - include both full paths and just filenames
    filename_set = set(file_names)
    for fname in file_names:
        # Also add just the filename part (in case line has "file.js" but file_names has "src/file.js")
        filename_set.add(fname.split('/')[-1])
    
    for line in initial_solution.split('\n'):
        stripped = line.strip()
        
        # Check if this line exactly matches any extracted filename
        if stripped in filename_set:
            write_file()
            # Use the original filename from file_names if available, otherwise use stripped
            current_file = next((f for f in file_names if f == stripped or f.endswith('/' + stripped) or f.split('/')[-1] == stripped), stripped)
            current_file, content = current_file, []
        elif current_file:
            content.append(line)
    
    write_file()
    return created_files
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
    selected_model = GLM_MODEL_NAME
    for retry in range(10):
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
                            if isinstance(concepts, dict) and "search_terms" in concepts:
                                return concepts
                        except json.JSONDecodeError:
                            pass
            return {"search_terms": [], "domain": "", "common_edge_cases": []}
        except Exception:
            if retry > 1:
                other_models = [m for m in AGENT_MODELS if m != selected_model]
                if other_models:
                    selected_model = random.choice(other_models)
            time.sleep(1)
    return {"search_terms": [], "domain": "", "common_edge_cases": []}
def generate_initial_solution(problem_statement: str, code_skeleton: str, temperature: float = 0.7) -> str:
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
        9. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
        10. **IMPORTANT**: Generate a docstring for every class or function. Each docstring should describe what the class/function does, its parameters (if applicable), return values (if applicable), and any important behavior or edge cases.
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
        {"role": "system", "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial files:\n{code_skeleton}\nGenerate the complete and correct implementation in files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
        },
    ]
    selected_model = QWEN_MODEL_NAME
    print("[GENERATE_INITIAL_SOLUTION] Requesting code generation from model")
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(
                code_generation_messages,
                model=selected_model,
                temperature=temperature
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
            time.sleep(2)
    if retry >= 10:
        return ""
    return ""
def get_run_command_for_file(file_path: str) -> list[str]:
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    retry = 0
    while retry < 5:
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
                    """
                }
            ]
            raw_text, _ = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            json_result = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
            command = json_result.get('command')
            if command and isinstance(command, list) and all(isinstance(arg, str) for arg in command):
                return command
        except Exception as e:
            time.sleep(2)
            retry += 1
    
    # Fallback: return a default command
    return ["./" + file_path]
def generate_single_testset(problem_statement: str, files_to_test: str, code_skeleton: str, temperature: float = 0.0) -> str:
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
            - **CRITICAL: Add useful comments to test cases**:
                - Add a comment above each test case explaining what it tests (e.g., "Test case: verify the function correctly handles [test case content]")
                - Add comments for edge cases explaining why they are important (e.g., "Edge case: tests [edge case content]")
                - Add comments for complex test logic explaining the test strategy (e.g., "This test verifies the function correctly handles multiple edge cases in sequence")
                - Use clear, descriptive comments that explain the purpose and expected behavior of each test
                - Comments should help developers understand what scenario is being tested and why it matters
            - **IMPORTANT**: Generate a docstring for every test case. Each docstring should describe what the test case does, its parameters (if applicable), return values (if applicable), and any important behavior or edge cases.
            
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
        {
            "role": "system",
            "content": GENERATE_TESTCASES_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```\n```javascript\ntest_a.js\ncontents of test_a.js\n\ntest_b.js\ncontents of test_b.js\n```"
        }
    ]
    selected_model = QWEN_MODEL_NAME
    
    while retry < 10:
        try:
            result = EnhancedNetwork.make_request(
                test_generation_messages,
                model=selected_model,
                temperature=temperature
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
                test_generation_messages.append({"role": "assistant", "content": testcode_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\n{{content}}\n\ntest_b.py\n{{content}}\n```\n```javascript\ntest_a.js\n{{content}}\n\ntest_b.js\n{{content}}\n```"})
                continue
            return testcases
            
        except Exception as e:
            retry += 1
            time.sleep(2)
    
    return ""
def write_one_solution(code_skeleton: str, problem_statement: str, temperature: float = 0.0) -> list:
    print(f"[Writing Solution] Starting with temperature={temperature}")
    
    # Continue loop until we get one initial solution
    initial_solution = None
    while not initial_solution:
        initial_solution = generate_initial_solution(problem_statement, code_skeleton, temperature)
        if not initial_solution:
            print("[Writing Solution] Failed to generate initial solution, retrying...")
            time.sleep(2)
    created_files = extract_and_write_files(initial_solution)    
    print(f"[Writing Solution] Initial solution:\n\n{initial_solution}")
    print(f"[Writing Solution] is finished")
    return created_files
def write_one_testset(code_skeleton: str, problem_statement: str, created_files: list, temperature: float = 0.0) -> list:
    print(f"[Writing Testset] Starting with temperature={temperature}")
    
    # Continue loop until we get one test_cases
    test_cases = None
    while not test_cases:
        test_cases = generate_single_testset(problem_statement, str(created_files), code_skeleton, temperature)
        if not test_cases:
            print("[Writing Testset] Failed to generate test cases, retrying...")
            time.sleep(2)
    
    test_files = extract_and_write_files(test_cases)
    print(f"[Writing Testset] Test cases:\n\n{test_cases}")
    print(f"[Writing Testset] is finished")
    return test_files
def get_files_to_modify(problem_statement: str) -> str:
    global run_id
    tool_manager = FixTaskEnhancedToolManager(
        available_tools = [
            "get_file_content",
            "list_directory_structure",
            "finish_find_files_to_fix"
        ]
    )
    FIND_FILES_TO_MODIFY = textwrap.dedent(
        """
        You are a helpful assistant that finds the files to modify related to the problem statement.
        You must check the directory structure using `list_directory_structure` tool and then determine which files are needed for the problem statement.            
        **IMPORTANT**: After finding all files to modify, you MUST call the `finish_find_files_to_fix` tool to signal the completion of the file finding workflow execution.
        Do NOT call `finish_find_files_to_fix` until you have identified all the files that need to be modified.
        You have access to the following tools:-
        {tools_docs}
        {format_prompt}
        """
    ).format(tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_FIND_FILES_TO_FIX)
    # Continue loop until we get a successful result
    while True:
        try:
            cot=EnhancedCOT(latest_observations_to_keep=10, summarize_batch_size=10)
            instance_prompt = f"Problem Statement:\n{problem_statement}"
            result = get_files_to_modify_workflow(
                cot,
                tool_manager,
                FIND_FILES_TO_MODIFY,
                instance_prompt,
                300,
                finish_tool_name="finish_find_files_to_fix",
            )
            logger.info(f"[GET_FILES_TO_MODIFY] Result: {result}")
            if not result:
                print("[GET_FILES_TO_MODIFY] No result returned, retrying...")
                time.sleep(2)
                continue
            
            if not isinstance(result, list):
                result = [result]
            
            contents = []
            valid_all_files_found = False
            for file_path in result:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        contents.append(f"{file_path}\n{{\n{f.read()}\n}}")
                        valid_all_files_found = True
                except Exception as e:
                    valid_all_files_found = False
                    logger.error(f"Failed to open file {file_path}: {e}")
            
            if valid_all_files_found and contents:
                return "\n\n".join(contents)
            else:
                print("[GET_FILES_TO_MODIFY] No valid files found, retrying...")
                time.sleep(2)
                continue
                
        except Exception as e:
            logger.error(f"Error in get files to modify: {e}")
            print("[GET_FILES_TO_MODIFY] Exception occurred, retrying...")
            time.sleep(2)
            continue
def get_files_to_modify_workflow(cot: EnhancedCOT, tool_manager: EnhancedToolManager, system_prompt: str, instance_prompt: str, timeout: int, finish_tool_name="finish_find_files_to_fix") -> str:
    global run_id, START_TIME
    
    logger.info(f"[GET_FILES_TO_MODIFY] Starting file finding workflow... ")
    def finding_workflow(step):
        if time.time() - START_TIME > timeout:
            return False, ""
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
             
        # Retry inference up to 3 times and dfs search
        for retry in range(3):
            temperature=0.0
            try:
                selected_model = GLM_MODEL_NAME
                next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages, used_model = EnhancedNetwork.inference(
                    messages, 
                    model=selected_model, 
                    run_id=run_id, 
                    temperature=temperature
                )               
                # Check if inference returned valid response
                if next_thought is None:
                    temperature=0.7
                    continue                    
            except Exception as e:
                logger.warning(f"Inference error on retry {retry + 1}: {e}")
                temperature=0.7
                continue
            # Process the valid inference response
            tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
            tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]
            all_observations = []
            success_observation_count = 0
            error_observation_count = 0
            for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
                try:
                    if '"' in tool_name or "'" in tool_name:
                        tool_name = tool_name.replace('"', '').replace("'", "")
                    tool_func = tool_manager.get_tool(tool_name)
                    if isinstance(tool_func, str):
                        # get_tool returned an error string
                        observation = tool_func
                    elif tool_args:
                        observation = tool_func(**tool_args)
                    else:
                        observation = tool_func()
                    all_observations.append(observation)
                    success_observation_count += 1
                except EnhancedToolManager.Error as e:
                    error_msg = f"Tool {idx + 1} ({tool_name}) error: {e.message}"
                    all_observations.append(error_msg)
                    error_observation_count += 1
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    error_msg = f"Tool {idx + 1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                    all_observations.append(error_msg)
                    error_observation_count += 1
            
            # Retry if more errors than successes
            if error_observation_count >= success_observation_count:
                temperature=0.7
                continue
            
            if finish_tool_name in tool_names_list:
                logger.info(f"[GET_FILES_TO_MODIFY] Finish tool called: {finish_tool_name}")
                for name, obs in zip(tool_names_list, all_observations):
                    if name == finish_tool_name:
                        logger.info(f"[GET_FILES_TO_MODIFY] Finish tool called: {finish_tool_name}")   
                        logger.info(f"[GET_FILES_TO_MODIFY] Finish tool observation: {obs}")
                        return True, obs
            
            # Combine observations
            if len(all_observations) == 1:
                combined_observation = all_observations[0]
            else:
                combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                    [
                        f"Tool {i + 1} ({tool_names_list[i]}):\n{obs}"
                        for i, obs in enumerate(all_observations)
                    ]
                )
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,  # Keep original format (list or single)
                    next_tool_args=next_tool_args,
                    observation=combined_observation,
                    is_error=not error_observation_count == 0,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages
                )
            )
            
            # Recursively continue search
            is_found, obs = finding_workflow(step + 1)
            if is_found:
                return True, obs
        return False, ""
    
    is_found, obs = finding_workflow(0)
    if is_found:
        return obs
    else:
        return ""
def create_task_solve_workflow(problem_statement: str,  timeout: int, enhancement: str):
    print("[CREATE_TASK_SOLVE_WORKFLOW] Initializing workflow")
    global run_id, START_TIME 
    
    cot = EnhancedCOT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE
    )
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "list_directory_structure",
            "get_file_content",
            "apply_code_edit",
            "modify_test_case",
            "run_shell_cmd",
            "create_new_file",
            "run_code",
            "finish"
        ]
    )
    system_prompt = CREATE_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_CREATE_TASK
    )
    enhanced_problem = problem_statement
    if enhancement:
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement
    # Progress tracking for multi-file changes
    modified_files = set()    
    def workflow(step: int):
        """
        DFS-based recursive workflow function, try to do complete search.
        Returns True if terminating condition detected for both of timeout or finish tool call, False otherwise.
        """        
        # Base case: check timeout
        if time.time() - START_TIME > timeout:
            print("[CREATE_TASK_SOLVE_WORKFLOW] Timeout reached")
            return True
        
        print(f"[CREATE_TASK_SOLVE_WORKFLOW] Starting workflow step {step}")
    
        instance_prompt = INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        temperature = 0.0
        repeated_thoughts_count = cot.count_repeated_thoughts()
        if repeated_thoughts_count == 2:
            # Remove the last thought to backtrack and try a different approach
            if cot.thoughts:
                cot.thoughts.pop()
            return False
        temperature = 0.7
        if repeated_thoughts_count == 1:
            if cot.thoughts:
                last_thought = cot.thoughts[-1]
                messages.append(
                    {"role": "user", "content": MAKE_REAL_PROGRESS.format(
                        previous_response=f"next_thought:{last_thought.next_thought}\n next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    )}
                )
                temperature = 0.7
        tried_thoughts = []
        for retry in range(3):
            if time.time() - START_TIME > timeout:
                print("[CREATE_TASK_SOLVE_WORKFLOW] Timeout reached")
                return True
            try:
                next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages, used_model = EnhancedNetwork.inference(
                    messages,
                    model=GLM_MODEL_NAME,
                    run_id=run_id,
                    temperature=temperature
                )
            except Exception as e:
                temperature=0.0
                logger.error(f"[CREATE_TASK_SOLVE_WORKFLOW] Inference failed on retry {retry + 1}: {e}")
                continue
            if time.time() - START_TIME > timeout:
                logger.info("[CREATE_TASK_SOLVE_WORKFLOW] Timeout reached")
                return True
            if next_thought is None:
                temperature=0.0
                continue
            if next_thought in tried_thoughts:
                temperature=0.7
                continue
            tried_thoughts.append(next_thought)
            tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
            tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]
            all_observations = []
            success_observation_count = 0
            error_observation_count = 0
            for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
                try:
                    if '"' in tool_name or "'" in tool_name:
                        tool_name = tool_name.replace('"', '').replace("'", "")
                    print("[CREATE_TASK_SOLVE_WORKFLOW] Executing tool")
                    tool_func = tool_manager.get_tool(tool_name)
                    if isinstance(tool_func, str):
                        # get_tool returned an error string
                        observation = tool_func
                    elif tool_args:
                        observation = tool_func(**tool_args)
                    else:
                        observation = tool_func()
                    # Track file modifications for multi-file change awareness
                    if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
                        file_path = tool_args["file_path"]
                        if "ok, code edit applied successfully" in str(observation).lower():
                            modified_files.add(file_path)
                    all_observations.append(observation)
                    success_observation_count += 1
                except EnhancedToolManager.Error as e:
                    error_msg = f"Tool {idx + 1} ({tool_name}) error: {e.message}"
                    all_observations.append(error_msg)
                    error_observation_count += 1
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    error_msg = f"Tool {idx + 1} ({tool_name}) exception: {str(e)}\n{error_traceback}"
                    all_observations.append(error_msg)
                    error_observation_count += 1
            if error_observation_count >= success_observation_count:
                if temperature == 0.0:
                    temperature = 0.7
                else:
                    temperature = 0.0
                continue
            # Combine observations
            if len(all_observations) == 1:
                combined_observation = all_observations[0]
            else:
                combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                    [
                        f"Tool {i + 1} ({tool_names_list[i]}):\n{obs}"
                        for i, obs in enumerate(all_observations)
                    ]
                )
            show_transition(step, next_thought, next_tool_name, next_tool_args, combined_observation)
            finish_called = any(name == "finish" for name in tool_names_list)
            if finish_called:
                return True
            
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,  # Keep original format (list or single)
                    next_tool_args=next_tool_args,
                    observation=combined_observation,
                    is_error=not error_observation_count == 0,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages
                )
            )
            # Recursive call to next step (DFS)
            is_finished = workflow(step + 1)
            if is_finished:
                return True
            # If not finished, continue to next retry (backtracking in DFS)
            if time.time() - START_TIME > timeout:
                return True
        return False
    
    workflow(0)
def write_files(created_files_contents: List[tuple[str, str | None]]):
    # Step 3: Rewrite contents of files in created_files
    for file_path, content in created_files_contents:
        if content is not None:
            try:
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Error writing file {file_path}: {e}")
def get_files_content(created_files: List[str]) -> List[tuple[str, str | None]]:
    
    # Step 1: Get all contents of files in created_files and save it to a list
    created_files_contents = []
    for file_path in created_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    created_files_contents.append((file_path, f.read()))
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            created_files_contents.append((file_path, None))
    return created_files_contents
def initialize_repo():
    try:
        os.system("git reset --hard")
        os.system("git clean -fd")
    except Exception as e:
        logger.error(f"Error running git commands: {e}")
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
    while retry < 5:
        try:
            result, _ = EnhancedNetwork.make_request(messages=[{"role": "user", "content": check_all_tests_passed_prompt.format(output=output)}], model=QWEN_MODEL_NAME)
            print(f"[IS_ALL_TESTS_PASSED]: {result}")
            if result.lower() == "true":
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"[IS_ALL_TESTS_PASSED] Exception: {e}")
            retry += 1
            time.sleep(2)
    return False
def basic_approach(code_skeleton: str, problem_statement: str, temperature: float = 0.0) -> tuple[str, str] | tuple[None, None]:
    print(f"[BASIC_APPROACH] Starting basic_approach with temperature={temperature}")
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, temperature)
    if not initial_solution:
        print("[BASIC_APPROACH] Failed to generate initial solution")
        return (None, None)
    created_files = extract_and_write_files(initial_solution)
    print(f"[BASIC_APPROACH] Initial solution:\n\n {initial_solution}")
    print(f"[BASIC_APPROACH] Created {len(created_files)} files from initial solution")
    test_cases = generate_single_testset(problem_statement, str(created_files), code_skeleton, temperature)
    print(f"[BASIC_APPROACH] Test cases:\n\n {test_cases}")
    if not test_cases:
        print("[BASIC_APPROACH] Failed to generate test cases")
        return (None, None)
    test_files = extract_and_write_files(test_cases)
    print(f"[BASIC_APPROACH] Created {len(test_files)} test files")
    print("[BASIC_APPROACH] Running generated test cases")
    for file in test_files:
        try:
            # Get the appropriate command for the file type
            run_command = get_run_command_for_file(file)
            print(f"[BASIC_APPROACH] Running command: {' '.join(run_command)}")
            
            result = subprocess.run(
                run_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30
            )
            print(f"[BASIC_APPROACH] Test result:\n\n {result.stdout}")
        except subprocess.TimeoutExpired as e:
            print(f"[BASIC_APPROACH] Test execution timed out after 30s for file: {file}")
            return (None, None)
        except ValueError as e:
            print(f"[BASIC_APPROACH] Unsupported file type: {e}")
            return (None, None)
        except Exception as e:
            print(f"[BASIC_APPROACH] Error running test file {file}: {e}")
            return (None, None)
        if not is_all_tests_passed(result.stdout):
            print(f"[BASIC_APPROACH] Tests failed in {file}")
            return (None, None)
    print("[BASIC_APPROACH] All tests passed successfully")
    return (initial_solution, test_cases)
    
def process_create_task(input_dict: Dict[str, Any], enhancement: str):
        
    problem_statement = input_dict.get("problem_statement", "")
    code_skeleton = get_files_to_modify(problem_statement)
     
    time_limit_for_workflow = 1200
    success_count = 0
    initial_solutions = []
    test_cases_lists = []
    while success_count < 1:
        initial_solution, test_cases = basic_approach(code_skeleton, problem_statement, temperature=0.0)
        if initial_solution is not None:
            success_count += 1
            initial_solutions.append(initial_solution)
            test_cases_lists.append(test_cases)
        time.sleep(5)
        print(f"success_count: {success_count}")
        if time.time() - START_TIME > time_limit_for_workflow:
            break
    
    print(f"[PROCESS CREATE TASK] success_count: {success_count}")
    initialize_repo()
    if success_count == 0:
        created_files = write_one_solution(code_skeleton, problem_statement, temperature=0.7)
        test_files = write_one_testset(code_skeleton, problem_statement, created_files, temperature=0.0)
    else:
        created_files = extract_and_write_files(initial_solutions[0])
        test_files    = extract_and_write_files(test_cases_lists[0])
    # Get misunderstanding points and append to problem statement
    misunderstanding_point = get_misunderstanding_point(problem_statement, code_skeleton)
    if misunderstanding_point:
        logger.info(f"[PROCESS_CREATE_TASK] Misunderstanding point: {misunderstanding_point}")
        problem_statement = problem_statement + "\n\n--- Misunderstanding Points Analysis ---\n" + misunderstanding_point
    global MODIFIED_SOLUTION_FILES_CONTENTS
    MODIFIED_SOLUTION_FILES_CONTENTS = get_files_content(created_files)
    count = 0
    while count < 1:        
        create_task_solve_workflow(problem_statement, 1300, enhancement = enhancement)
        count += 1
    print(f"[PROCESS CREATE TASK] Create task work flow successfully finished")
    if time.time() - START_TIME > 1370:
        return 
    MODIFIED_SOLUTION_FILES_CONTENTS = get_files_content(created_files)
    
# ==================== Debug Functions ====================
def show_transition(step: int, next_thought: str, next_tool_name: str, next_tool_args: dict, combined_observation: str):
        # Print step information in a formatted way
    print("\n" + "=" * 80)
    print(f"╔{'═' * 78}╗")
    print(f"║{'STEP ' + str(step):^78}║")
    print(f"╚{'═' * 78}╝")
    print(f"\n📝 NEXT_THOUGHT:")
    print(f"{'─' * 80}")
    print(f"{next_thought}")
    print(f"{'─' * 80}")
    
    # Handle tool_name (can be list or single)
    tool_name_display = next_tool_name
    if isinstance(next_tool_name, list):
        tool_name_display = ", ".join(next_tool_name)
    
    print(f"\n🔧 NEXT_TOOL_NAME:")
    print(f"{'─' * 80}")
    print(f"{tool_name_display}")
    print(f"{'─' * 80}")
    
    # Handle tool_args (can be list or single)
    tool_args_display = next_tool_args
    if isinstance(next_tool_args, list):
        tool_args_display = "\n".join([f"  Tool {i+1}: {json.dumps(args, indent=2, ensure_ascii=False)}" for i, args in enumerate(next_tool_args)])
    else:
        tool_args_display = json.dumps(next_tool_args, indent=2, ensure_ascii=False)
    
    print(f"\n⚙️  NEXT_TOOL_ARGS:")
    print(f"{'─' * 80}")
    print(f"{tool_args_display}")
    print(f"{'─' * 80}")
    
    # observation_display = str(combined_observation)
    
    # print(f"\n👁️  OBSERVATION:")
    # print(f"{'─' * 80}")
    # print(f"{observation_display}")
    # print(f"{'─' * 80}")
    print("=" * 80 + "\n")
def show_messages(step: int, messages: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(f"╔{'═' * 78}╗")
    print(f"║{'INFERENCE MESSAGES (Step ' + str(step) + ')':^78}║")
    print(f"╚{'═' * 78}╝")
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")          
        print(f"\n[{i}] Role: {role.upper()}")
        print(f"{'─' * 80}")
        print(f"{content}")
        print(f"{'─' * 80}")      
    print("=" * 80 + "\n")