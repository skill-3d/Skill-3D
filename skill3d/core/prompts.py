"""
Unified Prompt Templates

This module contains prompt templates for the SPAgent system.
"""

from typing import List, Dict, Any
import json
import os


def create_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Create system prompt with available tools
    
    Args:
        tools: List of tool function schemas
        
    Returns:
        System prompt string
    """
    if not tools:
        return """You are a helpful assistant that can analyze images and answer questions."""
    
    tools_json = json.dumps(tools, indent=2)
    
    return f"""You are a helpful assistant that can analyze images and answer questions.

# Tools
You have access to the following tools to assist with user queries:
<tools>
{tools_json}
</tools>

# How to call a tool
When you need to use a tool, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

You can call multiple tools if needed by using multiple <tool_call> blocks.

# SFT Rollout Tag Protocol
This rollout may be used as supervised multi-turn training data. The tag format is mandatory.
- The FIRST non-whitespace token of every assistant response MUST be `<think>`.
- Put all reasoning text inside exactly one paired `<think>...</think>` block.
- Do not write any natural-language reasoning before `<think>` or after the final closing tag.
- If you need more evidence, use this shape:
  <think>reasoning for this turn</think>
  <tool_call>{{"name":"tool_name","arguments":{{...}}}}</tool_call>
- If you are ready to answer, use this shape:
  <think>reasoning for the final decision</think>
  <answer>final option/number only</answer>
- Never output `<tool_choice>`.

# Output Protocol
- Every response turn must start with <think></think>; no text is allowed before `<think>`.
- If you need more evidence, output one or more <tool_call></tool_call> blocks after <think>.
- If you are ready to answer, output <answer></answer> after <think>.
- Never output <tool_call> and <answer> in the same response.
- Do not use <tool_choice>; executable tool decisions must always use <tool_call>.

# Tool Use Policy
Decide which tools to use based on your own reasoning.
- Use as many or as few tool calls as needed.
- Avoid repeating the same tool call unless new parameters or new images may change the evidence.
- Stop when additional tool calls are unlikely to change the answer.

# Multi-Step Workflow
You can perform MULTIPLE rounds of tool calls and analysis. Stop early when additional tool calls will not change the answer.

Pi3 usage notes (only if you decide to call pi3_tool):
- The input image(s) already show the scene at (azimuth=0°, elevation=0°). If you want NEW evidence, do not call pi3_tool with (0°, 0°).
- Prefer coarse new angles first (e.g., azimuth ±45/±90, elevation 30–60, back ±135/180).
- If you have multiple input images: use rotation_reference_camera=1,2,3,... to rotate around different camera positions.
- Use camera_view=true only when first-person evidence is needed; otherwise keep camera_view=false.

Only put the final option/number in <answer></answer> tags (no extra text).
"""


def create_follow_up_prompt(question: str, initial_response: str, tool_results: Dict[str, Any], original_images: List[str], additional_images: List[str], description: str=None) -> str:
    """
    Create follow-up prompt after tool execution
    
    Args:
        question: Original user question
        initial_response: Model's initial response
        tool_results: Results from tool execution
        original_images: List of original image paths
        additional_images: List of additional image paths from tools
        description: Optional description from tool execution
        
    Returns:
        Follow-up prompt string
    """
    tool_summary = []
    for tool_name, result in tool_results.items():
        if result.get('success'):
            tool_summary.append(f"- {tool_name}: Successfully executed")
        else:
            tool_summary.append(f"- {tool_name}: Failed - {result.get('error', 'Unknown error')}")
    
    original_images_info = "\n".join([f"- {path}" for path in original_images])
    additional_images_info = "\n".join([f"- {path}" for path in additional_images]) if additional_images else "None"
    
    # 构建基本的 prompt
    prompt = f"""Based on the tool results, please provide a comprehensive answer to the original question.

Original Images:
{original_images_info}

Additional Images from Tools:
{additional_images_info}

Original Question: {question}

Your Initial Analysis: {initial_response}

Tool Execution Summary:
{chr(10).join(tool_summary)}"""

    # 只有当提供了 description 时才添加 Tool Description 部分
    if description is not None:
        prompt += f"""

Tool Description: {description}"""

    prompt += """

Now please provide a detailed final answer that incorporates the tool results with your initial analysis. If tools provided additional images or data, reference them in your response.

**If you decide to call pi3_tool again**: the original input image(s) are already at (0°,0°), so use a different angle if you want new evidence. With multiple input images, you can also try different `rotation_reference_camera` values. Use `camera_view=True` only when first-person evidence is useful.

You MUST output your thinking process in <think></think> and final choice in <answer></answer>. 
"""

    return prompt

def create_user_prompt(question: str, image_paths: List[str], tool_schemas: List[Dict[str, Any]] = None) -> str:
    """
    Create user prompt template
    
    Args:
        question: User's question
        image_paths: List of image paths to analyze
        tool_schemas: List of tool function schemas, optional
    Returns:
        Formatted user prompt
    """
    images_info = "\n".join([f"- {path}" for path in image_paths])
    base_prompt = f"""Please analyze the following image(s):

Images to analyze:
{images_info}

Question:
{question}

Think step by step to analyze the question and provide a detailed answer."""

    prompt_mode = str(os.environ.get("SPAGENT_SKILL_PROMPT_MODE", "")).strip().lower()
    suppress_user_protocol = str(os.environ.get("SPAGENT_SUPPRESS_USER_FORMAT_PROMPT", "")).strip().lower()
    if not suppress_user_protocol:
        suppress_user_protocol = "1" if prompt_mode in {"compact", "sft", "rollout", "sft_rollout"} else "0"
    suppress_user_protocol = suppress_user_protocol not in {"0", "false", "no", "off"}

    if tool_schemas and not suppress_user_protocol:
        base_prompt += """

Important Notes:
- You can call tools MULTIPLE times with different parameters to gather comprehensive information
- After each tool execution, you'll see the results and can decide if you need more information
- Only provide your final <answer></answer> when you have gathered sufficient information

Required response format:
- Start the response immediately with <think>; do not put any text before it.
- Tool turn: <think>...</think><tool_call>{...}</tool_call>
- Final turn: <think>...</think><answer>...</answer>

You MUST output your thinking process in <think></think> and executable tool calls in <tool_call></tool_call>. When you have enough information, output your final choice in <answer></answer>. Never use <tool_choice>. Only put Options in <answer></answer> tags, do not put any other text."""
    elif tool_schemas:
        base_prompt += """

Use the system-level Skill-3D protocol and available tools when evidence is needed."""
    else:
        base_prompt += """

You MUST output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text."""

    return base_prompt 


def create_fallback_prompt(question: str, initial_response: str) -> str:
    """
    Create fallback prompt when tools fail but initial response lacks <answer> tags
    
    Args:
        question: Original question
        initial_response: Initial model response
        
    Returns:
        Fallback prompt string
    """
    return f"""The tools could not be executed successfully. Based on your initial analysis, please provide a final answer to the question.

Original Question: {question}

Your Initial Analysis: {initial_response}

Since the tools are unavailable, please provide your best answer based on the original image analysis alone.

You MUST output your thinking process in <think></think> and final choice in <answer></answer>. Only put Options (A,B,C,D) in <answer></answer> tags, do not put any other text.
"""
