"""
Shared utilities for LLM arena and puzzle mining.
"""

import re
import subprocess
import os
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Known pricing per 1M tokens: (input, output) in USD
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-5-20251101": (15.0, 75.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20250514": (0.80, 4.0),
    # OpenAI
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.2-pro": (21.0, 168.0),
    "gpt-5-mini": (0.25, 2.0),
    # Google
    "gemini-3-pro-preview": (2.0, 12.0),
    "gemini-3-flash": (0.50, 3.0),
}

# Models routed through OpenRouter: local name -> OpenRouter model ID
OPENROUTER_MODELS = {
    "deepseek-v3.2-thinking": "deepseek/deepseek-v3.2",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
}

# Global token tracking by model: {model_name: {"input_tokens": N, "output_tokens": N}}
token_usage = {}


def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate cost in USD if pricing is known, else return None."""
    if model_name in MODEL_PRICING:
        input_price, output_price = MODEL_PRICING[model_name]
        return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return None


def format_usage(model_name, input_tokens, output_tokens):
    """Format usage string with cost if known."""
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    if cost is not None:
        return f"{input_tokens:,} in / {output_tokens:,} out (${cost:.2f})"
    return f"{input_tokens:,} in / {output_tokens:,} out"


def get_llm(model):
    """
    Get a LangChain LLM instance based on the model string.

    model options:
      - local ollama models like: "llama3:70b"
      - OpenAI models like: "gpt-5.2", "gpt-4", "gpt-4o", etc.
      - Anthropic models like: "claude-4.5-opus", "claude-sonnet-4.5", etc.
    """
    # --- Local / Ollama ---
    if ":" in model:  # heuristic for local names like llama3:70b
        # num_predict=-1 means unlimited tokens
        return ChatOllama(model=model, base_url="http://localhost:11434", num_predict=-1)

    # On limits: Seems Claude insists on a limit <= 64,000, so let's
    # impose this limit on everyone for fairness.

    # --- OpenAI models ---
    elif model.startswith("gpt-") or model.startswith("o4"):
        # mini models only support up to "high"
        effort = "high" if ("mini" in model or model.startswith("o4")) else "xhigh"
        return ChatOpenAI(model=model, max_tokens=96000, output_version="responses/v1", reasoning_effort=effort)

    # --- Anthropic models ---
    elif model.startswith("claude-"):
        # 4-7+ require adaptive thinking; control budget via output_config.effort.
        # 4-6 and older accept thinking.type=enabled with an explicit budget.
        if "4-7" in model or "4-8" in model or "4-9" in model:
            return ChatAnthropic(
                model=model,
                max_tokens=64000,
                thinking={"type": "adaptive"},
                model_kwargs={"output_config": {"effort": "high"}},
            )
        return ChatAnthropic(model=model, max_tokens=64000, thinking={"type": "enabled", "budget_tokens": 56000})

    # Google models
    elif model.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model, max_tokens=64000, thinking_budget=-1)

    elif model.startswith("grok-"):
        return ChatXAI(model=model, max_tokens=64000)

    elif model.startswith("deepseek-") and model not in OPENROUTER_MODELS:
        return ChatDeepSeek(model=model, max_tokens=64000)

    elif model in OPENROUTER_MODELS:
        return ChatOpenAI(
            model=OPENROUTER_MODELS[model],
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            max_tokens=140000,
            extra_body={"reasoning": {"effort": "high"}},
        )

    else:
        raise ValueError(f"Unknown model: {model}")


def invoke_with_tracking(model, prompt, model_name):
    """Invoke model and track token usage. Returns (content, raw_response, usage_dict)."""
    response = model.invoke(prompt)
    content = StrOutputParser().invoke(response)

    usage = {"input_tokens": 0, "output_tokens": 0}

    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        usage["input_tokens"] = um.get("input_tokens", 0) or 0
        usage["output_tokens"] = um.get("output_tokens", 0) or 0
        if usage["input_tokens"] == 0:
            usage["input_tokens"] = um.get("prompt_tokens", 0) or 0
        if usage["output_tokens"] == 0:
            usage["output_tokens"] = um.get("completion_tokens", 0) or 0
        # Some providers (xAI/grok) report reasoning tokens separately
        output_details = um.get("output_token_details") or {}
        reasoning = output_details.get("reasoning", 0) or 0
        if reasoning > 0 and reasoning > usage["output_tokens"]:
            usage["output_tokens"] += reasoning

    # Fallback: check response_metadata
    if usage["input_tokens"] == 0 and hasattr(response, 'response_metadata') and response.response_metadata:
        rm = response.response_metadata
        if "usage" in rm:
            usage["input_tokens"] = rm["usage"].get("input_tokens", 0) or rm["usage"].get("prompt_tokens", 0) or 0
            usage["output_tokens"] = rm["usage"].get("output_tokens", 0) or rm["usage"].get("completion_tokens", 0) or 0
        if "usage_metadata" in rm:
            usage["input_tokens"] = rm["usage_metadata"].get("input_tokens", 0) or rm["usage_metadata"].get("prompt_tokens", 0) or 0
            usage["output_tokens"] = rm["usage_metadata"].get("output_tokens", 0) or rm["usage_metadata"].get("completion_tokens", 0) or 0

    if usage["input_tokens"] == 0 and usage["output_tokens"] == 0:
        print(f"  [DEBUG] No token usage found for {model_name}. Response attrs: {[a for a in dir(response) if not a.startswith('_')]}")
        if hasattr(response, 'usage_metadata'):
            print(f"  [DEBUG] usage_metadata: {response.usage_metadata}")
        if hasattr(response, 'response_metadata'):
            print(f"  [DEBUG] response_metadata: {response.response_metadata}")

    if hasattr(response, "model_dump"):
        response = response.model_dump()

    # Update global tracking
    if model_name not in token_usage:
        token_usage[model_name] = {"input_tokens": 0, "output_tokens": 0}
    token_usage[model_name]["input_tokens"] += usage["input_tokens"]
    token_usage[model_name]["output_tokens"] += usage["output_tokens"]

    return content, response, usage


def get_reasoning_tokens(raw):
    """Extract reasoning token count from a raw response dict."""
    if not isinstance(raw, dict):
        return 0
    # Google path stores it directly
    if "reasoning_tokens" in raw:
        return raw["reasoning_tokens"]
    # LangChain model_dump path: usage_metadata.output_token_details.reasoning
    um = raw.get("usage_metadata") or {}
    output_details = um.get("output_token_details") or {}
    reasoning = output_details.get("reasoning", 0) or 0
    if reasoning:
        return reasoning
    # Fallback: response_metadata.token_usage.completion_tokens_details.reasoning_tokens
    rm = raw.get("response_metadata") or {}
    tu = rm.get("token_usage") or {}
    ctd = tu.get("completion_tokens_details") or {}
    reasoning = ctd.get("reasoning_tokens", 0) or 0
    if reasoning:
        return reasoning
    # OpenRouter: reasoning in additional_kwargs
    ak = raw.get("additional_kwargs") or {}
    reasoning_obj = ak.get("reasoning") or {}
    if isinstance(reasoning_obj, dict):
        reasoning_content = reasoning_obj.get("content") or []
        reasoning_text = "".join(
            (item.get("text") or "") for item in reasoning_content
            if isinstance(item, dict)
        )
        if reasoning_text:
            return max(1, len(reasoning_text) // 4)
    # Anthropic: count tokens in "thinking" content blocks
    blocks = raw.get("content") or []
    if isinstance(blocks, list):
        thinking_text = "".join(
            (b.get("thinking") or "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        )
        if thinking_text:
            # Approximate: ~4 chars per token
            return max(1, len(thinking_text) // 4)
    return 0


def promptSolvePuzzle(puzzle):
    """Generate a prompt asking to solve a puzzle."""
    return f"Here's a Python function that takes a value x and returns a boolean. Please give me a value for x such that mystery(x) is True. The last line of your response should be: `SOLUTION: x` where x is the value. Ensure that `mystery(x)` is valid Python code given your x. For example, if you believe `x` is a string, use quotes. Example: `SOLUTION: \"Hello, world!\"`\n\n```python\n{puzzle}\n```"


def findSolution(response):
    """Extract solution from model response."""
    pattern = r'SOLUTION:\s*(.*?)\s*(?:\n|$)'
    match = re.search(pattern, response)
    if match:
        return str(match.group(1))
    else:
        raise ValueError("No SOLUTION line found in the input string.")


def checkSolution(code, value):
    """Check if a solution is correct by running it."""
    program = f"""
import importlib

{code}

result = mystery({value})
print("TRUE" if result else "FALSE")
    """

    sandbox = False if os.environ.get('SANDBOX', '1') == '0' else True

    try:
        if not sandbox:
            process = subprocess.run(
                ["python3", "-"],
                input=program,
                capture_output=True,
                text=True,
                timeout=5,
            )
        else:
            process = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--memory=256m",
                    "--pids-limit=64",
                    "--cpus=1",
                    "--read-only",
                    "-i",
                    "python:3.12",
                    "python", "-"
                ],
                input=program,
                capture_output=True,
                text=True,
                timeout=5,
            )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"mystery() execution timed out after {e.timeout}s") from e

    if process.returncode != 0:
        raise RuntimeError(f"Error running mystery():\n{process.stderr}")
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No output from subprocess.")
    last_line = lines[-1]
    if last_line.startswith("TRUE"):
        return True
    elif last_line.startswith("FALSE"):
        return False
    raise ValueError(f"Expected last line to be either 'TRUE' or 'FALSE', got:\n{last_line}")
