"""
core/llm.py
Groq client wrapper — unified interface for all agents.
Token-budget-aware: tracks spend and warns before hitting limits.
"""
from __future__ import annotations
import os, time, json, re
from typing import Any, Optional
from groq import Groq


def _load_config() -> dict:
    import yaml, pathlib
    cfg_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
LLM_CFG = CFG["llm"]

# Groq free-tier limit per minute for 70b model is 6000 tokens/min,
# but the per-request context window is 128k. We cap prompts to keep well under.
TOKEN_BUDGET_WARN = 80_000   # warn in logs after this many total tokens
TOKEN_BUDGET_HARD = 95_000   # switch to fast model after this (safety valve)


def _trim_messages(messages: list[dict], max_chars: int = 6000) -> list[dict]:
    """
    Hard-trim the user message content so a single prompt never
    exceeds max_chars characters (≈ 1500 tokens). Keeps system prompt intact.
    """
    trimmed = []
    for msg in messages:
        if msg["role"] == "system":
            trimmed.append(msg)
        else:
            content = msg.get("content", "")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[...truncated for token budget...]"
            trimmed.append({**msg, "content": content})
    return trimmed


class ForgeGroqClient:
    """Singleton Groq client shared across all agents."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Please set it via:\n"
                "  import os; os.environ['GROQ_API_KEY'] = 'gsk_...'"
            )
        self.client = Groq(api_key=api_key)
        self.total_tokens = 0
        self.total_calls = 0

    def call(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        agent_name: str = "unknown",
        expect_json: bool = False,
    ) -> tuple[str, dict]:
        """
        Make a Groq API call.
        Returns (response_text, usage_dict).
        Auto-trims prompts and falls back to fast model near token budget.
        """
        # Token budget safety valve — fall back to fast model if close to limit
        if self.total_tokens >= TOKEN_BUDGET_HARD and model == LLM_CFG["codegen_model"]:
            model = LLM_CFG["fast_model"]
            max_tokens = LLM_CFG.get("max_tokens_fast", 1024)

        model = model or LLM_CFG["codegen_model"]
        temperature = temperature if temperature is not None else LLM_CFG["temperature_codegen"]

        # Use smaller cap for fast model
        if model == LLM_CFG["fast_model"]:
            max_tokens = max_tokens or LLM_CFG.get("max_tokens_fast", 1024)
        else:
            max_tokens = max_tokens or LLM_CFG["max_tokens"]

        # Trim prompts to avoid blowing context
        messages = _trim_messages(messages, max_chars=5000)

        t0 = time.time()
        # httpx timeout: (connect_timeout, read_timeout)
        # connect_timeout catches stalled connections fast.
        # read_timeout is generous — a legitimate 2048-token response can take 60s+
        # on a loaded model. We never want to cut a response mid-generation.
        CONNECT_TIMEOUT = 15   # seconds to establish connection
        READ_TIMEOUT    = 180  # seconds to wait for full response (3 min max)

        def _do_call(mdl, msgs, temp, max_tok):
            import httpx
            return self.client.chat.completions.create(
                model=mdl,
                messages=msgs,
                temperature=temp,
                max_tokens=max_tok,
                timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            )

        try:
            response = _do_call(model, messages, temperature, max_tokens)
        except Exception as e:
            err_str = str(e)
            # Rate limit — wait 20s and retry on fast model
            if "rate_limit" in err_str.lower() or "429" in err_str:
                log_msg = f"[{agent_name}] Rate limit hit — waiting 20s then retrying on fast model"
                print(log_msg)
                time.sleep(20)
                response = _do_call(
                    LLM_CFG["fast_model"], messages, temperature,
                    LLM_CFG.get("max_tokens_fast", 1024),
                )
            # Connection timeout (stalled connection, not slow generation) — retry once
            elif "connect" in err_str.lower() and "timeout" in err_str.lower():
                print(f"[{agent_name}] Connection timeout — retrying once")
                time.sleep(5)
                response = _do_call(model, messages, temperature, max_tokens)
            else:
                raise

        latency = time.time() - t0
        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_s": round(latency, 3),
            "model": model,
            "agent": agent_name,
        }
        self.total_tokens += usage["total_tokens"]
        self.total_calls += 1

        if self.total_tokens >= TOKEN_BUDGET_WARN:
            print(f"[TokenBudget] WARNING: {self.total_tokens:,} tokens used — approaching limit")

        if expect_json:
            text = _extract_json(text)

        return text, usage

    def call_reasoning(self, messages: list[dict], agent_name: str = "unknown",
                       max_tokens: int = None) -> tuple[str, dict]:
        """Use the planning model (same as codegen, just lower temperature)."""
        return self.call(
            messages,
            model=LLM_CFG["planning_model"],
            temperature=LLM_CFG["temperature_reasoning"],
            max_tokens=max_tokens or LLM_CFG["max_tokens"],
            agent_name=agent_name,
        )

    def call_fast(self, messages: list[dict], agent_name: str = "unknown",
                  max_tokens: int = None) -> tuple[str, dict]:
        """Use the fast lightweight model — for boilerplate and non-Python files."""
        return self.call(
            messages,
            model=LLM_CFG["fast_model"],
            temperature=0.1,
            max_tokens=max_tokens or LLM_CFG.get("max_tokens_fast", 1024),
            agent_name=agent_name,
        )


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract JSON from LLM output."""
    # Try to find JSON block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object/array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, with fallback."""
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Second attempt: fix common LLM JSON mistakes
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse LLM JSON response: {e}\nRaw: {text[:500]}")


# Module-level singleton
_client: Optional[ForgeGroqClient] = None


def get_client() -> ForgeGroqClient:
    global _client
    if _client is None:
        _client = ForgeGroqClient()
    return _client


def reset_client():
    """Force re-initialisation (useful if API key is set after import)."""
    global _client
    _client = None