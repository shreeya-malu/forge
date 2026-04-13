"""
agents/debug_agent.py
Classifies failure type and applies targeted fixes.
Called inline by codegen_agent on validation failures.
"""
from __future__ import annotations
import time
from core.llm import get_client
from core.observability import log_action


SYSTEM_PROMPT = """Fix the broken Python code. Output ONLY the corrected file. No markdown. No explanation.

Fix ONLY the reported error. Do not change anything else. Keep the fix minimal."""


def run_targeted_fix(
    code: str,
    error_info: str,
    filepath: str,
    agent_name: str = "DebugAgent",
) -> tuple[str, dict]:
    """Apply a targeted fix. Returns (fixed_code, usage_dict)."""
    t0 = time.time()
    client = get_client()
    error_type = _classify_error(error_info)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"File: {filepath}\n"
                f"Error type: {error_type}\n"
                f"Error:\n{error_info[:600]}\n\n"
                f"Code:\n{code}\n\n"
                "Output the complete corrected file."
            ),
        },
    ]

    response, usage = client.call(messages, agent_name=agent_name)
    fixed = _clean_code(response)
    usage["latency_s"] = round(time.time() - t0, 3)
    usage["error_type"] = error_type

    log_action("DebugAgent", f"Fix applied to {filepath}", f"error_type={error_type}")
    return fixed, usage


def _classify_error(error_info: str) -> str:
    s = error_info.lower()
    if "syntaxerror" in s or "indentationerror" in s:
        return "syntax"
    if "importerror" in s or "modulenotfounderror" in s or "no module" in s:
        return "import"
    if "typeerror" in s or "attributeerror" in s:
        return "type"
    if "assertionerror" in s or "failed" in s:
        return "logic"
    return "lint"


def _clean_code(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        end = len(lines) - 1
        while end > 0 and lines[end].strip().startswith("```"):
            end -= 1
        raw = "\n".join(lines[1:end + 1])
    return raw
