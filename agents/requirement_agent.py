"""
agents/requirement_agent.py
Parses the natural-language prompt, infers project structure, generates
clarifying questions, and produces a structured JSON specification.
"""
from __future__ import annotations
import time
from core.llm import get_client, parse_json_response
from core.observability import log_action, log_agent_metrics, log_decision
from core.state import ForgeState


SYSTEM_PROMPT = """You are RequirementAgent. Analyse a project description and output JSON.

Output ONLY valid JSON — no explanation, no markdown fences:
{
  "project_summary": "Brief description",
  "complexity_tier": 1,
  "inferred_stack": {"framework": "FastAPI", "database": "SQLite", "auth": "none"},
  "clarifying_questions": [
    {"id": "q1", "question": "...", "options": ["A","B"], "default": "A"}
  ],
  "acceptance_criteria": ["criterion 1", "criterion 2"],
  "priority_weights": {"speed":0.2,"quality":0.25,"test_coverage":0.25,"security":0.2,"simplicity":0.1},
  "overall_confidence": 80,
  "confidence_reasoning": "one sentence",
  "alternatives_considered": ["alt 1 rejected because..."]
}

Tiers: 1=CRUD, 2=rules engine, 3=async API integration, 4=OAuth2+RBAC, 5=MongoDB joins+Change Streams.
Ask at most 3 clarifying questions, only for genuinely ambiguous aspects."""


def run(state: ForgeState) -> ForgeState:
    t0 = time.time()
    log_action("RequirementAgent", "Starting requirement analysis")

    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Project description:\n\n{state['raw_prompt']}"},
    ]

    response, usage = client.call_reasoning(messages, agent_name="RequirementAgent")

    try:
        spec = parse_json_response(response)
    except ValueError as e:
        log_action("RequirementAgent", "JSON parse failed, retrying", str(e), "WARN")
        # Retry with explicit reminder
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Your response was not valid JSON. Please respond ONLY with the JSON object, no explanation."})
        response, usage2 = client.call_reasoning(messages, agent_name="RequirementAgent")
        usage["total_tokens"] += usage2["total_tokens"]
        spec = parse_json_response(response)

    latency = time.time() - t0
    confidence = spec.get("overall_confidence", 70)

    # Record decision in audit trail
    decision = {
        "agent": "RequirementAgent",
        "confidence": confidence,
        "reasoning": spec.get("confidence_reasoning", ""),
        "alternatives": spec.get("alternatives_considered", []),
        "outcome": None,
    }

    metrics = {
        "latency_s": round(latency, 3),
        "tokens": usage["total_tokens"],
        "confidence": confidence,
        "questions_generated": len(spec.get("clarifying_questions", [])),
    }
    log_agent_metrics("RequirementAgent", metrics)
    log_decision(decision)

    log_action(
        "RequirementAgent",
        "Analysis complete",
        f"tier={spec.get('complexity_tier')} confidence={confidence}% "
        f"questions={len(spec.get('clarifying_questions', []))}",
    )

    state["structured_spec"] = spec
    state["complexity_tier"] = spec.get("complexity_tier", 1)
    state["inferred_stack"] = spec.get("inferred_stack", {})
    state["clarifying_questions"] = spec.get("clarifying_questions", [])
    state["priority_weights"] = spec.get("priority_weights", {})
    state["current_confidence"] = confidence
    state["decision_audit"] = state.get("decision_audit", []) + [decision]
    state["agent_metrics"]["RequirementAgent"] = metrics
    state["api_call_count"] = state.get("api_call_count", 0) + 1
    state["total_tokens"] = state.get("total_tokens", 0) + usage["total_tokens"]
    state["phase"] = "requirements_done"

    log_action(
        "RequirementAgent",
        "Structured specification produced",
        f"Awaiting user confirmation of {len(spec.get('clarifying_questions', []))} inferences",
    )

    return state
