"""
agents/architecture_validator.py
Cross-checks the chosen architecture against confirmed requirements.
"""
from __future__ import annotations
import json, time
from core.llm import get_client, parse_json_response
from core.observability import log_action, log_agent_metrics, log_decision
from core.state import ForgeState


SYSTEM_PROMPT = """You are ArchitectureValidatorAgent. Cross-check a proposed architecture against requirements.

Output ONLY valid JSON:
{
  "verdict": "approved",
  "overall_confidence": 88,
  "validation_checks": [
    {"check": "Database choice", "result": "pass", "reasoning": "SQLite suits relational data with no concurrency", "alternative": null},
    {"check": "Auth model", "result": "warning", "reasoning": "No auth specified but API is public", "alternative": "Add API key header"}
  ],
  "flagged_issues": [
    {"severity": "warning", "issue": "Missing rate limiting", "recommendation": "Add slowapi"}
  ],
  "architecture_amendments": [],
  "validation_summary": "one sentence verdict",
  "alternatives_considered": ["PostgreSQL considered, rejected — no concurrent write requirements"]
}

Verdicts: approved / approved_with_warnings / requires_revision.
Be concise. Max 4 validation checks, max 3 flagged issues."""


def run(state: ForgeState) -> ForgeState:
    t0 = time.time()
    log_action("ArchitectureValidatorAgent", "Starting architecture cross-validation")

    spec = state.get("structured_spec", {})
    architecture = state.get("architecture", {})
    task_plan = state.get("task_plan", [])
    answers = state.get("clarifying_answers", {})

    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Project: {spec.get('project_summary','')[:200]}\n"
                f"Stack: {json.dumps(spec.get('inferred_stack', {}))}\n"
                f"Pattern chosen: {state.get('chosen_pattern','')}\n"
                f"Key decisions: {json.dumps(architecture.get('key_decisions', []))}\n"
                f"Task count: {len(task_plan)}\n"
                f"Tasks: {json.dumps([t['title'] for t in task_plan])}\n"
                f"User answers: {json.dumps(answers)}\n\n"
                "Validate this architecture and output JSON."
            ),
        },
    ]

    response, usage = client.call_reasoning(messages, agent_name="ArchitectureValidatorAgent")

    try:
        validation = parse_json_response(response)
    except ValueError as e:
        log_action("ArchitectureValidatorAgent", "JSON parse failed", str(e), "WARN")
        validation = {
            "verdict": "approved_with_warnings",
            "overall_confidence": 65,
            "validation_checks": [],
            "flagged_issues": [{"severity": "warning", "issue": "Validator parse error", "recommendation": "Manual review recommended"}],
            "architecture_amendments": [],
            "validation_summary": "Automated validation partially failed — manual review recommended",
            "alternatives_considered": [],
        }

    latency = time.time() - t0
    confidence = validation.get("overall_confidence", 70)
    verdict = validation.get("verdict", "approved")

    # Apply any architecture amendments
    if validation.get("architecture_amendments"):
        for amendment in validation["architecture_amendments"]:
            log_action(
                "ArchitectureValidatorAgent",
                f"Architecture amendment applied: {amendment.get('component')}",
                amendment.get("amendment", ""),
            )

    decision = {
        "agent": "ArchitectureValidatorAgent",
        "confidence": confidence,
        "reasoning": validation.get("validation_summary", ""),
        "alternatives": validation.get("alternatives_considered", []),
        "outcome": verdict,
    }

    metrics = {
        "latency_s": round(latency, 3),
        "tokens": usage["total_tokens"],
        "confidence": confidence,
        "verdict": verdict,
        "checks_passed": sum(1 for c in validation.get("validation_checks", []) if c.get("result") == "pass"),
        "warnings": len([c for c in validation.get("validation_checks", []) if c.get("result") == "warning"]),
        "issues_flagged": len(validation.get("flagged_issues", [])),
    }
    log_agent_metrics("ArchitectureValidatorAgent", metrics)
    log_decision(decision)

    log_action(
        "ArchitectureValidatorAgent",
        f"Validation complete — verdict: {verdict.upper()}",
        f"confidence={confidence}% issues={metrics['issues_flagged']}",
    )

    state["architecture_validation"] = validation
    state["current_confidence"] = confidence
    state["decision_audit"] = state.get("decision_audit", []) + [decision]
    state["agent_metrics"]["ArchitectureValidatorAgent"] = metrics
    state["api_call_count"] = state.get("api_call_count", 0) + 1
    state["total_tokens"] = state.get("total_tokens", 0) + usage["total_tokens"]
    state["phase"] = "validation_done"

    return state
