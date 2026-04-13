"""
core/graph.py
LangGraph pipeline graph — wires all 9 agents into a stateful workflow
with confidence-triggered checkpoints and conditional routing.
"""
from __future__ import annotations
import yaml
from pathlib import Path
from langgraph.graph import StateGraph, END
from core.state import ForgeState
from core.observability import log_action, log_metric


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
PIPELINE_CFG = CFG["pipeline"]

CONFIDENCE_AUTO = PIPELINE_CFG["confidence_autonomous"]   # 80
CONFIDENCE_FLAG = PIPELINE_CFG["confidence_flag"]         # 60


# ── Node wrappers ──────────────────────────────────────────────────────────────

def node_requirements(state: ForgeState) -> ForgeState:
    from agents.requirement_agent import run
    return run(state)


def node_planning(state: ForgeState) -> ForgeState:
    from agents.planning_agent import run
    return run(state)


def node_arch_validation(state: ForgeState) -> ForgeState:
    from agents.architecture_validator import run
    return run(state)


def node_qa(state: ForgeState) -> ForgeState:
    from agents.qa_agent import run
    return run(state)


def node_codegen(state: ForgeState) -> ForgeState:
    from agents.codegen_agent import run
    return run(state)


def node_security_audit(state: ForgeState) -> ForgeState:
    from agents.security_audit_agent import run_audit
    return run_audit(state)


def node_security_fix(state: ForgeState) -> ForgeState:
    from agents.security_audit_agent import run_fix
    return run_fix(state)


def node_docker(state: ForgeState) -> ForgeState:
    from agents.docker_agent import run
    return run(state)


def node_summary(state: ForgeState) -> ForgeState:
    """Build final workflow summary and finish observability."""
    from core.observability import build_summary, finish_wandb, log_action
    import json
    from pathlib import Path

    summary = build_summary(state)
    state["workflow_summary"] = summary

    # Write output artifacts
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if state.get("security_report"):
        (output_dir / "security_report.json").write_text(
            json.dumps(state["security_report"], indent=2)
        )

    if state.get("structured_spec"):
        (output_dir / "specification.json").write_text(
            json.dumps(state["structured_spec"], indent=2)
        )

    # Write generated project files
    project_dir = Path("./generated_project")
    project_dir.mkdir(exist_ok=True)
    for filepath, content in state.get("generated_files", {}).items():
        full_path = project_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    finish_wandb(summary)
    log_action("Pipeline", "Workflow complete", str(summary))

    state["phase"] = "complete"
    state["pipeline_status"] = "completed"
    return state


# ── Conditional routing ────────────────────────────────────────────────────────

def route_after_requirements(state: ForgeState) -> str:
    """Route after RequirementAgent based on confidence."""
    confidence = state.get("current_confidence", 80)
    if confidence < CONFIDENCE_FLAG:
        log_action("Router", f"LOW CONFIDENCE ({confidence}%) — pausing for human input")
        state["awaiting_human_input"] = True
        state["human_input_context"] = {
            "reason": "RequirementAgent confidence below threshold",
            "confidence": confidence,
            "questions": state.get("clarifying_questions", []),
        }
    # Always proceed to planning (human input is collected in dashboard)
    return "planning"


def route_after_validation(state: ForgeState) -> str:
    """Route after ArchitectureValidatorAgent based on verdict."""
    validation = state.get("architecture_validation", {})
    verdict = validation.get("verdict", "approved")

    if verdict == "requires_revision":
        log_action("Router", "Architecture requires revision — routing back to planning")
        return "planning"  # Re-run planning with validation feedback
    return "qa"


def route_after_security(state: ForgeState) -> str:
    """Route after security audit."""
    findings = state.get("security_findings", [])
    if findings:
        return "security_fix"
    return "docker"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the Forge LangGraph pipeline."""

    graph = StateGraph(ForgeState)

    # Register nodes
    graph.add_node("requirements",    node_requirements)
    graph.add_node("planning",        node_planning)
    graph.add_node("arch_validation", node_arch_validation)
    graph.add_node("qa",              node_qa)
    graph.add_node("codegen",         node_codegen)
    graph.add_node("security_audit",  node_security_audit)
    graph.add_node("security_fix",    node_security_fix)
    graph.add_node("docker",          node_docker)
    graph.add_node("summary",         node_summary)

    # Entry point
    graph.set_entry_point("requirements")

    # Edges
    graph.add_conditional_edges(
        "requirements",
        route_after_requirements,
        {"planning": "planning"},
    )
    graph.add_edge("planning", "arch_validation")
    graph.add_conditional_edges(
        "arch_validation",
        route_after_validation,
        {"qa": "qa", "planning": "planning"},
    )
    graph.add_edge("qa", "codegen")
    graph.add_edge("codegen", "security_audit")
    graph.add_conditional_edges(
        "security_audit",
        route_after_security,
        {"security_fix": "security_fix", "docker": "docker"},
    )
    graph.add_edge("security_fix", "docker")
    graph.add_edge("docker", "summary")
    graph.add_edge("summary", END)

    return graph.compile()


def get_initial_state(prompt: str, priority_weights: dict = None) -> ForgeState:
    """Return a fully initialised ForgeState for a new run."""
    return ForgeState(
        raw_prompt=prompt,
        priority_weights=priority_weights or {
            "speed": 0.2, "quality": 0.25,
            "test_coverage": 0.25, "security": 0.2, "simplicity": 0.1,
        },
        clarifying_questions=[],
        clarifying_answers={},
        structured_spec={},
        complexity_tier=1,
        inferred_stack={},
        chosen_pattern="",
        architecture={},
        architecture_validation={},
        task_plan=[],
        user_approved_plan=False,
        current_task_index=0,
        generated_files={},
        test_files={},
        file_diffs=[],
        ruff_results={},
        pylint_results={},
        test_results={},
        debug_attempts={},
        security_findings=[],
        security_report={},
        dockerfile="",
        docker_compose="",
        docker_validated=False,
        decision_audit=[],
        current_confidence=100,
        awaiting_human_input=False,
        human_input_context={},
        wandb_run_id=None,
        api_call_count=0,
        total_tokens=0,
        agent_metrics={
            "RequirementAgent": {},
            "PlanningAgent": {},
            "ArchitectureValidatorAgent": {},
            "QAAgent": {},
            "CodeGenAgent": {},
            "DebugAgent": {},
            "SecurityAuditAgent": {},
            "SecurityFixAgent": {},
            "DockerAgent": {},
        },
        activity_log=[],
        workflow_summary={},
        errors=[],
        phase="init",
        pipeline_status="running",
    )
