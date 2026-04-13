"""
core/state.py
Forge pipeline state — typed TypedDict passed through every LangGraph node.
"""
from __future__ import annotations
from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class AgentDecision(TypedDict):
    agent: str
    confidence: int          # 0-100
    reasoning: str
    alternatives: list[str]
    outcome: Optional[str]   # filled in post-validation


class TaskItem(TypedDict):
    id: str
    title: str
    description: str
    risk_level: Literal["low", "medium", "high"]
    checkpoint_flag: bool
    status: Literal["pending", "in_progress", "done", "failed", "skipped"]
    files: list[str]
    test_file: Optional[str]


class SecurityFinding(TypedDict):
    file: str
    severity: str
    issue: str
    line: int
    fix_applied: bool
    fix_description: str


class ForgeState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    raw_prompt: str
    priority_weights: dict[str, float]   # speed, quality, test_coverage, security, simplicity

    # ── Specification ──────────────────────────────────────────────────────
    clarifying_questions: list[dict]
    clarifying_answers: dict[str, str]
    structured_spec: dict[str, Any]      # JSON spec artifact
    complexity_tier: int                 # 1-5
    inferred_stack: dict[str, str]       # {database, auth, framework, ...}

    # ── Architecture ───────────────────────────────────────────────────────
    chosen_pattern: str
    architecture: dict[str, Any]
    architecture_validation: dict[str, Any]
    task_plan: list[TaskItem]
    user_approved_plan: bool

    # ── Code Generation ────────────────────────────────────────────────────
    current_task_index: int
    generated_files: dict[str, str]      # filepath → content
    test_files: dict[str, str]
    file_diffs: list[dict]

    # ── Validation ─────────────────────────────────────────────────────────
    ruff_results: dict[str, Any]
    pylint_results: dict[str, Any]
    test_results: dict[str, Any]
    debug_attempts: dict[str, int]       # filepath → attempt count

    # ── Security ───────────────────────────────────────────────────────────
    security_findings: list[SecurityFinding]
    security_report: dict[str, Any]

    # ── Docker ─────────────────────────────────────────────────────────────
    dockerfile: str
    docker_compose: str
    docker_validated: bool

    # ── Decision Intelligence ──────────────────────────────────────────────
    decision_audit: list[AgentDecision]
    current_confidence: int
    awaiting_human_input: bool
    human_input_context: dict[str, Any]

    # ── Observability ──────────────────────────────────────────────────────
    wandb_run_id: Optional[str]
    api_call_count: int
    total_tokens: int
    agent_metrics: dict[str, dict]       # agent_name → {latency, tokens, confidence}

    # ── Artifacts ──────────────────────────────────────────────────────────
    activity_log: list[dict]
    workflow_summary: dict[str, Any]
    errors: list[dict]

    # ── Pipeline Control ───────────────────────────────────────────────────
    phase: str
    pipeline_status: Literal["running", "waiting", "completed", "failed"]
