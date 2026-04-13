"""
core/observability.py
W&B + LangSmith initialisation and metric logging helpers.
"""
from __future__ import annotations
import os, time, json, datetime
from typing import Any, Optional
from pathlib import Path


# ── Activity Log ──────────────────────────────────────────────────────────────

_log_path: Optional[Path] = None
_log_buffer: list[dict] = []


def init_log(log_file: str = "./outputs/activity.log"):
    global _log_path
    _log_path = Path(log_file)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path.write_text("")   # reset


def log_action(agent: str, action: str, detail: str = "", level: str = "INFO"):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "agent": agent,
        "action": action,
        "detail": detail,
    }
    _log_buffer.append(entry)
    line = f"[{entry['timestamp']}] [{level}] [{agent}] {action}"
    if detail:
        line += f" — {detail}"
    print(line)
    if _log_path:
        with open(_log_path, "a") as f:
            f.write(line + "\n")
    return entry


# ── W&B ───────────────────────────────────────────────────────────────────────

_wandb_run = None
_wandb_enabled = False


def init_wandb(project: str, config: dict):
    global _wandb_run, _wandb_enabled
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        _wandb_run = wandb.init(
            project=project,
            config=config,
            reinit=True,
        )
        _wandb_enabled = True
        log_action("Observability", "W&B initialised", f"project={project}")
    except Exception as e:
        log_action("Observability", "W&B init failed (continuing without)", str(e), "WARN")
        _wandb_enabled = False


def log_metric(key: str, value: Any, step: int = None):
    if _wandb_enabled and _wandb_run:
        try:
            import wandb
            if step is not None:
                wandb.log({key: value}, step=step)
            else:
                wandb.log({key: value})
        except Exception:
            pass


def log_agent_metrics(agent_name: str, metrics: dict):
    """Log a batch of agent metrics to W&B."""
    if _wandb_enabled and _wandb_run:
        try:
            import wandb
            prefixed = {f"{agent_name}/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed)
        except Exception:
            pass


def log_decision(decision: dict):
    """Log a decision audit entry to W&B as a table row."""
    if _wandb_enabled and _wandb_run:
        try:
            import wandb
            wandb.log({
                "decision/agent": decision.get("agent"),
                "decision/confidence": decision.get("confidence"),
                "decision/reasoning_length": len(decision.get("reasoning", "")),
            })
        except Exception:
            pass


def finish_wandb(summary: dict):
    if _wandb_enabled and _wandb_run:
        try:
            import wandb
            for k, v in summary.items():
                wandb.run.summary[k] = v
            wandb.finish()
            log_action("Observability", "W&B run finished")
        except Exception:
            pass


# ── LangSmith ─────────────────────────────────────────────────────────────────

_langsmith_enabled = False


def init_langsmith(project: str):
    global _langsmith_enabled
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        log_action("Observability", "LangSmith skipped (no API key)", level="WARN")
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_API_KEY"] = api_key
    _langsmith_enabled = True
    log_action("Observability", "LangSmith initialised", f"project={project}")


def is_langsmith_enabled() -> bool:
    return _langsmith_enabled


# ── Summary Builder ───────────────────────────────────────────────────────────

def build_summary(state: dict) -> dict:
    tasks = state.get("task_plan", [])
    files = state.get("generated_files", {})
    test_results = state.get("test_results", {})
    security = state.get("security_findings", [])
    metrics = state.get("agent_metrics", {})

    completed = [t for t in tasks if t.get("status") == "done"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    skipped = [t for t in tasks if t.get("status") == "skipped"]

    tests_passed = sum(
        1 for v in test_results.values() if v.get("passed", False)
    )
    tests_total = len(test_results)

    summary = {
        "tasks_total": len(tasks),
        "tasks_completed": len(completed),
        "tasks_failed": len(failed),
        "tasks_skipped": len(skipped),
        "files_generated": len(files),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_pass_rate": round(tests_passed / max(tests_total, 1) * 100, 1),
        "security_findings_total": len(security),
        "security_findings_fixed": sum(1 for s in security if s.get("fix_applied")),
        "api_calls_total": state.get("api_call_count", 0),
        "total_tokens": state.get("total_tokens", 0),
        "complexity_tier": state.get("complexity_tier", 0),
        "agent_metrics": metrics,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return summary


def get_log_buffer() -> list[dict]:
    return _log_buffer
