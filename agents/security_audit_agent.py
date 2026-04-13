"""
agents/security_audit_agent.py + security_fix_agent.py
Runs bandit after each module, passes findings to SecurityFixAgent
for immediate remediation. Findings are fixed then and there.
"""
from __future__ import annotations
import json, time
from core.llm import get_client
from core.validators import run_bandit
from core.observability import log_action, log_agent_metrics, log_metric
from core.state import ForgeState, SecurityFinding


# ── Security Audit Agent ───────────────────────────────────────────────────────

def run_audit(state: ForgeState) -> ForgeState:
    """Run bandit on all generated Python files and collect findings."""
    log_action("SecurityAuditAgent", "Starting security audit")

    generated_files   = state.get("generated_files", {})
    security_findings = state.get("security_findings", [])

    # Bandit codes that are noise / false positives we never want to fix:
    #   B101 — assert used. Correct in test files; never fix.
    #   B104 — hardcoded bind all interfaces — fine for a dev server.
    SKIP_CODES = {"B101", "B104"}

    audit_results = {}
    total_findings = 0

    for filepath, code in generated_files.items():
        if not filepath.endswith(".py"):
            continue
        if "PLACEHOLDER" in (code or "")[:100]:
            continue
        # Never audit test files — assert, hardcoded values etc. are intentional
        if "/test" in filepath or filepath.startswith("test"):
            continue

        bandit_result = run_bandit(code, filepath)
        audit_results[filepath] = bandit_result

        actionable = [
            f for f in bandit_result.get("findings", [])
            if f.get("code", "") not in SKIP_CODES
        ]
        total_findings += len(actionable)

        if actionable:
            for finding in actionable:
                security_findings.append({
                    "file":            filepath,
                    "severity":        finding.get("severity", "LOW"),
                    "issue":           finding.get("issue", ""),
                    "line":            finding.get("line", 0),
                    "fix_applied":     False,
                    "fix_description": "",
                })
            high = sum(1 for f in actionable if f.get("severity") == "HIGH")
            log_action(
                "SecurityAuditAgent",
                f"Findings in {filepath}",
                f"{len(actionable)} actionable issues ({high} high)",
            )

    log_metric("security/total_findings", total_findings)
    log_metric("security/files_audited", len(audit_results))

    metrics = {
        "files_audited":  len(audit_results),
        "total_findings": total_findings,
        "high_severity":  sum(r.get("high_severity", 0) for r in audit_results.values()),
    }
    log_agent_metrics("SecurityAuditAgent", metrics)

    state["security_findings"]              = security_findings
    state["agent_metrics"]["SecurityAuditAgent"] = metrics
    state["api_call_count"]                 = state.get("api_call_count", 0)
    state["phase"]                          = "security_audit_done"

    log_action("SecurityAuditAgent", "Security audit complete",
               f"total_actionable_findings={total_findings}")
    return state


# ── Security Fix Agent ─────────────────────────────────────────────────────────

FIX_SYSTEM_PROMPT = """You are the SecurityFixAgent for Forge.

You receive Python code with a specific security vulnerability identified by bandit.
Your job: apply a minimal, targeted fix to resolve the vulnerability.

Common fixes:
- SQL injection: use parameterised queries instead of string formatting
- Hardcoded credentials: replace with os.environ.get() calls
- eval/exec usage: replace with ast.literal_eval or safe alternatives
- Insecure random: replace random with secrets module
- HTTP without TLS: flag but do not change (infrastructure concern)
- Broad exception: catch specific exceptions

Rules:
1. Output ONLY the fixed Python code — no explanation, no markdown fences
2. Fix ONLY the security issue — do not change unrelated code
3. The fix must not break existing functionality
4. Add a comment explaining the security fix"""


def run_fix(state: ForgeState) -> ForgeState:
    """Apply fixes for all security findings."""
    log_action("SecurityFixAgent", "Applying security fixes")

    security_findings = state.get("security_findings", [])
    generated_files = state.get("generated_files", {})
    client = get_client()

    fixes_applied = 0
    fixes_failed = 0

    # Group findings by file
    findings_by_file: dict[str, list] = {}
    for finding in security_findings:
        f = finding["file"]
        findings_by_file.setdefault(f, []).append(finding)

    for filepath, findings in findings_by_file.items():
        if filepath not in generated_files:
            continue

        code = generated_files[filepath]
        all_fixed = True

        for finding in findings:
            if finding["fix_applied"]:
                continue

            success, fixed_code, description = _apply_fix(
                code=code,
                filepath=filepath,
                finding=finding,
                client=client,
                state=state,
            )

            if success:
                code = fixed_code
                finding["fix_applied"] = True
                finding["fix_description"] = description
                fixes_applied += 1
                log_action("SecurityFixAgent", f"Fixed: {finding['issue'][:60]}", filepath)
            else:
                fixes_failed += 1
                log_action("SecurityFixAgent", f"Could not fix: {finding['issue'][:60]}", filepath, "WARN")

        generated_files[filepath] = code

    # Build security report
    security_report = {
        "total_findings": len(security_findings),
        "fixes_applied": fixes_applied,
        "fixes_failed": fixes_failed,
        "findings_by_severity": {
            "high": len([f for f in security_findings if f["severity"] == "HIGH"]),
            "medium": len([f for f in security_findings if f["severity"] == "MEDIUM"]),
            "low": len([f for f in security_findings if f["severity"] == "LOW"]),
        },
        "findings": security_findings,
    }

    log_metric("security/fixes_applied", fixes_applied)

    metrics = {
        "fixes_applied": fixes_applied,
        "fixes_failed": fixes_failed,
        "fix_rate": round(fixes_applied / max(len(security_findings), 1) * 100, 1),
    }
    log_agent_metrics("SecurityFixAgent", metrics)

    state["generated_files"] = generated_files
    state["security_findings"] = security_findings
    state["security_report"] = security_report
    state["agent_metrics"]["SecurityFixAgent"] = metrics
    state["phase"] = "security_fixed"

    log_action(
        "SecurityFixAgent",
        "Security remediation complete",
        f"fixed={fixes_applied} failed={fixes_failed}",
    )
    return state


def _apply_fix(code, filepath, finding, client, state, max_retries=3):
    """Apply a single security fix with retries."""
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"File: {filepath}\n"
                    f"Security issue: {finding['issue']}\n"
                    f"Severity: {finding['severity']}\n"
                    f"Line: {finding['line']}\n\n"
                    f"Code to fix:\n{code}\n\n"
                    f"Output ONLY the fixed Python code."
                ),
            },
        ]

        try:
            response, usage = client.call_fast(messages, agent_name="SecurityFixAgent")
            fixed = _clean_code(response)
            state["api_call_count"] = state.get("api_call_count", 0) + 1
            state["total_tokens"] = state.get("total_tokens", 0) + usage["total_tokens"]

            # Verify fix didn't break syntax
            from core.validators import run_syntax_check
            syntax = run_syntax_check(fixed)
            if syntax["passed"]:
                return True, fixed, f"Fixed {finding['issue'][:100]} (attempt {attempt + 1})"
        except Exception as e:
            log_action("SecurityFixAgent", f"Fix attempt {attempt + 1} failed", str(e), "WARN")

    return False, code, "Fix failed after max retries"


def _clean_code(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        end = len(lines) - 1
        while end > 0 and lines[end].strip() in ("```", ""):
            end -= 1
        raw = "\n".join(lines[1:end + 1])
    return raw