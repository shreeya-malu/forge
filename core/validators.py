"""
core/validators.py
Objective validation tools: ruff, pylint, bandit, pytest.
Each returns a structured result dict for agent consumption.
"""
from __future__ import annotations
import subprocess, json, tempfile, os, sys
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], cwd: str = None, timeout: int = 60) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "TIMEOUT"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


def write_temp_file(code: str, suffix: str = ".py") -> str:
    """Write code to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    tmp.write(code)
    tmp.close()
    return tmp.name


def run_ruff(code: str, filepath: str = "code.py") -> dict:
    """Run ruff linter on a code string. Returns structured result."""
    tmp = write_temp_file(code)
    try:
        rc, stdout, stderr = _run(
            ["python", "-m", "ruff", "check", "--output-format=json", tmp]
        )
        issues = []
        try:
            raw = json.loads(stdout) if stdout.strip() else []
            for item in raw:
                issues.append({
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "line": item.get("location", {}).get("row", 0),
                    "fixable": item.get("fix") is not None,
                })
        except (json.JSONDecodeError, Exception):
            if stdout.strip():
                issues.append({"code": "PARSE_ERROR", "message": stdout[:500], "line": 0})

        return {
            "passed": rc == 0,
            "issue_count": len(issues),
            "issues": issues,
            "raw": stdout[:2000],
            "tool": "ruff",
        }
    finally:
        os.unlink(tmp)


def run_pylint(code: str, filepath: str = "code.py") -> dict:
    """Run pylint on a code string. Returns structured result."""
    tmp = write_temp_file(code)
    try:
        rc, stdout, stderr = _run(
            ["python", "-m", "pylint", tmp,
             "--output-format=json",
             "--disable=C0114,C0115,C0116,C0301,R0903",
             "--score=no"]
        )
        issues = []
        try:
            raw_list = json.loads(stdout) if stdout.strip().startswith("[") else []
            for item in raw_list:
                if item.get("type") in ("error", "warning", "convention"):
                    issues.append({
                        "code": item.get("message-id", ""),
                        "message": item.get("message", ""),
                        "line": item.get("line", 0),
                        "type": item.get("type", ""),
                    })
        except (json.JSONDecodeError, Exception):
            pass

        errors = [i for i in issues if i.get("type") == "error"]
        return {
            "passed": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len([i for i in issues if i.get("type") == "warning"]),
            "issues": issues,
            "raw": stdout[:2000],
            "tool": "pylint",
        }
    finally:
        os.unlink(tmp)


def run_bandit(code: str, filepath: str = "code.py") -> dict:
    """Run bandit security scanner. Returns structured result."""
    tmp = write_temp_file(code)
    try:
        rc, stdout, stderr = _run(
            ["python", "-m", "bandit", "-f", "json", "-q", tmp]
        )
        findings = []
        try:
            data = json.loads(stdout) if stdout.strip() else {}
            for result in data.get("results", []):
                findings.append({
                    "issue": result.get("issue_text", ""),
                    "severity": result.get("issue_severity", "LOW"),
                    "confidence": result.get("issue_confidence", "LOW"),
                    "line": result.get("line_number", 0),
                    "code": result.get("test_id", ""),
                    "cwe": result.get("issue_cwe", {}).get("id", ""),
                })
        except (json.JSONDecodeError, Exception):
            pass

        high_sev = [f for f in findings if f.get("severity") == "HIGH"]
        return {
            "passed": len(high_sev) == 0,
            "finding_count": len(findings),
            "high_severity": len(high_sev),
            "medium_severity": len([f for f in findings if f.get("severity") == "MEDIUM"]),
            "findings": findings,
            "raw": stdout[:2000],
            "tool": "bandit",
        }
    finally:
        os.unlink(tmp)


def run_syntax_check(code: str) -> dict:
    """Check Python syntax via compile()."""
    try:
        compile(code, "<forge_generated>", "exec")
        return {"passed": True, "error": None}
    except SyntaxError as e:
        return {"passed": False, "error": str(e), "line": e.lineno}


def run_pytest(
    test_code: str,
    impl_code: str,
    impl_filename: str = "implementation.py",
    all_generated_files: dict = None,
    project_dir_path: str = None,
) -> dict:
    """
    Run pytest against the generated implementation.

    Mirrors the full generated_project directory into a temp dir, installs
    the project's requirements.txt so cross-module imports (FastAPI, SQLAlchemy
    etc.) resolve, then runs pytest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if all_generated_files and len(all_generated_files) > 1:
            # Mirror every generated file
            for rel_path, content in all_generated_files.items():
                dest = tmp / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content or "")

            # Ensure every directory is a package
            for d in sorted(tmp.rglob("*")):
                if d.is_dir() and not (d / "__init__.py").exists():
                    (d / "__init__.py").write_text("")

            # Install requirements if present so imports resolve
            req_file = tmp / "requirements.txt"
            if req_file.exists():
                _run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "-r", str(req_file), "--quiet"],
                    timeout=120,
                )

            # Write test file
            test_dest = tmp / "test_run.py"
            test_dest.write_text(test_code)
            test_target = str(test_dest)
        else:
            # Single-file fallback
            (tmp / impl_filename).write_text(impl_code)
            test_dest = tmp / "test_implementation.py"
            test_dest.write_text(test_code)
            (tmp / "conftest.py").write_text("")
            test_target = str(test_dest)

        report_file = f"/tmp/forge_pytest_{os.getpid()}.json"
        rc, stdout, stderr = _run(
            [sys.executable, "-m", "pytest", test_target,
             "-v", "--tb=short",
             "--json-report", f"--json-report-file={report_file}",
             "--no-header", "-q",
             "--ignore=generated_project"],   # avoid pytest scanning project_dir
            cwd=tmpdir,
            timeout=90,
        )

        tests_passed = 0
        tests_failed = 0
        test_details = []
        try:
            with open(report_file) as f:
                report = json.load(f)
            summary = report.get("summary", {})
            tests_passed = summary.get("passed", 0)
            tests_failed = summary.get("failed", 0) + summary.get("error", 0)
            for test in report.get("tests", []):
                entry = {
                    "name":     test.get("nodeid", ""),
                    "outcome":  test.get("outcome", ""),
                    "duration": test.get("duration", 0),
                    "message":  "",
                }
                if test.get("outcome") == "failed":
                    entry["message"] = str(
                        test.get("call", {}).get("longrepr", "")
                    )[:400]
                test_details.append(entry)
        except Exception:
            import re
            m = re.search(r"(\d+) passed", stdout)
            if m:
                tests_passed = int(m.group(1))
            m = re.search(r"(\d+) failed", stdout)
            if m:
                tests_failed = int(m.group(1))
        finally:
            try:
                os.unlink(report_file)
            except Exception:
                pass

        return {
            "passed":       tests_failed == 0 and tests_passed > 0,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "test_details": test_details,
            "stdout":       stdout[:2000],
            "stderr":       stderr[:800],
            "return_code":  rc,
            "tool":         "pytest",
        }


def validate_file(code: str, filepath: str = "code.py") -> dict:
    """
    Run validation suite on a single file.
    Blocking failures: syntax errors, ruff E-codes (actual errors).
    Non-blocking: pylint warnings/conventions, ruff W-codes (warnings), unused imports.
    """
    syntax = run_syntax_check(code)
    if not syntax["passed"]:
        return {
            "overall_passed": False,
            "syntax": syntax,
            "ruff": None,
            "pylint": None,
            "blocking_error": f"SyntaxError at line {syntax.get('line')}: {syntax.get('error')}",
        }

    ruff = run_ruff(code, filepath)
    pylint = run_pylint(code, filepath)

    # Only block on ruff E-codes that indicate real errors (not style)
    # E1, E2, E3, E4, E5, E7, E9 are real errors; E501 (line length) is not blocking
    BLOCKING_RUFF_PREFIXES = ("E1", "E2", "E3", "E4", "E5", "E7", "E9", "F8", "F9")
    NON_BLOCKING_RUFF = {"E501", "W291", "W293", "W292", "W391"}

    ruff_blocking_issues = [
        i for i in ruff.get("issues", [])
        if (
            any(i.get("code", "").startswith(p) for p in BLOCKING_RUFF_PREFIXES)
            and i.get("code") not in NON_BLOCKING_RUFF
        )
    ]

    # Only block on pylint actual errors (E-codes), not warnings or conventions
    pylint_errors = [
        i for i in pylint.get("issues", [])
        if i.get("type") == "error"
        and i.get("code", "").startswith("E")
        and i.get("code") not in (
            "E0401",  # import-error (missing dep in test env)
            "E0611",  # no-name-in-module
            "E0213",  # no-self-argument — false positive on Pydantic @validator
        )
    ]

    blocking = len(ruff_blocking_issues) > 0 or len(pylint_errors) > 0
    blocking_error = None
    if blocking:
        parts = []
        if ruff_blocking_issues:
            parts.append("Ruff: " + "; ".join(
                f"L{i['line']} {i['code']} {i['message']}" for i in ruff_blocking_issues[:3]
            ))
        if pylint_errors:
            parts.append("Pylint: " + "; ".join(
                f"L{i['line']} {i['message']}" for i in pylint_errors[:3]
            ))
        blocking_error = " | ".join(parts)

    return {
        "overall_passed": not blocking,
        "syntax": syntax,
        "ruff": ruff,
        "pylint": pylint,
        "blocking_error": blocking_error,
        "ruff_blocking": ruff_blocking_issues,
        "pylint_errors": pylint_errors,
    }