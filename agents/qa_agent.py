"""
agents/qa_agent.py
Writes failing pytest test cases BEFORE production code is generated.
TDD-first is non-negotiable. Tests define the acceptance contract.
"""
from __future__ import annotations
import json, time
from core.llm import get_client
from core.observability import log_action, log_agent_metrics
from core.state import ForgeState, TaskItem


SYSTEM_PROMPT = """You are the QAAgent for Forge. Write pytest tests for the given task.

CRITICAL RULES:
1. Output ONLY valid Python code. No markdown. No explanation.
2. Always start with: import pytest
3. For FastAPI apps: use TestClient from starlette.testclient, import the app from its module.
4. For SQLAlchemy models: use an in-memory SQLite database in fixtures, never assume the DB exists.
5. Write a pytest fixture for the test client and database if needed.
6. Test HTTP status codes and response structure — not exact field values.
7. Use pytest.mark.parametrize for multiple similar cases.
8. Every test function name: test_<what>_<condition>_returns_<expected>.
9. Do NOT import things that don't exist yet — only import from files listed in the task's files list.
10. Write 3-5 focused test functions. Quality over quantity.
11. Use try/except ImportError and pytest.skip() for any optional dependency.

Example structure for a FastAPI route test:
    import pytest
    from starlette.testclient import TestClient

    @pytest.fixture
    def client():
        from app.main import app
        return TestClient(app)

    def test_create_item_valid_returns_201(client):
        response = client.post("/items", json={"name": "test", "value": 1.0})
        assert response.status_code in (200, 201)
        data = response.json()
        assert "id" in data or "name" in data

Example structure for a model test:
    import pytest
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    @pytest.fixture
    def db():
        from app.database import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()"""


def run(state: ForgeState) -> ForgeState:
    log_action("QAAgent", "Writing TDD test suite")

    task_plan = state.get("task_plan", [])
    architecture = state.get("architecture", {})
    spec = state.get("structured_spec", {})
    test_files = state.get("test_files", {})

    client = get_client()
    tasks_with_tests = 0

    for i, task in enumerate(task_plan):
        if task.get("test_file") is None:
            continue

        log_action("QAAgent", f"Writing tests for: {task['title']}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Write pytest tests for this task.\n\n"
                    f"Task title: {task['title']}\n"
                    f"Task description: {task['description'][:400]}\n"
                    f"Files this task will produce: {json.dumps(task.get('files', []))}\n"
                    f"Test file to write: {task['test_file']}\n\n"
                    f"API endpoints available:\n"
                    f"{json.dumps(architecture.get('api_endpoints', [])[:6], indent=2)}\n\n"
                    f"Data models:\n"
                    f"{json.dumps(architecture.get('data_models', [])[:3], indent=2)}\n\n"
                    "Write 3-5 focused test functions. "
                    "Only import from the files listed above. "
                    "Use fixtures for test client and DB session. "
                    "Test status codes and response shape, not exact values."
                ),
            },
        ]

        t0 = time.time()
        response, usage = client.call(messages, agent_name="QAAgent")
        latency = time.time() - t0

        # Clean up response
        test_code = response.strip()
        if test_code.startswith("```"):
            lines = test_code.split("\n")
            test_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        test_files[task["test_file"]] = test_code
        tasks_with_tests += 1

        state["api_call_count"] = state.get("api_call_count", 0) + 1
        state["total_tokens"] = state.get("total_tokens", 0) + usage["total_tokens"]

        log_action(
            "QAAgent",
            f"Tests written for task {task['id']}",
            f"file={task['test_file']} latency={latency:.1f}s",
        )

    metrics = {
        "tasks_with_tests": tasks_with_tests,
        "test_files_generated": len(test_files),
    }
    log_agent_metrics("QAAgent", metrics)
    log_action("QAAgent", f"TDD suite complete — {tasks_with_tests} test files written")

    state["test_files"] = test_files
    state["agent_metrics"]["QAAgent"] = metrics
    state["phase"] = "tests_written"
    return state