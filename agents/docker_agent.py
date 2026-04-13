"""
agents/docker_agent.py

Template-based Docker file generation — no LLM calls.
The Dockerfile and docker-compose.yml are deterministic from the project stack.
This removes the hang risk entirely and produces more reliable output
than an LLM that might stall or return incomplete YAML.
"""
from __future__ import annotations
import time
from pathlib import Path
from core.observability import log_action, log_agent_metrics
from core.state import ForgeState


# ── Dockerfile template ────────────────────────────────────────────────────────

DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Layer caching: install deps before copying code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

CMD ["uvicorn", "{entrypoint}:app", "--host", "0.0.0.0", "--port", "{port}"]
"""

# ── docker-compose template ────────────────────────────────────────────────────

COMPOSE_BASE = """\
version: '3.8'

services:
  app:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - DATABASE_URL={db_url}
      - SECRET_KEY=changeme-in-production
      - DEBUG=false
    depends_on:{depends}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
{extra_services}
volumes:{volumes}
"""

POSTGRES_SERVICE = """
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=forge
      - POSTGRES_PASSWORD=forgepass
      - POSTGRES_DB=forgedb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
"""

MONGO_SERVICE = """
  mongo:
    image: mongo:7
    environment:
      - MONGO_INITDB_ROOT_USERNAME=forge
      - MONGO_INITDB_ROOT_PASSWORD=forgepass
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped
"""

REDIS_SERVICE = """
  redis:
    image: redis:7-alpine
    restart: unless-stopped
"""


def run(state: ForgeState) -> ForgeState:
    t0 = time.time()
    log_action("DockerAgent", "Generating Docker configuration (template-based)")

    inferred_stack  = state.get("inferred_stack", {})
    generated_files = state.get("generated_files", {})
    architecture    = state.get("architecture", {})

    # ── Detect stack ──────────────────────────────────────────────────────────
    stack_str = str(inferred_stack).lower()
    decisions = " ".join(
        d.get("decision", "") + " " + d.get("rationale", "")
        for d in architecture.get("key_decisions", [])
    ).lower()
    combined = stack_str + " " + decisions

    use_postgres = any(x in combined for x in ("postgres", "postgresql", "asyncpg"))
    use_mongo    = any(x in combined for x in ("mongo", "mongodb", "motor"))
    use_redis    = "redis" in combined

    # ── Detect entry point ────────────────────────────────────────────────────
    entrypoint = "app.main"
    for f in sorted(generated_files.keys()):
        if f.endswith("main.py"):
            entrypoint = f.replace("/", ".").replace(".py", "")
            break

    port = 8000

    # ── Dockerfile ────────────────────────────────────────────────────────────
    dockerfile = DOCKERFILE_TEMPLATE.format(port=port, entrypoint=entrypoint)

    # ── docker-compose ────────────────────────────────────────────────────────
    extra_services = ""
    volumes_list   = []
    depends_list   = []

    if use_postgres:
        extra_services += POSTGRES_SERVICE
        volumes_list.append("  postgres_data:")
        depends_list.append("      - db")
        db_url = "postgresql://forge:forgepass@db:5432/forgedb"
    elif use_mongo:
        extra_services += MONGO_SERVICE
        volumes_list.append("  mongo_data:")
        depends_list.append("      - mongo")
        db_url = "mongodb://forge:forgepass@mongo:27017/forgedb"
    else:
        db_url = "sqlite:///./app.db"

    if use_redis:
        extra_services += REDIS_SERVICE
        depends_list.append("      - redis")

    depends_str  = ("\n" + "\n".join(depends_list)) if depends_list else ""
    volumes_str  = ("\n" + "\n".join(volumes_list)) if volumes_list else " {}"

    docker_compose = COMPOSE_BASE.format(
        port=port,
        db_url=db_url,
        depends=depends_str,
        extra_services=extra_services,
        volumes=volumes_str,
    )

    # ── Write to disk ─────────────────────────────────────────────────────────
    project_dir = Path("./generated_project")
    project_dir.mkdir(exist_ok=True)

    for filename, content in [("Dockerfile", dockerfile), ("docker-compose.yml", docker_compose)]:
        generated_files[filename] = content
        (project_dir / filename).write_text(content)
        log_action("DockerAgent", f"  ✓ {filename}")

    stack_detected = "postgres" if use_postgres else "mongo" if use_mongo else "sqlite"
    metrics = {"latency_s": round(time.time() - t0, 3), "tokens": 0, "stack": stack_detected}
    log_agent_metrics("DockerAgent", metrics)

    state["dockerfile"]                   = dockerfile
    state["docker_compose"]               = docker_compose
    state["docker_validated"]             = True
    state["generated_files"]              = generated_files
    state["agent_metrics"]["DockerAgent"] = metrics
    state["phase"]                        = "docker_done"

    log_action("DockerAgent", "Docker configuration complete",
               f"stack={stack_detected} entry={entrypoint} — no LLM call")
    return state