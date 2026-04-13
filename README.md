# ⚙️ FORGE — Engineering Intelligence System

> From natural-language prompt → fully-verified, Dockerized software  
> with transparent reasoning, confidence-aware agents, and measurable decision intelligence.

---

## Quick Start (Google Colab)

1. Upload the `forge/` folder to Colab
2. Open `Forge_Launcher.ipynb` in Colab
3. Run cells top to bottom
4. Set your Groq API key when prompted
5. Launch the dashboard or run CLI

## Project Structure

```
forge/
├── Forge_Launcher.ipynb      ← Start here (Colab notebook)
├── main.py                   ← CLI entry point
├── config.yaml               ← All configurable settings
├── requirements.txt
│
├── agents/
│   ├── requirement_agent.py  ← Parses prompt, generates spec
│   ├── planning_agent.py     ← Selects pattern, decomposes tasks
│   ├── architecture_validator.py  ← Cross-checks architecture
│   ├── qa_agent.py           ← TDD-first test generation
│   ├── codegen_agent.py      ← File-by-file code generation
│   ├── debug_agent.py        ← Targeted error fixing
│   ├── security_audit_agent.py    ← bandit + immediate fixes
│   └── docker_agent.py       ← Dockerfile + docker-compose
│
├── core/
│   ├── state.py              ← LangGraph TypedDict state schema
│   ├── graph.py              ← Pipeline graph + conditional routing
│   ├── llm.py                ← Groq client wrapper
│   ├── observability.py      ← W&B + LangSmith + activity log
│   └── validators.py         ← ruff, pylint, bandit, pytest runners
│
├── dashboard/
│   └── app.py                ← Gradio real-time dashboard
│
└── patterns/
    └── library.json          ← Architecture pattern library
```

## Agent Pipeline

```
RequirementAgent → PlanningAgent → ArchitectureValidatorAgent
    → QAAgent → CodeGenAgent → SecurityAuditAgent
    → SecurityFixAgent → DockerAgent → Summary
```

## Confidence Routing

| Score | Behaviour |
|-------|-----------|
| ≥ 80% | Proceed autonomously |
| 60–79% | Flag in dashboard, proceed |
| < 60%  | Pause, request human input |

## Configuration

All settings in `config.yaml`:
- Model selection (planning vs codegen vs fast)
- Confidence thresholds
- Retry limits
- W&B / LangSmith project names
- Validator toggles

## Complexity Tiers

| Tier | Project | Core Challenge |
|------|---------|----------------|
| 1 | The Ledger | Basic CRUD with strict schema validation |
| 2 | Logic Engine | Dynamic business rules engine |
| 3 | Live Bridge | Async 3rd-party API integration |
| 4 | The Gatekeeper | OAuth2 + JWT + RBAC |
| 5 | Mongo-SQL Engine | MongoDB joins + Change Streams |

## API Keys

| Key | Required | Where to get |
|-----|----------|--------------|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) (free) |
| `WANDB_API_KEY` | Optional | [wandb.ai](https://wandb.ai) (free) |
| `LANGSMITH_API_KEY` | Optional | [smith.langchain.com](https://smith.langchain.com) (free) |
