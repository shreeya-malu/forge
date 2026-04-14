"""
dashboard/app.py
Gradio dashboard for Forge — real-time reasoning traces, confidence scores,
test status, security findings, and W&B metrics.
"""
from __future__ import annotations
import json, threading, time
from typing import Optional
import gradio as gr

# ── State shared between pipeline thread and dashboard ────────────────────────
_pipeline_state: dict = {}
_pipeline_running: bool = False
_pipeline_done: bool = False
_pipeline_thread: Optional[threading.Thread] = None


def get_state() -> dict:
    return _pipeline_state


def set_state(state: dict):
    global _pipeline_state
    _pipeline_state = dict(state)


# ── Dashboard refresh helpers ──────────────────────────────────────────────────

def _confidence_bar(score: int) -> str:
    if score >= 80:
        color = "#27AE60"
        label = "HIGH"
    elif score >= 60:
        color = "#F2994A"
        label = "MED"
    else:
        color = "#EB5757"
        label = "LOW"
    bar = "█" * (score // 10) + "░" * (10 - score // 10)
    return f'<span style="color:{color};font-weight:bold">[{bar}] {score}% {label}</span>'


def _render_decisions(audit: list) -> str:
    if not audit:
        return "<i>No decisions recorded yet.</i>"
    rows = []
    for d in audit[-8:]:  # Show last 8
        conf = d.get("confidence", 0)
        color = "#27AE60" if conf >= 80 else ("#F2994A" if conf >= 60 else "#EB5757")
        alts = "<br>".join(f"• {a}" for a in d.get("alternatives", [])[:3])
        rows.append(f"""
        <tr>
          <td style="padding:6px;font-weight:bold;color:#1B4F8A">{d.get('agent','')}</td>
          <td style="padding:6px;color:{color};font-weight:bold">{conf}%</td>
          <td style="padding:6px">{d.get('reasoning','')[:120]}</td>
          <td style="padding:6px;font-size:0.85em;color:#718096">{alts or '—'}</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">Agent</th>
        <th style="padding:6px;text-align:left">Confidence</th>
        <th style="padding:6px;text-align:left">Reasoning</th>
        <th style="padding:6px;text-align:left">Alternatives</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_tasks(task_plan: list) -> str:
    if not task_plan:
        return "<i>Task plan not yet generated.</i>"
    rows = []
    status_colors = {
        "done": "#27AE60", "failed": "#EB5757",
        "in_progress": "#2D9CDB", "pending": "#718096", "skipped": "#F2994A",
    }
    status_icons = {
        "done": "✅", "failed": "❌", "in_progress": "⚙️", "pending": "⏳", "skipped": "⏭️",
    }
    for t in task_plan:
        status = t.get("status", "pending")
        color = status_colors.get(status, "#718096")
        icon = status_icons.get(status, "•")
        rows.append(f"""
        <tr>
          <td style="padding:5px;color:#1B4F8A;font-weight:bold">{t.get('id','')}</td>
          <td style="padding:5px">{t.get('title','')}</td>
          <td style="padding:5px;color:{color};font-weight:bold">{icon} {status.upper()}</td>
          <td style="padding:5px;color:#718096">{t.get('risk_level','').upper()}</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">ID</th>
        <th style="padding:6px;text-align:left">Task</th>
        <th style="padding:6px;text-align:left">Status</th>
        <th style="padding:6px;text-align:left">Risk</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_tests(test_results: dict) -> str:
    if not test_results:
        return "<i>No tests run yet.</i>"
    rows = []
    for filepath, result in test_results.items():
        passed = result.get("tests_passed", 0)
        failed = result.get("tests_failed", 0)
        status = "✅ PASS" if result.get("passed") else "❌ FAIL"
        color = "#27AE60" if result.get("passed") else "#EB5757"
        rows.append(f"""
        <tr>
          <td style="padding:5px;font-family:monospace;font-size:0.85em">{filepath}</td>
          <td style="padding:5px;color:{color};font-weight:bold">{status}</td>
          <td style="padding:5px;color:#27AE60">{passed} passed</td>
          <td style="padding:5px;color:#EB5757">{failed} failed</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">Test File</th>
        <th style="padding:6px;text-align:left">Result</th>
        <th style="padding:6px;text-align:left">Passed</th>
        <th style="padding:6px;text-align:left">Failed</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_security(findings: list) -> str:
    if not findings:
        return '<span style="color:#27AE60;font-weight:bold">✅ No security findings.</span>'
    rows = []
    sev_colors = {"HIGH": "#EB5757", "MEDIUM": "#F2994A", "LOW": "#2D9CDB"}
    for f in findings:
        color = sev_colors.get(f.get("severity", "LOW"), "#718096")
        fix_status = "✅ Fixed" if f.get("fix_applied") else "⚠️ Open"
        fix_color = "#27AE60" if f.get("fix_applied") else "#F2994A"
        rows.append(f"""
        <tr>
          <td style="padding:5px;font-family:monospace;font-size:0.85em">{f.get('file','')}</td>
          <td style="padding:5px;color:{color};font-weight:bold">{f.get('severity','')}</td>
          <td style="padding:5px">{f.get('issue','')[:80]}</td>
          <td style="padding:5px;color:{fix_color};font-weight:bold">{fix_status}</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">File</th>
        <th style="padding:6px;text-align:left">Severity</th>
        <th style="padding:6px;text-align:left">Issue</th>
        <th style="padding:6px;text-align:left">Fix</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_metrics(state: dict) -> str:
    metrics = state.get("agent_metrics", {})
    total_tokens = state.get("total_tokens", 0)
    api_calls = state.get("api_call_count", 0)
    phase = state.get("phase", "init")
    status = state.get("pipeline_status", "running")

    status_color = "#27AE60" if status == "completed" else ("#EB5757" if status == "failed" else "#2D9CDB")

    agent_rows = ""
    for agent, m in metrics.items():
        if not m:
            continue
        agent_rows += f"""
        <tr>
          <td style="padding:4px;font-weight:bold;color:#1B4F8A">{agent}</td>
          <td style="padding:4px">{m.get('latency_s', '—')}s</td>
          <td style="padding:4px">{m.get('tokens', '—')}</td>
          <td style="padding:4px">{m.get('confidence', '—')}{'%' if m.get('confidence') else ''}</td>
        </tr>"""

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:16px">
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.8em;font-weight:bold;color:#1B4F8A">{api_calls}</div>
        <div style="color:#718096;font-size:0.85em">API Calls</div>
      </div>
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.8em;font-weight:bold;color:#1B4F8A">{total_tokens:,}</div>
        <div style="color:#718096;font-size:0.85em">Total Tokens</div>
      </div>
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.4em;font-weight:bold;color:#1B4F8A">{phase.replace('_',' ').title()}</div>
        <div style="color:#718096;font-size:0.85em">Current Phase</div>
      </div>
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.4em;font-weight:bold;color:{status_color}">{status.upper()}</div>
        <div style="color:#718096;font-size:0.85em">Status</div>
      </div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:0.88em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:5px;text-align:left">Agent</th>
        <th style="padding:5px;text-align:left">Latency</th>
        <th style="padding:5px;text-align:left">Tokens</th>
        <th style="padding:5px;text-align:left">Confidence</th>
      </tr>
      {agent_rows or '<tr><td colspan="4" style="padding:8px;text-align:center;color:#718096">Pipeline not started</td></tr>'}
    </table>"""


def _render_preview(state: dict) -> str:
    files     = state.get("generated_files", {})
    arch      = state.get("architecture", {})
    stack     = state.get("inferred_stack", {})
    summary   = state.get("workflow_summary", {})

    if not files:
        return "<i>No files generated yet.</i>"

    py_files   = sorted(f for f in files if f.endswith(".py") and "PLACEHOLDER" not in (files[f] or "")[:80])
    cfg_files  = sorted(f for f in files if not f.endswith(".py"))
    total_lines = sum(len((files[f] or "").splitlines()) for f in py_files)

    tree_rows = ""
    for f in py_files + cfg_files:
        lines = len((files[f] or "").splitlines())
        icon  = "🐍" if f.endswith(".py") else ("🐳" if "docker" in f.lower() or f == "Dockerfile" else "📄")
        tree_rows += f'<tr><td style="padding:3px 8px;font-family:monospace;font-size:0.85em">{icon} {f}</td><td style="padding:3px 8px;color:#718096;font-size:0.85em">{lines} lines</td></tr>'

    req = files.get("requirements.txt", "")
    req_html = ""
    if req:
        pkgs = [l.strip() for l in req.splitlines() if l.strip() and not l.startswith("#")]
        req_html = "<b>Dependencies:</b> " + ", ".join(
            f'<code style="background:#EDF2F7;padding:1px 4px;border-radius:3px">{p.split(">=")[0].split("==")[0]}</code>'
            for p in pkgs[:12]
        )
        if len(pkgs) > 12:
            req_html += f" +{len(pkgs)-12} more"

    framework  = stack.get("framework", "FastAPI")
    entrypoint = next((f.replace("/", ".").replace(".py", "") for f in py_files if f.endswith("main.py")), "app.main")
    run_cmd    = f"uvicorn {entrypoint}:app --reload --port 8000" if "fastapi" in framework.lower() else "python -m app.main"

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div>
        <div style="font-weight:bold;color:#0D1B2A;margin-bottom:8px">📁 Generated File Tree</div>
        <table style="width:100%;border-collapse:collapse;background:#F7FAFC;border-radius:6px;overflow:hidden">
          {tree_rows}
        </table>
        <div style="margin-top:8px;color:#718096;font-size:0.85em">
          {len(py_files)} Python files · {len(cfg_files)} config files · {total_lines:,} total lines
        </div>
      </div>
      <div>
        <div style="font-weight:bold;color:#0D1B2A;margin-bottom:8px">🚀 How to run locally</div>
        <div style="background:#1A202C;color:#E2E8F0;padding:12px;border-radius:6px;font-family:monospace;font-size:0.85em;line-height:1.8">
          <span style="color:#68D391"># 1. Extract the downloaded ZIP</span><br>
          cd generated_project<br><br>
          <span style="color:#68D391"># 2. Install dependencies</span><br>
          pip install -r requirements.txt<br><br>
          <span style="color:#68D391"># 3. Run the server</span><br>
          {run_cmd}<br><br>
          <span style="color:#68D391"># 4. Open in browser</span><br>
          http://localhost:8000/docs
        </div>
        <div style="margin-top:12px">{req_html}</div>
        <div style="margin-top:12px">
          <div style="font-weight:bold;color:#0D1B2A;margin-bottom:6px">🏗️ Stack</div>
          {"".join(f'<span style="background:#EBF8FF;color:#1B4F8A;padding:3px 8px;border-radius:12px;font-size:0.85em;margin:2px;display:inline-block">{k}: {v}</span>' for k,v in stack.items())}
        </div>
      </div>
    </div>
    <div style="margin-top:16px;padding:12px;background:#F0FFF4;border-radius:6px;border:1px solid #27AE60">
      <b style="color:#22543D">📊 Stats:</b>
      <span style="color:#276749;margin-left:12px">{summary.get("tasks_completed",0)}/{summary.get("tasks_total",0)} tasks completed · {summary.get("test_pass_rate",0)}% tests passing · {summary.get("api_calls_total",0)} API calls · {summary.get("total_tokens",0):,} tokens used</span>
    </div>"""


def _make_zip() -> str:
    import shutil
    from pathlib import Path
    project_dir = Path("./generated_project")
    if not project_dir.exists():
        return None
    zip_path = "/tmp/forge_generated_project"
    shutil.make_archive(zip_path, "zip", ".", "generated_project")
    return zip_path + ".zip"



def _render_summary(summary: dict) -> str:
    if not summary:
        return "<i>Pipeline not complete yet.</i>"
    return f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
      <div style="background:#F0FFF4;padding:10px;border-radius:8px;text-align:center;border:1px solid #27AE60">
        <div style="font-size:2em;font-weight:bold;color:#27AE60">{summary.get('tasks_completed',0)}/{summary.get('tasks_total',0)}</div>
        <div style="color:#718096">Tasks Completed</div>
      </div>
      <div style="background:#EBF8FF;padding:10px;border-radius:8px;text-align:center;border:1px solid #2D9CDB">
        <div style="font-size:2em;font-weight:bold;color:#2D9CDB">{summary.get('files_generated',0)}</div>
        <div style="color:#718096">Files Generated</div>
      </div>
      <div style="background:#F0FFF4;padding:10px;border-radius:8px;text-align:center;border:1px solid #27AE60">
        <div style="font-size:2em;font-weight:bold;color:#27AE60">{summary.get('test_pass_rate',0)}%</div>
        <div style="color:#718096">Test Pass Rate</div>
      </div>
      <div style="background:#FFF5F5;padding:10px;border-radius:8px;text-align:center;border:1px solid #EB5757">
        <div style="font-size:2em;font-weight:bold;color:#EB5757">{summary.get('security_findings_total',0)}</div>
        <div style="color:#718096">Security Findings</div>
      </div>
      <div style="background:#F0FFF4;padding:10px;border-radius:8px;text-align:center;border:1px solid #27AE60">
        <div style="font-size:2em;font-weight:bold;color:#27AE60">{summary.get('security_findings_fixed',0)}</div>
        <div style="color:#718096">Findings Fixed</div>
      </div>
      <div style="background:#EBF8FF;padding:10px;border-radius:8px;text-align:center;border:1px solid #2D9CDB">
        <div style="font-size:2em;font-weight:bold;color:#1B4F8A">{summary.get('api_calls_total',0)}</div>
        <div style="color:#718096">API Calls Total</div>
      </div>
    </div>"""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def create_dashboard():
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(
        title="Forge — Engineering Intelligence System",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="orange",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .forge-header { background: #0D1B2A; padding: 20px; border-radius: 8px; margin-bottom: 16px; }
        .forge-header h1 { color: #2D9CDB; margin: 0; font-size: 2em; }
        .forge-header p { color: #A0AEC0; margin: 4px 0 0 0; }
        .section-label { color: #0D1B2A; font-weight: bold; font-size: 1.05em; margin-bottom: 4px; }
        """,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="forge-header">
          <h1>⚙️ FORGE</h1>
          <p>Engineering Intelligence System — Agentic AI Framework</p>
        </div>""")

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Project Specification",
                placeholder="Describe the software project you want to build...\nExample: Build a REST API for a personal expense tracker with categories, tags, and monthly budget limits.",
                lines=4,
                scale=4,
            )
            with gr.Column(scale=1):
                run_btn = gr.Button(
                    "🚀 Run Forge", variant="primary", size="lg",
                    interactive=False,   # disabled until prompt is entered
                )
                stop_btn = gr.Button(
                    "⏹ Stop", variant="stop", size="sm",
                    interactive=False,   # disabled until pipeline is running
                )
                status_badge = gr.HTML('<span style="color:#718096">Ready — enter a prompt to begin</span>')

        # Priority weights
        with gr.Accordion("⚙️ Priority Weights (optional)", open=False):
            with gr.Row():
                w_speed    = gr.Slider(0, 1, value=0.2,  step=0.05, label="Generation Speed")
                w_quality  = gr.Slider(0, 1, value=0.25, step=0.05, label="Code Quality")
                w_tests    = gr.Slider(0, 1, value=0.25, step=0.05, label="Test Coverage")
                w_security = gr.Slider(0, 1, value=0.2,  step=0.05, label="Security Hardness")
                w_simple   = gr.Slider(0, 1, value=0.1,  step=0.05, label="Simplicity")

        gr.HTML("<hr>")

        # Tabs
        with gr.Tabs():

            with gr.TabItem("📊 Live Metrics"):
                metrics_html = gr.HTML("<i>Start the pipeline to see metrics.</i>")

            with gr.TabItem("🧠 Decision Audit Trail"):
                gr.HTML('<p style="color:#718096;font-size:0.9em">Every architectural decision — what was considered, why it was chosen, confidence score.</p>')
                decisions_html = gr.HTML("<i>No decisions yet.</i>")

            with gr.TabItem("📋 Task Plan"):
                tasks_html = gr.HTML("<i>Task plan not generated yet.</i>")

            with gr.TabItem("🧪 Test Results"):
                tests_html = gr.HTML("<i>No tests run yet.</i>")

            with gr.TabItem("🔐 Security"):
                security_html = gr.HTML("<i>Security audit not run yet.</i>")

            with gr.TabItem("🏗️ Architecture"):
                arch_html = gr.HTML("<i>Architecture not generated yet.</i>")

            with gr.TabItem("📁 Files & Editor"):
                with gr.Row():
                    file_selector = gr.Dropdown(
                        label="Select file", choices=[], interactive=True, scale=3
                    )
                    download_btn = gr.Button("⬇ Download ZIP", size="sm", scale=1)
                file_content = gr.Textbox(
                    label="File content (editable — select a file above)",
                    lines=25,
                    max_lines=60,
                    interactive=True,
                    show_copy_button=True,
                )
                with gr.Row():
                    save_btn   = gr.Button("💾 Save changes", variant="primary", size="sm",
                                           interactive=False)
                    save_status = gr.HTML("")
                gr.HTML('<hr style="margin:8px 0">')
                gr.HTML('<div style="font-weight:bold;color:#0D1B2A;margin-bottom:4px">✏️ AI Edit — describe a change to apply to the selected file</div>')
                with gr.Row():
                    edit_prompt = gr.Textbox(
                        placeholder='e.g. "Add a /expenses/summary endpoint that returns total by category"',
                        label="", scale=4, lines=2,
                    )
                    ai_edit_btn = gr.Button("Apply change", variant="secondary", size="sm", scale=1)
                ai_edit_status = gr.HTML("")
                download_file  = gr.File(label="Download", visible=False)

            with gr.TabItem("🚀 Project Preview"):
                preview_html = gr.HTML("<i>Run the pipeline to see the project preview.</i>")

            with gr.TabItem("✅ Summary"):
                summary_html = gr.HTML("<i>Pipeline not complete.</i>")

        # Error banner — shown on crash, hidden otherwise
        error_banner = gr.HTML("", visible=False)

        # Activity log
        with gr.Accordion("📜 Activity Log", open=False):
            activity_log = gr.Textbox(
                label="",
                lines=12,
                max_lines=20,
                interactive=False,
            )

        # ── Event handlers ────────────────────────────────────────────────────

        # Enable Run button only when prompt has content
        def on_prompt_change(text):
            has_text = bool(text and text.strip())
            return gr.update(interactive=has_text)

        prompt_input.change(fn=on_prompt_change, inputs=[prompt_input], outputs=[run_btn])

        def stop_pipeline():
            global _pipeline_running, _pipeline_done
            _pipeline_running = False
            _pipeline_done = True
            log_action("Dashboard", "Stop requested by user")
            return (
                '<span style="color:#F2994A;font-weight:bold">⏹ Stopped by user</span>',
                gr.update(interactive=True),   # run_btn
                gr.update(interactive=False),  # stop_btn
            )

        stop_btn.click(
            fn=stop_pipeline,
            inputs=[],
            outputs=[status_badge, run_btn, stop_btn],
        )

        def run_pipeline(prompt, w_spd, w_qual, w_tst, w_sec, w_smp):
            global _pipeline_running, _pipeline_done, _pipeline_thread, _pipeline_state

            if not prompt.strip():
                yield (
                    '<span style="color:#EB5757">⚠️ Please enter a project specification.</span>',
                    gr.update(interactive=True),   # run_btn
                    gr.update(interactive=False),  # stop_btn
                    gr.update(visible=False),      # error_banner
                    *([gr.update()] * 8),          # 8 tab html outputs
                    "",                            # activity_log
                    gr.update(choices=[]),         # file_selector
                )
                return

            _pipeline_running = True
            _pipeline_done = False
            _pipeline_state = {}

            priority_weights = {
                "speed": w_spd, "quality": w_qual,
                "test_coverage": w_tst, "security": w_sec, "simplicity": w_smp,
            }

            def _run():
                global _pipeline_state, _pipeline_done, _pipeline_running
                try:
                    from core.graph import build_graph, get_initial_state
                    from core.observability import init_wandb, init_langsmith, init_log
                    import os, yaml
                    from pathlib import Path

                    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))

                    init_log(cfg["output"]["log_file"])
                    if os.environ.get("WANDB_API_KEY"):
                        init_wandb(cfg["observability"]["wandb_project"], {
                            "prompt": prompt[:200],
                            "priority_weights": priority_weights,
                        })
                    if os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"):
                        init_langsmith(cfg["observability"]["langsmith_project"])

                    initial = get_initial_state(prompt, priority_weights)
                    graph = build_graph()

                    for step in graph.stream(initial):
                        if not _pipeline_running:  # stop button was pressed
                            break
                        for node_name, node_state in step.items():
                            set_state(node_state)

                    _pipeline_done = True

                except Exception as e:
                    import traceback as tb
                    err_msg = str(e)
                    trace = tb.format_exc()
                    friendly = _classify_crash(err_msg)
                    current = dict(_pipeline_state)
                    current["pipeline_status"] = "failed"
                    current["crash_error"] = err_msg
                    current["crash_traceback"] = trace
                    current["crash_friendly"] = friendly
                    set_state(current)
                    _pipeline_done = True

                finally:
                    _pipeline_running = False

            _pipeline_thread = threading.Thread(target=_run, daemon=True)
            _pipeline_thread.start()

            # Poll and yield updates every second
            for _ in range(600):  # max 10 minutes
                time.sleep(1)
                state = get_state()

                phase  = state.get("phase", "init")
                status = state.get("pipeline_status", "running")
                crash  = state.get("crash_error")
                done   = _pipeline_done

                # ── Button states ─────────────────────────────────────────────
                run_btn_update  = gr.update(interactive=done)
                stop_btn_update = gr.update(interactive=not done)

                # ── Status badge ──────────────────────────────────────────────
                if status == "completed":
                    status_html = '<span style="color:#27AE60;font-weight:bold">✅ Completed</span>'
                elif status == "failed":
                    status_html = '<span style="color:#EB5757;font-weight:bold">❌ Failed</span>'
                elif not _pipeline_running and done:
                    status_html = '<span style="color:#F2994A;font-weight:bold">⏹ Stopped</span>'
                else:
                    status_html = (
                        f'<span style="color:#2D9CDB;font-weight:bold">'
                        f'⚙️ {phase.replace("_", " ").title()}'
                        f'</span>'
                    )

                # ── Error banner ──────────────────────────────────────────────
                if crash:
                    friendly = state.get("crash_friendly", "An unexpected error occurred.")
                    trace_lines = (state.get("crash_traceback", "").strip().split("\n"))
                    trace_display = "\n".join(trace_lines[-6:])
                    err_html = f"""
                    <div style="background:#FFF5F5;border:2px solid #EB5757;border-radius:8px;
                                padding:16px;margin:8px 0;">
                      <div style="font-size:1.1em;font-weight:bold;color:#EB5757;margin-bottom:8px;">
                        ❌ Pipeline crashed
                      </div>
                      <div style="color:#C53030;font-weight:bold;margin-bottom:8px;">{friendly}</div>
                      <details>
                        <summary style="cursor:pointer;color:#718096;font-size:0.9em;">
                          Show technical details
                        </summary>
                        <pre style="background:#1A202C;color:#FC8181;padding:10px;border-radius:4px;
                                    font-size:0.8em;overflow-x:auto;margin-top:8px;">{trace_display}</pre>
                        <div style="color:#718096;font-size:0.85em;margin-top:4px;">
                          Full error: {str(crash)[:300]}
                        </div>
                      </details>
                      <div style="margin-top:12px;font-size:0.9em;color:#4A5568;">
                        💡 <b>What to try:</b> Check the Activity Log for the last successful step.
                        Any files generated so far are still in the Generated Files tab.
                      </div>
                    </div>"""
                    err_update = gr.update(value=err_html, visible=True)
                else:
                    err_update = gr.update(visible=False)

                # ── Activity log ──────────────────────────────────────────────
                from core.observability import get_log_buffer
                log_lines = "\n".join(
                    f"[{e.get('timestamp','')[-8:-1]}] [{e.get('level','INFO')}]"
                    f" [{e.get('agent','')}] {e.get('action','')}"
                    + (f" — {e.get('detail','')[:80]}" if e.get('detail') else "")
                    for e in get_log_buffer()[-40:]
                )

                # ── Architecture tab ──────────────────────────────────────────
                arch = state.get("architecture", {})
                arch_text = f"<b>Pattern:</b> {state.get('chosen_pattern', '—')}<br><br>"
                if arch.get("key_decisions"):
                    arch_text += "<b>Key Decisions:</b><ul>" + "".join(
                        f"<li><b>{d.get('decision','')}</b>: {d.get('rationale','')}</li>"
                        for d in arch["key_decisions"]
                    ) + "</ul>"
                validation = state.get("architecture_validation", {})
                if validation.get("verdict"):
                    vcolor = "#27AE60" if "approved" in validation["verdict"] else "#EB5757"
                    arch_text += (
                        f'<br><b>Validation:</b> '
                        f'<span style="color:{vcolor};font-weight:bold">'
                        f'{validation["verdict"].upper()}</span><br>'
                        f'{validation.get("validation_summary", "")}'
                    )
                if validation.get("flagged_issues"):
                    arch_text += "<br><b>Flagged:</b><ul>" + "".join(
                        f'<li style="color:#F2994A">[{i.get("severity","")}] '
                        f'{i.get("issue","")} — {i.get("recommendation","")}</li>'
                        for i in validation["flagged_issues"]
                    ) + "</ul>"

                files = list(state.get("generated_files", {}).keys())

                yield (
                    status_html,
                    run_btn_update,
                    stop_btn_update,
                    err_update,
                    _render_metrics(state),
                    _render_decisions(state.get("decision_audit", [])),
                    _render_tasks(state.get("task_plan", [])),
                    _render_tests(state.get("test_results", {})),
                    _render_security(state.get("security_findings", [])),
                    arch_text,
                    _render_preview(state),
                    _render_summary(state.get("workflow_summary", {})),
                    log_lines,
                    gr.update(choices=files),
                )

                if done:
                    break

        def view_file(filename):
            if not filename:
                return "", gr.update(interactive=False), ""
            state = get_state()
            content = state.get("generated_files", {}).get(filename, "")
            if not content:
                return f"# {filename} — not found", gr.update(interactive=False), ""
            # Return content as plain str — gr.Code/Textbox does not accept tuple
            return content, gr.update(interactive=True), ""

        def save_file(filename, new_content):
            if not filename:
                return gr.update(interactive=True), '<span style="color:#EB5757">⚠ No file selected</span>'
            state = get_state()
            state.setdefault("generated_files", {})[filename] = new_content
            set_state(state)
            from pathlib import Path
            out = Path("./generated_project") / filename
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(new_content)
                return gr.update(interactive=True), f'<span style="color:#27AE60">✅ Saved {filename}</span>'
            except Exception as e:
                return gr.update(interactive=True), f'<span style="color:#EB5757">Save error: {e}</span>'

        def ai_edit_file(filename, current_content, change_prompt):
            """Apply a natural-language edit to the selected file using the LLM."""
            if not filename:
                return current_content, '<span style="color:#EB5757">⚠ No file selected</span>'
            if not change_prompt.strip():
                return current_content, '<span style="color:#EB5757">⚠ Describe the change to make</span>'
            try:
                from core.llm import get_client
                client = get_client()
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are editing a source file. Apply ONLY the requested change. "
                            "Return the complete updated file as raw code. "
                            "No markdown fences, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"File: {filename}\n\n"
                            f"Current content:\n{current_content[:4000]}\n\n"
                            f"Change to apply: {change_prompt}\n\n"
                            "Return the complete updated file."
                        ),
                    },
                ]
                response, _ = client.call(messages, agent_name="AIEditor")
                new_code = response.strip()
                if new_code.startswith("```"):
                    lines = new_code.split("\n")
                    end = len(lines) - 1
                    while end > 0 and lines[end].strip().startswith("```"):
                        end -= 1
                    new_code = "\n".join(lines[1:end + 1]).strip()
                from pathlib import Path
                out = Path("./generated_project") / filename
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(new_code)
                state = get_state()
                state.setdefault("generated_files", {})[filename] = new_code
                set_state(state)
                return new_code, f'<span style="color:#27AE60">✅ Applied: {change_prompt[:60]}</span>'
            except Exception as e:
                return current_content, f'<span style="color:#EB5757">Error: {e}</span>'

        def download_zip():
            zip_path = _make_zip()
            if not zip_path:
                return gr.update(visible=False)
            return gr.update(value=zip_path, visible=True)

        file_selector.change(
            fn=view_file,
            inputs=[file_selector],
            outputs=[file_content, save_btn, save_status],
        )
        save_btn.click(
            fn=save_file,
            inputs=[file_selector, file_content],
            outputs=[save_btn, save_status],
        )
        ai_edit_btn.click(
            fn=ai_edit_file,
            inputs=[file_selector, file_content, edit_prompt],
            outputs=[file_content, ai_edit_status],
        )
        download_btn.click(
            fn=download_zip,
            inputs=[],
            outputs=[download_file],
        )

        def _classify_crash(err: str) -> str:
            """Return a human-friendly explanation of a crash cause."""
            e = err.lower()
            if "rate_limit" in e or "429" in e:
                return ("🚦 Groq rate limit hit. The pipeline made too many API calls too quickly. "
                        "Wait 60 seconds and try again, or reduce the number of tasks in your project.")
            if "token" in e and ("limit" in e or "exceed" in e or "context" in e):
                return ("📏 Token limit exceeded. The project was too large for one run. "
                        "Try a simpler project description, or break it into smaller scopes.")
            if "groq_api_key" in e or "api_key" in e or "authentication" in e:
                return ("🔑 Groq API key missing or invalid. "
                        "Make sure GROQ_API_KEY is set correctly in your environment.")
            if "connection" in e or "timeout" in e or "network" in e:
                return ("🌐 Network error connecting to Groq API. "
                        "Check your internet connection and try again.")
            if "json" in e and "parse" in e:
                return ("🧩 The LLM returned malformed JSON. This sometimes happens with complex prompts. "
                        "Try running again — it usually succeeds on the second attempt.")
            if "filenotfounderror" in e or "no such file" in e:
                return ("📂 A required file was not found. "
                        "Make sure the forge/ directory structure is intact.")
            return ("⚠️ An unexpected error stopped the pipeline. "
                    "See the technical details below, and check the Activity Log for context.")

        run_btn.click(
            fn=run_pipeline,
            inputs=[prompt_input, w_speed, w_quality, w_tests, w_security, w_simple],
            outputs=[
                status_badge,
                run_btn,
                stop_btn,
                error_banner,
                metrics_html,
                decisions_html,
                tasks_html,
                tests_html,
                security_html,
                arch_html,
                preview_html,
                summary_html,
                activity_log,
                file_selector,
            ],
        )

    return demo


def launch(share: bool = True, port: int = 7860):
    """Launch the Forge dashboard."""
    demo = create_dashboard()
    demo.launch(
        share=share,
        server_port=port,
        show_error=True,
        quiet=False,
    )