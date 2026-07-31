"""
observability/tracing.py

Arize Phoenix observability for FinTrust Compass.

Responsibilities:
  1. Launch the Phoenix in-process server (on PHOENIX_PORT, default 6006)
  2. Register an OpenTelemetry TracerProvider that exports to Phoenix
  3. Auto-instrument LangChain so every LLM call, retrieval, and
     chain step is captured as a span automatically
  4. Expose helpers used by FastAPI and evaluation scripts

Usage (standalone — starts Phoenix + instruments immediately):
    python -W ignore observability/tracing.py

Imported by FastAPI lifespan — call setup_tracing() once at startup.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)


# ── internal state ─────────────────────────────────────────────────────────────
_tracing_active = False
_phoenix_url: str | None = None
_lock = threading.Lock()


# ── public API ─────────────────────────────────────────────────────────────────

def setup_tracing(
    phoenix_port: int | None = None,
    project_name: str = "fintrust-compass",
    enabled: bool | None = None,
) -> str | None:
    """
    Start Phoenix + register OTEL tracing with LangChain auto-instrumentation.

    Returns the Phoenix UI URL if tracing was enabled, else None.
    Can be called multiple times safely — only initialises once.
    """
    global _tracing_active, _phoenix_url

    with _lock:
        if _tracing_active:
            return _phoenix_url

        # Determine whether tracing is enabled
        if enabled is None:
            enabled = os.getenv("ENABLE_PHOENIX_TRACING", "true").lower() in ("1", "true", "yes")

        if not enabled:
            log.info("[Tracing] ENABLE_PHOENIX_TRACING=false — skipping")
            return None

        if phoenix_port is None:
            phoenix_port = int(os.getenv("PHOENIX_PORT", "6006"))

        # ── 1. Launch Phoenix in-process ─────────────────────────────────────
        try:
            import phoenix as px
            # Set port via env var (launch_app `port` param is deprecated in v19+)
            os.environ.setdefault("PHOENIX_PORT", str(phoenix_port))
            session = px.launch_app(run_in_thread=True)
            _phoenix_url = session.url if hasattr(session, "url") else f"http://localhost:{phoenix_port}"
            log.info("[Tracing] Phoenix UI: %s", _phoenix_url)
        except Exception as exc:
            log.warning("[Tracing] Could not launch Phoenix: %s", exc)
            _phoenix_url = f"http://localhost:{phoenix_port}"

        # ── 2. Register OTEL TracerProvider → Phoenix collector ───────────────
        try:
            from phoenix.otel import register
            register(
                endpoint=f"http://localhost:{phoenix_port}/v1/traces",
                project_name=project_name,
                batch=True,
                set_global_tracer_provider=True,
                verbose=False,
            )
            log.info("[Tracing] OTEL TracerProvider registered → %s", _phoenix_url)
        except Exception as exc:
            log.warning("[Tracing] Could not register OTEL TracerProvider: %s", exc)

        # ── 3. Auto-instrument LangChain ─────────────────────────────────────
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            LangChainInstrumentor().instrument()
            log.info("[Tracing] LangChain auto-instrumentation active")
        except Exception as exc:
            log.warning("[Tracing] Could not instrument LangChain: %s", exc)

        _tracing_active = True
        return _phoenix_url


def get_phoenix_url() -> str | None:
    """Return the Phoenix UI URL if tracing has been started, else None."""
    return _phoenix_url


def is_tracing_active() -> bool:
    return _tracing_active


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    url = setup_tracing()
    if not url:
        print("[Tracing] Tracing is disabled (ENABLE_PHOENIX_TRACING=false).")
        sys.exit(0)

    print(f"\n  Phoenix UI  →  {url}")
    print("  Tracing is active. Run a query to generate traces.")
    print("  Press Ctrl+C to stop.\n")

    # Run a sample query so there's immediately something to see in Phoenix
    try:
        from agents.orchestrator import build_graph
        print("  [Demo] Running a sample query through the pipeline...")
        graph = build_graph()
        result = graph.invoke({
            "query": "What is the minimum balance for a FinTrust savings account?",
            "messages": [],
        })
        print(f"  [Demo] Domain: {result.get('domain')} | Answer[:80]: {result.get('answer', '')[:80]}")
        print(f"\n  Open {url} to see the trace in Phoenix.\n")
    except Exception as e:
        print(f"  [Demo] Could not run sample query: {e}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopped.")
