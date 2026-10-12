"""Eval session wiring: print the metrics table and write results JSON after the run."""

import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from tests.evals.report import RESULTS, render_table, summarize, write_json

from duitku.agent.prompts import SYSTEM_PROMPT_TEMPLATE
from duitku.config import get_settings
from duitku.tools.registry import build_tools

RESULTS_DIR = Path(__file__).parent / "results"


def _prompt_version() -> str:
    """Short hash of the system prompt + tool descriptions/schemas, to compare runs."""
    parts = [SYSTEM_PROMPT_TEMPLATE]
    for t in build_tools():
        parts.append(json.dumps(convert_to_openai_tool(t), sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:10]


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not RESULTS:
        return
    meta = {"model": get_settings().model_agent, "prompt_version": _prompt_version()}
    summary = summarize(sorted(RESULTS, key=lambda r: r.id), meta)
    path = write_json(summary, RESULTS_DIR)
    terminalreporter.section("eval results")
    terminalreporter.write_line(f"model={meta['model']} prompt_version={meta['prompt_version']}")
    terminalreporter.write_line(render_table(summary))
    terminalreporter.write_line(f"json: {path}")
