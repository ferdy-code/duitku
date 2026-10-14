"""Per-run metrics: summary numbers, terminal table, JSON file. Pure except `write_json`."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    passed: bool
    failures: list[str] = field(default_factory=lambda: list[str]())
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    replies: list[str] = field(default_factory=lambda: list[str]())  # agent text, per turn
    calls: list[str] = field(default_factory=lambda: list[str]())  # 'name(args) -> envelope'


# Filled by test_evals.py, read by the terminal-summary hook in conftest.py.
RESULTS: list[CaseResult] = []


def _rate(results: list[CaseResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def summarize(results: list[CaseResult], meta: dict[str, Any]) -> dict[str, Any]:
    by_tag: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        for t in r.tags:
            by_tag[t].append(r)
    n = len(results)
    return {
        **meta,
        "cases": n,
        "passed": sum(r.passed for r in results),
        "pass_rate": _rate(results),
        "avg_steps": sum(r.steps for r in results) / n if n else 0.0,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "pass_rate_by_tag": {t: _rate(rs) for t, rs in sorted(by_tag.items())},
        "results": [asdict(r) for r in results],
    }


def render_table(summary: dict[str, Any]) -> str:
    lines = [f"{'case':<6} {'result':<6} {'steps':>5} {'tok in':>8} {'tok out':>8} {'sec':>6}"]
    for r in summary["results"]:
        lines.append(
            f"{r['id']:<6} {'PASS' if r['passed'] else 'FAIL':<6} {r['steps']:>5} "
            f"{r['input_tokens']:>8} {r['output_tokens']:>8} {r['latency_s']:>6.1f}"
        )
        lines += [f"         - {f}" for f in r["failures"]]
    tags = ", ".join(f"{t} {v:.0%}" for t, v in summary["pass_rate_by_tag"].items())
    lines += [
        f"pass rate {summary['passed']}/{summary['cases']} = {summary['pass_rate']:.0%}; "
        f"avg steps {summary['avg_steps']:.1f}; "
        f"tokens in={summary['input_tokens']} out={summary['output_tokens']}",
        f"by tag: {tags}",
    ]
    return "\n".join(lines)


def write_json(summary: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
