"""CLI: `python -m duitku.interfaces.cli [--verbose] [--user NAME] [--thread NAME]`.

A REPL over `TurnRunner`: one line in, one reply out, same thread across lines (and across runs,
because the thread lives in the Postgres checkpointer). Empty line, `/exit` or EOF quits.
`--verbose` prints every model step with its tool calls and observations: the ReAct loop made
visible. That prints message/tool content on purpose (developer asked for it); it is never logged.
"""

import argparse
import asyncio
import hashlib
import logging
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import TextIO

from duitku.agent.checkpointer import cli_thread_id, open_checkpointer
from duitku.agent.graph import build_graph
from duitku.config import ConfigError, get_settings
from duitku.interfaces.runner import TurnResult, TurnRunner
from duitku.interfaces.trace import format_trace

logger = logging.getLogger(__name__)

EXIT_COMMANDS = frozenset({"", "/exit"})


def cli_telegram_id(name: str) -> int:
    """Stable fake identity for a CLI user. `users` is keyed by telegram_user_id (BIGINT), so a
    name must become a number; negative values can never collide with real Telegram ids."""
    return -(int.from_bytes(hashlib.sha256(name.encode()).digest()[:7]) + 1)


async def repl(
    run: Callable[[str], Awaitable[TurnResult]],
    *,
    stdin: TextIO,
    out: TextIO,
    verbose: bool = False,
) -> None:
    interactive = stdin.isatty()
    while True:
        if interactive:
            out.write("> ")
            out.flush()
        line = stdin.readline()
        if not line:  # EOF
            return
        text = line.strip()
        if not interactive and text not in EXIT_COMMANDS:
            out.write(f"> {text}\n")  # piped runs: echo the input so the transcript reads
        if text in EXIT_COMMANDS:
            return
        result = await run(text)
        if verbose and result.trace:
            out.write(format_trace(result.trace) + "\n")
        out.write(result.text + "\n")
        out.flush()


async def _run_cli(user_name: str, thread_name: str, verbose: bool) -> None:
    async with open_checkpointer() as saver:
        runner = TurnRunner(build_graph(checkpointer=saver))
        user = await runner.resolve_user(cli_telegram_id(user_name), user_name)
        thread_id = cli_thread_id(thread_name)

        async def run(text: str) -> TurnResult:
            return await runner.run(thread_id, user, text)

        await repl(run, stdin=sys.stdin, out=sys.stdout, verbose=verbose)


def _raise_interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="duitku-cli", description="Chat with the Duitku agent.")
    ap.add_argument("--verbose", action="store_true", help="show tool calls and observations")
    ap.add_argument("--user", default="cli", help="CLI user name (separate data per name)")
    ap.add_argument("--thread", default=None, help="conversation name (default: the user name)")
    args = ap.parse_args(argv)
    try:
        logging.basicConfig(level=get_settings().log_level)
        for noisy in ("httpx", "httpx2", "httpcore"):  # per-request INFO lines drown the REPL
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # asyncio.run's own SIGINT handler only cancels the main task, which a blocking
        # readline() never notices (Ctrl-C would hang until Enter). A plain handler that raises
        # interrupts the read; the exception then unwinds `async with open_checkpointer()`.
        signal.signal(signal.SIGINT, _raise_interrupt)
        asyncio.run(_run_cli(args.user, args.thread or args.user, args.verbose))
    except ConfigError as e:
        print(f"Konfigurasi belum lengkap: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nSampai jumpa.")
    except Exception as e:  # entry point: report the type only, never a traceback or a URL
        logger.error("CLI failed to start or crashed (%s)", type(e).__name__)
        print(
            f"Gagal menjalankan CLI ({type(e).__name__}). Cek konfigurasi dan koneksi.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
