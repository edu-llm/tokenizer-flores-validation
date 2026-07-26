"""Portable corpus building and subprocess resource benchmarking.

The benchmark intentionally treats tokenizer training as an external command.
That keeps local runs and AWS Batch runs identical: both consume a byte-bound
corpus manifest, execute the same pinned trainer, and emit the same result
schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import psutil

SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def discover_input_files(inputs: Iterable[Path], output_path: Path | None = None) -> list[Path]:
    """Expand files/directories deterministically, excluding the output file."""
    output_resolved = output_path.resolve() if output_path else None
    discovered: set[Path] = set()
    for candidate in inputs:
        if candidate.is_file():
            discovered.add(candidate.resolve())
        elif candidate.is_dir():
            discovered.update(path.resolve() for path in candidate.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(f"Input does not exist: {candidate}")

    files = sorted(
        path
        for path in discovered
        if output_resolved is None or path != output_resolved
    )
    if not files:
        raise ValueError("No input files found")
    return files


@dataclass
class SourceContribution:
    source_uri: str
    bytes_written: int = 0
    records_written: int = 0


def _fit_utf8(payload: bytes, remaining: int) -> bytes:
    """Return the longest valid UTF-8 prefix that fits within remaining bytes."""
    if remaining <= 0:
        return b""
    return payload[:remaining].decode("utf-8", errors="ignore").encode("utf-8")


def build_round_robin_corpus(
    input_files: Sequence[Path],
    output_path: Path,
    target_bytes: int,
) -> dict[str, Any]:
    """Build a deterministic, approximately balanced line corpus.

    Files are traversed in sorted round-robin order. This is suitable for local
    multilingual smoke tests where each input file represents one language.
    Production AWS manifests will be produced by the Dolma mixer instead.
    """
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")

    files = discover_input_files(input_files, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contributions = {path: SourceContribution(path.as_uri()) for path in files}
    digest = hashlib.sha256()
    bytes_written = 0
    records_written = 0

    with ExitStack() as stack:
        streams = {
            path: stack.enter_context(path.open("r", encoding="utf-8", errors="strict"))
            for path in files
        }
        active = list(files)
        with output_path.open("wb") as output:
            while active and bytes_written < target_bytes:
                next_active: list[Path] = []
                for path in active:
                    line = streams[path].readline()
                    if not line:
                        continue
                    next_active.append(path)
                    payload = line.rstrip("\r\n").encode("utf-8") + b"\n"
                    fitted = _fit_utf8(payload, target_bytes - bytes_written)
                    if not fitted:
                        break
                    output.write(fitted)
                    digest.update(fitted)
                    size = len(fitted)
                    bytes_written += size
                    records_written += 1
                    contributions[path].bytes_written += size
                    contributions[path].records_written += 1
                    if bytes_written >= target_bytes:
                        break
                active = next_active

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "tokenizer_benchmark_corpus",
        "output_uri": output_path.resolve().as_uri(),
        "target_bytes": target_bytes,
        "actual_bytes": bytes_written,
        "records": records_written,
        "sha256": digest.hexdigest(),
        "sampling": {
            "method": "sorted_file_round_robin_lines",
            "seed": None,
        },
        "sources": [asdict(contributions[path]) for path in files],
    }
    return manifest


def _process_tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [root]


def _tree_metrics(root: psutil.Process) -> tuple[int, float]:
    rss = 0
    cpu_seconds = 0.0
    for process in _process_tree(root):
        try:
            rss += process.memory_info().rss
            times = process.cpu_times()
            cpu_seconds += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, cpu_seconds


def _terminate_tree(root: psutil.Process) -> None:
    processes = list(reversed(_process_tree(root)))
    for process in processes:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _directory_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def run_monitored_command(
    *,
    name: str,
    command: Sequence[str],
    result_path: Path,
    log_path: Path,
    input_bytes: int,
    projection_bytes: int,
    cwd: Path | None = None,
    output_path: Path | None = None,
    poll_seconds: float = 0.5,
    max_rss_gb: float | None = None,
    min_available_gb: float = 1.0,
    metadata: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute and monitor a tokenizer command, always writing a result JSON."""
    if not command:
        raise ValueError("command must not be empty")
    if input_bytes <= 0 or projection_bytes <= 0:
        raise ValueError("input_bytes and projection_bytes must be positive")

    available_start = psutil.virtual_memory().available
    minimum_required = int(min_available_gb * 1024**3)
    if available_start < minimum_required:
        raise RuntimeError(
            f"Only {available_start / 1024**3:.2f} GiB RAM available; "
            f"benchmark requires at least {min_available_gb:.2f} GiB"
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    peak_rss = 0
    min_available = available_start
    final_cpu_seconds = 0.0
    aborted_reason: str | None = None

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        root = psutil.Process(process.pid)

        def pump_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    print(line, end="", flush=True)
                except UnicodeEncodeError:
                    # Windows consoles often default to cp1252. Preserve the
                    # complete UTF-8 log and render unsupported glyphs safely.
                    safe_line = line.encode(
                        sys.stdout.encoding or "utf-8", errors="replace"
                    ).decode(sys.stdout.encoding or "utf-8", errors="replace")
                    print(safe_line, end="", flush=True)
                log.write(line)
                log.flush()

        output_thread = threading.Thread(target=pump_output, daemon=True)
        output_thread.start()

        while process.poll() is None:
            rss, cpu_seconds = _tree_metrics(root)
            available = psutil.virtual_memory().available
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, available)
            final_cpu_seconds = max(final_cpu_seconds, cpu_seconds)

            if max_rss_gb is not None and rss > max_rss_gb * 1024**3:
                aborted_reason = f"process tree exceeded max_rss_gb={max_rss_gb}"
                _terminate_tree(root)
                break
            if available < minimum_required:
                aborted_reason = (
                    f"system available RAM fell below min_available_gb={min_available_gb}"
                )
                _terminate_tree(root)
                break
            time.sleep(poll_seconds)

        return_code = process.wait()
        output_thread.join(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        rss, cpu_seconds = _tree_metrics(root)
        peak_rss = max(peak_rss, rss)
        final_cpu_seconds = max(final_cpu_seconds, cpu_seconds)

    ended_at = time.time()
    elapsed = ended_at - started_at
    throughput = input_bytes / elapsed if elapsed > 0 else None
    projection = elapsed * projection_bytes / input_bytes if elapsed > 0 else None
    virtual_memory = psutil.virtual_memory()

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "tokenizer_resource_benchmark",
        "name": name,
        "status": "aborted" if aborted_reason else ("succeeded" if return_code == 0 else "failed"),
        "return_code": return_code,
        "aborted_reason": aborted_reason,
        "command": list(command),
        "cwd": str(cwd.resolve()) if cwd else None,
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "elapsed_seconds": elapsed,
        "input_bytes": input_bytes,
        "throughput_bytes_per_second": throughput,
        "projection": {
            "target_bytes": projection_bytes,
            "linear_runtime_seconds": projection,
            "warning": "Runtime projection is linear; peak RAM is not safely extrapolated.",
        },
        "resources": {
            "peak_process_tree_rss_bytes": peak_rss,
            "process_tree_cpu_seconds": final_cpu_seconds,
            "system_total_memory_bytes": virtual_memory.total,
            "system_available_start_bytes": available_start,
            "system_minimum_available_bytes": min_available,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "output_bytes": _directory_size(output_path),
        "log_uri": log_path.resolve().as_uri(),
        "metadata": metadata or {},
    }
    atomic_write_json(result_path, result)
    return result

