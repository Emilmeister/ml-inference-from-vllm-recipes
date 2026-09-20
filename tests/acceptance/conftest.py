from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def browser_type_launch_args() -> dict[str, str]:
    candidates = (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
    )
    executable = next((path for path in candidates if path.is_file()), None)
    return {"executable_path": str(executable)} if executable else {}


@pytest.fixture(scope="session")
def frontend_url() -> Iterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            [
                "pnpm",
                "--dir",
                "frontend",
                "dev",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--strictPort",
            ],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(url, timeout=0.5) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.05)
        else:
            process.terminate()
            pytest.fail("Vite did not start within 15 seconds")

        if process.poll() is not None:
            output.seek(0)
            pytest.fail(f"Vite stopped before acceptance tests:\n{output.read()}")

        try:
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
