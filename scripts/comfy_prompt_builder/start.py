"""Entry point to launch the Comfy Prompt Builder server."""
from __future__ import annotations

import os
import pathlib
import sys
import threading
import time
import webbrowser

import uvicorn


def _ensure_package_on_path() -> None:
    """Ensure the comfy_prompt_builder package can be imported."""
    package_dir = pathlib.Path(__file__).resolve().parent
    parent_dir = package_dir.parent
    parent_str = str(parent_dir)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)

PORT = int(os.getenv("COMFY_PROMPT_PORT", "8070"))
HOST = os.getenv("COMFY_PROMPT_HOST", "127.0.0.1")
AUTO_OPEN = os.getenv("COMFY_PROMPT_OPEN_BROWSER", "1") not in {"0", "false", "False"}


def _open_browser(url: str) -> None:
    time.sleep(1.5)
    try:  # pragma: no cover - depends on OS/browser availability
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    _ensure_package_on_path()
    display_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    url = f"http://{display_host}:{PORT}"
    if AUTO_OPEN:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()
    uvicorn.run("comfy_prompt_builder.server:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
