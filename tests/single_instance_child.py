"""Test-only CLI process with a blocking model adapter; never opens a real browser."""
import os
from pathlib import Path
import sys
import time

import bloodsmear.cli as cli
from ui_feedback_helpers import make_app


root = Path(sys.argv[1])
def load_service(settings):
    service = make_app(root).state.service
    service.settings = settings
    infer = service.adapter.infer
    def blocking_infer(rgb, options):
        with (root / "model-calls.txt").open("a", encoding="utf-8") as stream:
            stream.write("infer\n")
        (root / "image-started").touch()
        deadline = time.monotonic() + 25
        while not (root / "release-image").exists() and time.monotonic() < deadline:
            time.sleep(0.03)
        return infer(rgb, options)
    service.adapter.infer = blocking_infer
    return service

cli.InferenceService.from_settings = load_service
cli.webbrowser.open = lambda url: (root / "opened-browser.txt").write_text(url, encoding="utf-8")
raise SystemExit(cli.main(["serve", *sys.argv[2:]]))
