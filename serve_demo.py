"""Serve the demo and train user-drawn targets through a local API."""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from livingpixels.nca import NCA, NCAConfig
from livingpixels.training import PoolConfig, fit_pool
from livingpixels.web import load_checkpoint_payload, model_payload


ROOT = Path.cwd()


def json_response(handler: SimpleHTTPRequestHandler, payload: dict[str, object], status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class LivingPixelsHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/model":
            user_checkpoint = ROOT / "artifacts" / "user-model.npz"
            try:
                if user_checkpoint.exists():
                    payload = load_checkpoint_payload(user_checkpoint)
                else:
                    payload = json.loads((ROOT / "demo" / "model.json").read_text(encoding="utf-8"))
                json_response(self, payload)
            except Exception as error:  # pragma: no cover - surfaced in the browser
                json_response(self, {"error": str(error)}, status=500)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/train":
            json_response(self, {"error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ValueError("request is too large")
            request = json.loads(self.rfile.read(length))
            size = int(request.get("size", 32))
            if size != 32:
                raise ValueError("the interactive MVP expects a 32x32 target")
            target = np.asarray(request["target"], dtype=np.float32)
            if target.size != 4 * size * size:
                raise ValueError("target must contain exactly 4*32*32 RGBA values")
            target = np.clip(target.reshape(4, size, size), 0.0, 1.0)
            if float(target[3].sum()) < 1.0:
                raise ValueError("draw at least one non-transparent pixel")
            iterations = min(2000, max(1, int(request.get("iterations", 300))))
            steps = min(96, max(8, int(request.get("steps", 32))))
            pool_size = min(512, max(8, int(request.get("poolSize", 64))))
            batch_size = min(16, max(1, int(request.get("batchSize", 4))))
            if batch_size > pool_size:
                batch_size = pool_size

            model = NCA(NCAConfig(), seed=7)
            history = fit_pool(model, target, iterations=iterations, steps=steps, pool_config=PoolConfig(pool_size=pool_size, batch_size=batch_size), log_every=0)
            artifacts = ROOT / "artifacts"
            artifacts.mkdir(exist_ok=True)
            model.save(artifacts / "user-model.npz")
            np.save(artifacts / "user-target.npy", target)
            json_response(self, {"model": model_payload(model), "loss": history[-1], "iterations": iterations})
        except Exception as error:  # pragma: no cover - surfaced in the browser
            json_response(self, {"error": str(error)}, status=400)


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000/demo/")
    ThreadingHTTPServer(("127.0.0.1", 8000), LivingPixelsHandler).serve_forever()
