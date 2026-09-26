"""Serve the synthetic virtual-subject sandbox on loopback only."""

from pathlib import Path
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "twin")]
os.environ["TWIN_COMPILE"] = "0"

import torch
torch.set_num_threads(1)

from scenarios import simulate_scenario


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, payload, content_type="application/json"):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        public_assets = {
            "/benchmark": (ROOT / "examples/twin-benchmark-summary.json", "application/json; charset=utf-8"),
            "/benchmark-figure.png": (ROOT / "assets/twin-benchmark.png", "image/png"),
        }
        if self.path in public_assets:
            path, content_type = public_assets[self.path]
            self.send(200, path.read_bytes(), content_type)
            return
        if self.path in ("/", "/index.html"):
            self.send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/calibration":
            path = ROOT / "examples" / "synthetic-calibration.json"
            if not path.is_file():
                self.send(404, {"error": "Synthetic calibration example is not available."})
                return
            self.send(200, path.read_bytes(), "application/json; charset=utf-8")
            return
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/simulate":
            self.send(404, {"error": "Not found"})
            return
        origin = self.headers.get("Origin")
        allowed = (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
        if origin and origin not in allowed:
            self.send(403, {"error": "Local origin required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            self.send(200, simulate_scenario(request))
        except (ValueError, TypeError, json.JSONDecodeError):
            self.send(400, {"error": "Check the scenario inputs."})
        except Exception:
            self.send(500, {"error": "Simulation unavailable."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    print(f"Virtual-patient sandbox: http://127.0.0.1:{args.port}", flush=True)
    # HTTPServer is intentionally single-threaded: one bounded CPU simulation at a time.
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
