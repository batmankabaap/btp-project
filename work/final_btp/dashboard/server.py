"""Local dashboard: every network side by side on any test case.

    python dashboard/server.py        # loads every model_*.pt in ../artifacts/final

then open http://localhost:8000. Pre-computed results for the sample cases
(scripts/precompute.py) are served instantly; uploaded or edited cases run live.
"""

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from simulation import simulate
from swe.network import NeuralClosure, n_params


def make_handler(closures, artifacts):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(HERE), **kwargs)

        def translate_path(self, path):
            if path.startswith("/artifacts/"):
                return str(artifacts / path[len("/artifacts/"):].split("?")[0])
            return super().translate_path(path)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/models":
                return self.send_json(200, {name: dict(arch=c.arch, params=n_params(c.net))
                                            for name, c in closures.items()})
            if self.path == "/api/samples":
                samples = {p.stem: json.loads(p.read_text())
                           for p in sorted((HERE / "samples").glob("*.json"))}
                return self.send_json(200, samples)
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/simulate":
                return self.send_json(404, dict(error="not found"))
            try:
                length = int(self.headers.get("Content-Length", 0))
                self.send_json(200, simulate(closures, json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, TypeError) as err:
                self.send_json(400, dict(error=str(err)))

        def log_message(self, fmt, *args):
            print(f"  {self.command} {self.path}")

    return Handler


def load_closures(paths):
    """The paper's network first, so it leads the dashboard."""
    paths = sorted(paths, key=lambda p: (Path(p).stem != "model_h128", Path(p).stem))
    return {Path(p).stem.removeprefix("model_"): NeuralClosure.load(p) for p in paths}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default=str(HERE.parents[1] / "artifacts/final"))
    p.add_argument("--models", nargs="*", help="checkpoints (default: model_*.pt in --artifacts)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    closures = load_closures(args.models or Path(args.artifacts).glob("model_*.pt"))
    print("models:", ", ".join(closures))
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(closures, Path(args.artifacts)))
    print(f"dashboard on http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
