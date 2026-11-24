"""Acore Studio - research prototype. Local HTTP server (standard library only)."""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from acore_studio import Studio  # noqa: E402

STUDIO = None
LOCK = threading.Lock()


def get_studio():
    global STUDIO
    if STUDIO is None:
        STUDIO = Studio()
    return STUDIO


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        return data if isinstance(data, dict) else {}

    def _snapshot(self, s):
        snap = s.snapshot()
        snap["grid"] = s.grid()
        return snap

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            with LOCK:
                s = get_studio()
                if path == "/api/worlds":
                    return self._send(200, {"worlds": s.list_worlds(), "current": s.world})
                if path == "/api/snapshot":
                    return self._send(200, self._snapshot(s))
            return self._send(404, {"error": "not found"})
        except Exception as err:
            return self._send(500, {"error": str(err)})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            data = self._body()
        except Exception:
            return self._send(400, {"error": "invalid JSON"})
        try:
            with LOCK:
                s = get_studio()
                if path == "/api/new":
                    world = data.get("world", s.world)
                    if world not in s.list_worlds():
                        return self._send(400, {"error": f"unknown world {world}"})
                    s.new_world(world, int(data.get("seed", 0)))
                    return self._send(200, self._snapshot(s))
                if path == "/api/step":
                    return self._send(200, s.step())
                if path == "/api/run":
                    k = max(0, min(int(data.get("episodes", 1)), 200))
                    results = s.run_episodes(k)
                    return self._send(200, {"results": results, "snapshot": self._snapshot(s)})
                if path == "/api/ask":
                    return self._send(200, s.ask(str(data.get("text", ""))))
                if path == "/api/reset":
                    s.reset_agent()
                    return self._send(200, self._snapshot(s))
            return self._send(404, {"error": "not found"})
        except (ValueError, TypeError) as err:
            return self._send(400, {"error": str(err)})
        except Exception as err:
            return self._send(500, {"error": str(err)})


def main():
    ap = argparse.ArgumentParser(description="Acore Studio - research prototype")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Acore Studio - research prototype: http://{args.host}:{args.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
