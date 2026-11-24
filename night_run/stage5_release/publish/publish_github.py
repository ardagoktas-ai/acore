"""Create a PRIVATE GitHub repository and upload the contents of ../github.

Default is a dry run that only prints what would happen.
Real actions need --confirm. The token is read from the GITHUB_TOKEN
environment variable and is never printed or stored.
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "github"
SKIP_DIRS = {".git", "__pycache__", "build", "checkpoints", ".venv", "venv"}
SKIP_SUFFIXES = {".pt", ".safetensors", ".dll", ".so", ".dylib", ".pyc", ".vcd"}


def collect_files(source):
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        files.append(rel)
    return files


def request(method, url, token, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            payload = {"message": raw.decode("utf-8", "replace")}
        return err.code, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="acore", help="repository name")
    parser.add_argument("--description", default="My vertically integrated reasoning stack: Acore, NPU, driver, world model")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--confirm", action="store_true", help="really create and upload")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        sys.exit("Source folder not found: %s" % SOURCE)
    files = collect_files(SOURCE)

    print("Source folder : %s" % SOURCE)
    print("Repository    : %s (PRIVATE)" % args.name)
    print("Files         : %d" % len(files))
    for rel in files:
        print("  " + rel.as_posix())

    if not args.confirm:
        print("\nDRY RUN: nothing was created or uploaded. Add --confirm to publish.")
        return

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set.")

    status, user = request("GET", API + "/user", token)
    if status != 200:
        sys.exit("Authentication failed (HTTP %d): %s" % (status, user.get("message")))
    owner = user["login"]

    status, body = request("POST", API + "/user/repos", token, {
        "name": args.name,
        "description": args.description,
        "private": True,
        "auto_init": False,
    })
    if status == 201:
        print("Created private repository %s/%s" % (owner, args.name))
    elif status == 422:
        status, repo = request("GET", "%s/repos/%s/%s" % (API, owner, args.name), token)
        if status != 200:
            sys.exit("Repository exists but cannot be read (HTTP %d)." % status)
        if not repo.get("private"):
            sys.exit("Repository %s/%s exists and is PUBLIC. Refusing to upload." % (owner, args.name))
        print("Repository %s/%s already exists (private); updating files." % (owner, args.name))
    else:
        sys.exit("Could not create repository (HTTP %d): %s" % (status, body.get("message")))

    failures = 0
    for rel in files:
        path = rel.as_posix()
        url = "%s/repos/%s/%s/contents/%s" % (API, owner, args.name, path)
        content = base64.b64encode((SOURCE / rel).read_bytes()).decode("ascii")
        payload = {"message": "Add " + path, "content": content, "branch": args.branch}
        status, existing = request("GET", url + "?ref=" + args.branch, token)
        if status == 200 and isinstance(existing, dict) and "sha" in existing:
            payload["sha"] = existing["sha"]
            payload["message"] = "Update " + path
        status, body = request("PUT", url, token, payload)
        if status in (200, 201):
            print("  uploaded " + path)
        else:
            failures += 1
            print("  FAILED   %s (HTTP %d): %s" % (path, status, body.get("message")))

    print("\nDone: %d files, %d failures." % (len(files), failures))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
