"""Create a PRIVATE Hugging Face model repository and upload ../huggingface.

Default is a dry run that only prints what would happen.
Real actions need --confirm. The token is read from the HF_TOKEN
environment variable and is never printed or stored.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "huggingface"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="acore-wm", help="model repository name")
    parser.add_argument("--confirm", action="store_true", help="really create and upload")
    args = parser.parse_args()

    if not SOURCE.is_dir():
        sys.exit("Source folder not found: %s" % SOURCE)
    files = sorted(p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*") if p.is_file())

    print("Source folder : %s" % SOURCE)
    print("Repository    : <my-username>/%s (model, PRIVATE)" % args.name)
    print("Files         : %d" % len(files))
    for rel in files:
        print("  " + rel)

    if not args.confirm:
        print("\nDRY RUN: nothing was created or uploaded. Add --confirm to publish.")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set.")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = "%s/%s" % (username, args.name)

    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(repo_id)
    if not info.private:
        sys.exit("Repository %s exists and is PUBLIC. Refusing to upload." % repo_id)

    api.upload_folder(
        folder_path=str(SOURCE),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model card",
        ignore_patterns=["*.pt", "__pycache__/*"],
    )
    print("Uploaded %d files to private repository %s" % (len(files), repo_id))


if __name__ == "__main__":
    main()
