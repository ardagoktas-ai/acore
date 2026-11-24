# Publishing - step by step

Both scripts default to a **dry run**: they only print what they would do. Nothing is created until I add `--confirm`. Both create **private** repositories and refuse to upload into an existing public one.

## 1. Check the content first

1. Read `github/README.md`, `github/docs/ARCHITECTURE.md` and `huggingface/README.md`.
2. Fill in the [TODO] items I can already answer (name, license, contact).
3. Make sure no secrets are in `github/` (search for "token", "key", ".env").

## 2. GitHub

1. Create a token: GitHub -> Settings -> Developer settings -> Personal access tokens -> Fine-grained token. Permission: "Administration: Read and write" and "Contents: Read and write" for my repositories.
2. Set it only in the current terminal (never in a file):
   - PowerShell: `$env:GITHUB_TOKEN = "<token>"`
   - bash: `export GITHUB_TOKEN=<token>`
3. Dry run: `python publish/publish_github.py`
4. Check the printed file list.
5. Publish: `python publish/publish_github.py --confirm`
   (another name: `--name <repo-name>`; default is `acore`).
6. Open the repository on GitHub and confirm it shows "Private".

The script uploads one file per commit through the GitHub contents API. Large binary files (checkpoints, compiled libraries) are skipped on purpose.

## 3. Hugging Face

1. Create a token: Hugging Face -> Settings -> Access Tokens -> "Write".
2. `pip install huggingface_hub`
3. Set the token in the current terminal:
   - PowerShell: `$env:HF_TOKEN = "<token>"`
   - bash: `export HF_TOKEN=<token>`
4. Dry run: `python publish/publish_hf.py`
5. Publish: `python publish/publish_hf.py --confirm` (default name `acore-wm`).
6. Open the model page and confirm it shows "Private".

## 4. After publishing

- Clear the tokens: close the terminal, or `Remove-Item Env:GITHUB_TOKEN` / `unset GITHUB_TOKEN` (same for `HF_TOKEN`).
- If a token ever leaks, revoke it immediately in the settings page where I created it.
- Make a repository public only after a final review, by hand in the web settings.
- Upload trained weights to Hugging Face only after the real training run, and update the model card status at the same time.
