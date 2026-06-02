# Signup And API Key

Use this reference when `MOONDREAM_API_KEY` is missing or the user is new to Moondream Cloud/Lens.

## Rules

- Never ask the user to paste a raw key into chat.
- Never print, log, write, or commit the key.
- Scripts must read `os.environ["MOONDREAM_API_KEY"]`.
- If a command needs to check whether the key exists, use a boolean check and do not display the value.

## Setup

1. Get an API key from the Moondream Cloud Console: `https://moondream.ai/`.
2. Export it in the shell that will run the script:

```bash
export MOONDREAM_API_KEY="..."
```

3. Verify silently:

```bash
python3 -c 'import os, sys; sys.exit(0 if os.environ.get("MOONDREAM_API_KEY") else 1)'
```

## When A Key Is Needed

- Cloud inference: required.
- Photon local inference with `local=True`: required.
- Lens finetuning: required.
Lens finetuning runs in Moondream Cloud and spends credits. Confirm cost before creating or resuming a finetune.
