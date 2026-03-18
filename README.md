# stereo_berry
two cameras one cup (raspberry pi 5) 

## Live depth + motion

- Runs a rectified stereo depth map + moving-object detection with distance overlay:
	- `python moving_depth.py`

It reads calibration from `calib_auto.npz` and optional tuned overrides from `tuned_params.json`.

## Auto tuning loop (2 hours)

- Runs an autonomous parameter tuning loop and writes `tuned_params.json`:
	- `python auto_dev_loop.py`

- Optional: let a GPT-5.2 vision model suggest parameter tweaks from debug composites:
	- Option A (env var): `export OPENAI_API_KEY=...` then `python auto_dev_loop.py --vision`
	- Option B (no terminal paste): put your key in `.openai_api_key` then:
		- `python auto_dev_loop.py --vision --api-key-file .openai_api_key`

- Quick API test:
	- `python auto_dev_loop.py --vision --ping --api-key-file .openai_api_key`

Artifacts land in `auto_runs/` with per-iteration `composite.png` + metrics.

## Auto code-edit loop (LLM rewrites code)

This is the "full code edits" mode: it asks a GPT-5.2 model to propose file rewrites,
applies them with backups + syntax checks, and can revert automatically if an offline
score gets worse.

- Dry-run (saves proposals only):
	- `python auto_code_loop.py --vision --dry-run --offline`

- Apply edits (writes to your repo, keeps only improvements):
	- `python auto_code_loop.py --vision --apply --offline`

Offline eval uses `data/left/` and `data/right/` image pairs if present.

