# Publishing Checklist

## Safe to commit

- `ps2ambientvideo/`
- `tests/`
- `scripts/`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- CI/workflow files
- contributor and publishing docs

## Generated artifacts that should stay ignored

- rendered videos such as `*.mp4`
- contact sheets such as `*_contact_sheet.png`
- QA/export folders such as `debug_*`
- temporary proof outputs such as `tmp_*`
- local logs like `render_proof_*.txt`
- Python caches and test caches

## First public commit include list

- `LICENSE`
- `README.md`
- `CONTRIBUTING.md`
- `PUBLISHING_CHECKLIST.md`
- `.gitignore`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `requirements.txt`
- `ps2ambientvideo/`
- `tests/`
- `scripts/` for lightweight repo utilities such as QA and Blender diagnostics
- any small non-private helper text/config files needed by the source tree

## Do not commit

- `*.mp4`
- `*_contact_sheet.png`
- `contact_sheet*.png`
- `debug_*/`
- `tmp_*/`
- Blender proof logs, generated scripts, and metadata JSON kept under debug/output folders
- private song-derived renders or previews
- private local audio files or paths from personal music folders
- `smoke.wav` unless you explicitly decide to publish it as a redistributable sample
- local caches such as `__pycache__/` and `.pytest_cache/`
- any machine-specific temp or export files

## Private media warning

Do not commit:

- private song files from local music folders
- renders derived from private or unreleased tracks unless you intend to publish them
- debug folders that contain absolute local paths or private filenames in logs/JSON

## Suggested release-artifact policy

- keep the source repository focused on code, tests, and docs
- publish large renders, previews, and galleries as GitHub/Codeberg Releases or external storage
- if source-tree examples are needed, keep them very small and clearly redistributable

## Recommended first public commit contents

- source code
- tests
- README
- dependency files
- CI workflow
- `.gitignore`
- chosen `LICENSE`
- at most one tiny bundled sample input, if redistributable

## Git health

Before the first public push, confirm the repository has a valid `.git` directory and `git status` works.

If the current `.git` metadata is broken or empty:

- restore the real git metadata, or
- initialize a fresh repo with `git init`, after you are ready

Do not delete or replace `.git` blindly without confirming which history you want to preserve.

## Codeberg note

The GitHub Actions workflow is a good baseline for local/GitHub checks.

If you want Forgejo/Codeberg CI later, mirror the same steps:

- install Python
- install requirements
- `python -m compileall ps2ambientvideo tests`
- `python -m pytest -q`
