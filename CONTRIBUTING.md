# Contributing

## Development setup

```bash
python -m pip install -r requirements.txt
```

`ffmpeg` is required on `PATH` for MP4 export.

The experimental Blender backend can use either:

- `blender` on `PATH`, or
- `CITYPROMISEVID_BLENDER` pointing to a Blender executable

## Before opening a PR

Run:

```bash
python -m compileall ps2ambientvideo
python -m pytest -q
```

## Artifact policy

Do not commit generated renders, debug-frame folders, or private song-derived outputs by default.

If you want to share examples:

- prefer a tiny curated sample set, or
- attach larger renders to GitHub/Codeberg releases instead of the source tree

## Audio files

Tests and examples should use placeholder or redistributable audio only.

Do not hardcode local music-library paths such as `D:\music\...` in code, tests, or docs.
