# high-trust-renderer

Offline Python renderer for ambient music videos.

It was built to make videos for the JRJ Enterprises album **HIGH-TRUST_SOCIETY&#39640;&#20449;&#36084;&#31038;&#26371;** on Aggregatron Records.

Album link: <https://jrjenterprises.bandcamp.com/album/high-trust-society>

## What it includes

- `legacy_plaza` OpenCV scene grammar for the earlier lofi civic/transit look
- `worlds` OpenCV scene grammar for broader world-based ambient scenes
- experimental Blender backend for reflective/translucent proof rendering
- experimental `softbodies` Panda3D backend for gummy/gelatin motion

Public taxonomy used in this repo:

- `lofi`: the main civic/transit/glass preset family
- `lowpoly`: the cleaner low-poly visual treatment
- `frutiger_cyber`: the brighter glass/water/civic aesthetic profile

## Install

Core install:

```bash
python -m pip install .
```

Local/dev install:

```bash
python -m pip install -r requirements.txt
```

Optional extras:

- `ffmpeg` on `PATH` for MP4 export
- `HIGH_TRUST_RENDERER_BLENDER=/path/to/blender` for the Blender backend, or `blender` on `PATH`
- `python -m pip install .[softbodies]` for the Panda3D softbodies backend

## Basic usage

```bash
high-trust-renderer render path/to/input.wav --output path/to/output.mp4 --duration 120 --width 1280 --height 720 --fps 30 --preset lofi --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber
```

Make a contact sheet from exported debug frames:

```bash
high-trust-renderer contact-sheet debug_frames --output contact_sheet.png
```

## Example commands

OpenCV lofi preview:

```bash
high-trust-renderer render path/to/sample.wav --output lofi_preview.mp4 --duration 30 --width 640 --height 360 --fps 24 --preset lofi --scene-grammar legacy_plaza --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber --debug-frames debug_lofi
```

Worlds preview:

```bash
high-trust-renderer render path/to/sample.wav --output worlds_preview.mp4 --duration 30 --width 640 --height 360 --fps 24 --preset lofi --scene-grammar worlds --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber --debug-frames debug_worlds
```

Blender proof:

```bash
high-trust-renderer render path/to/sample.wav --output blender_proof.mp4 --duration 8 --width 640 --height 360 --fps 1 --preset worlds_material_proof --scene-grammar worlds --render-engine blender --blender-proof-stills --debug-frames debug_blender_proof
```

Softbodies proof:

```bash
high-trust-renderer render path/to/sample.wav --output softbodies.mp4 --duration 12 --width 1280 --height 720 --fps 30 --render-engine softbodies --softbodies-scene floating --softbody-preset stable_medium --softbody-visualization shaded
```

## Render profiles

- `final`: presentation-oriented output
- `qa`: debug-oriented output for frame inspection

## Notes

- `pyproject.toml` defines the package metadata and core dependencies.
- `requirements.txt` is the broader local/test environment used in this repo.
- Public examples intentionally use placeholder paths, not private local audio paths.
- Generated renders, contact sheets, debug folders, and private audio-derived outputs should stay out of source control.

## Author

Jeremy Ray Jewell  
[GitHub](https://github.com/jeremyrayjewell) | [LinkedIn](https://www.linkedin.com/in/jeremyrayjewell)
