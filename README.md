# high-trust-renderer

Offline Python renderer for ambient music videos with audio-reactive dream scenes, blended mode transitions, and ffmpeg MP4 export.

Built to make videos for the JRJ Enterprises album **HIGH-TRUST_SOCIETY高信賴社會** on Aggregatron Records.

Album link: <https://jrjenterprises.bandcamp.com/album/high-trust-society>

A public video index section can be added later with embedded YouTube links for finished uses.

The repo currently preserves two visual systems plus an experimental 3D backend:

- `legacy_plaza` OpenCV renderer: the earlier City Promise / plaza-corridor visual language
- `worlds` OpenCV renderer: the broader world-family grammar
- experimental Blender backend: persistent-process proof and short preview work for cleaner reflective/translucent material studies

Public classification shorthand used in this repo:

- `lofi`: the City Promise preset family and its civic/transit/glass mood arc
- `lowpoly`: the cleaner low-poly graphics treatment
- `retro_clean`: compatibility alias for the same `lowpoly` taxonomy
- the recent Blender material-study direction used for Reehov-style proof work fits under the same `lowpoly` label

## Install

Core install:

```bash
python -m pip install .
```

Full local/dev install matching the current test environment:

```bash
python -m pip install -r requirements.txt
```

`ffmpeg` must also be installed and available on `PATH`.

For the experimental Blender backend, either:

- install `blender` on `PATH`, or
- set `CITYPROMISEVID_BLENDER=/path/to/blender`

## Render

```bash
high-trust-renderer render path/to/input.wav --output path/to/output.mp4 --duration 120 --width 1280 --height 720 --fps 30 --seed 1234 --preset full_dream_cycle --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber
```

## Preset Examples

Depth showcase:

```bash
high-trust-renderer render path/to/sample.wav --output depth_showcase_final.mp4 --duration 60 --width 640 --height 360 --fps 24 --preset depth_showcase --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber --debug-frames debug_depth_final
high-trust-renderer contact-sheet debug_depth_final --output depth_contact_sheet_final.png
```

Lofi / City Promise 30s preview:

```bash
high-trust-renderer render path/to/sample.wav --output city_promise_30s_preview.mp4 --duration 30 --width 640 --height 360 --fps 24 --preset lofi --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber --debug-frames debug_city_promise_30s
high-trust-renderer contact-sheet debug_city_promise_30s --output city_promise_30s_contact_sheet.png
```

Lofi / City Promise 75s preview:

```bash
high-trust-renderer render path/to/sample.wav --output city_promise_final_preview.mp4 --duration 75 --width 640 --height 360 --fps 24 --preset lofi --render-scale 0.5 --render-profile final --aesthetic frutiger_cyber --debug-frames debug_city_promise_final
high-trust-renderer contact-sheet debug_city_promise_final --output city_promise_final_contact_sheet.png
```

Full-track render at 1280x720:

```bash
high-trust-renderer render path/to/song.wav --output song_full.mp4 --width 1280 --height 720 --fps 30 --preset lofi --render-scale 0.67 --render-profile final --aesthetic frutiger_cyber
```

Debug frames and contact sheet workflow:

```bash
high-trust-renderer render path/to/sample.wav --output qa.mp4 --duration 30 --width 640 --height 360 --fps 24 --preset showcase_30s --render-scale 0.5 --render-profile qa --aesthetic lowpoly --debug-frames debug_frames --debug-labels --debug-raw-frames
high-trust-renderer contact-sheet debug_frames --output contact_sheet.png
high-trust-renderer contact-sheet debug_frames_raw --output contact_sheet_raw.png
```

## Render Profiles

- `final`: presentation-ready output with no labels or bbox overlays, cleaner bloom, brighter glass and water, and calmer persistent layers.
- `qa`: inspection-focused output that pairs well with `--debug-frames`, `--debug-labels`, and `--debug-raw-frames`.
- For real song renders, prefer `--render-profile final`.

## Aesthetic Profiles

- `frutiger_cyber`: default. Clean glass-and-water civic futurism with brighter gradients, softer haze, and restrained retro artifacts.
- `lowpoly`: low-poly retro geometry with light nostalgia and minimal post degradation.
- `retro_clean`: compatibility alias for `lowpoly`.
- `crt_dark`: the older darker scanline-heavy look for comparison.

## Notes

- Uses `librosa` when available for beat/onset analysis and section estimates.
- Falls back to a lightweight NumPy-based analysis path when optional audio extras are missing.
- Defaults to internal half-resolution rendering and upscales for a much faster offline workflow.
- Streams raw frames directly into `ffmpeg` for offline encoding.
- `pyproject.toml` describes the core package dependencies; `requirements.txt` is the broader local/test install used in this repo today.
- Public examples in this README intentionally use placeholder paths; the codebase should not depend on private local paths such as `D:\music\...`.
- Generated renders and debug artifacts are intended to stay out of source control by default.
- This repo preserves both OpenCV visual systems (`legacy_plaza` and `worlds`) plus the experimental Blender proof/preview path.
- The older `citypromisevid` command name is still kept as a compatibility alias while the repo shifts to `high-trust-renderer`.

## Author

Jeremy Ray Jewell  
[GitHub](https://github.com/jeremyrayjewell) | [LinkedIn](https://www.linkedin.com/in/jeremyrayjewell)
