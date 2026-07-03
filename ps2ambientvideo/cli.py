from __future__ import annotations

import argparse
from pathlib import Path

from .contact_sheet import render_contact_sheet
from .renderer import render_video


def _parse_aesthetic(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "frutiger_cyber": "frutiger_cyber",
        "crt_dark": "crt_dark",
        "retro_clean": "lowpoly",
        "lowpoly": "lowpoly",
        "ps2_clean": "lowpoly",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "Aesthetic must be one of: frutiger_cyber, crt_dark, lowpoly, retro_clean."
        )
    return aliases[normalized]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="high-trust-renderer",
        description="Render an ambient music video from an audio file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render a video from audio")
    render_parser.add_argument("input", type=Path, help="Input audio file")
    render_parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    render_parser.add_argument("--duration", type=float, default=None, help="Optional duration cap in seconds")
    render_parser.add_argument("--width", type=int, default=1280, help="Output width")
    render_parser.add_argument("--height", type=int, default=720, help="Output height")
    render_parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    render_parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    render_parser.add_argument(
        "--preset",
        default="full_dream_cycle",
        help="Timeline preset to use",
    )
    render_parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="ffmpeg H.264 constant rate factor",
    )
    render_parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="AAC bitrate passed to ffmpeg",
    )
    render_parser.add_argument(
        "--analysis-backend",
        choices=("auto", "numpy", "librosa"),
        default="auto",
        help="Audio analysis backend. 'auto' prefers the fast NumPy path on newer Python versions.",
    )
    render_parser.add_argument(
        "--render-scale",
        type=float,
        default=0.5,
        help="Internal render scale before upscaling to output size.",
    )
    render_parser.add_argument(
        "--bloom-strength",
        type=float,
        default=0.55,
        help="Strength of nostalgic bloom. Lower values preserve edges more strongly.",
    )
    render_parser.add_argument(
        "--fog-strength",
        type=float,
        default=0.65,
        help="Global fog multiplier.",
    )
    render_parser.add_argument(
        "--exposure",
        type=float,
        default=0.94,
        help="Final exposure multiplier.",
    )
    render_parser.add_argument(
        "--retro-jitter",
        dest="ps2_jitter",
        type=float,
        default=0.75,
        help="Subpixel-style vertex jitter amount for a retro hardware wobble.",
    )
    render_parser.add_argument(
        "--ps2-jitter",
        dest="ps2_jitter",
        type=float,
        help=argparse.SUPPRESS,
    )
    render_parser.add_argument(
        "--line-scale",
        type=float,
        default=None,
        help="Optional global line thickness multiplier. Lower values produce thinner outlines.",
    )
    render_parser.add_argument(
        "--render-profile",
        choices=("qa", "final"),
        default=None,
        help="Rendering profile. Defaults to 'final' unless debug flags are passed.",
    )
    render_parser.add_argument(
        "--aesthetic",
        type=_parse_aesthetic,
        default="frutiger_cyber",
        metavar="AESTHETIC",
        help="Visual treatment to apply on top of the scene geometry: frutiger_cyber, crt_dark, lowpoly, or retro_clean.",
    )
    render_parser.add_argument(
        "--scene-grammar",
        choices=("legacy_plaza", "worlds"),
        default="worlds",
        help="Scene-building grammar. 'legacy_plaza' preserves the current platform/plaza/corridor grammar; 'worlds' routes to the newer world-family grammar.",
    )
    render_parser.add_argument(
        "--render-engine",
        choices=("opencv", "blender"),
        default="opencv",
        help="Rendering backend. 'opencv' uses the existing CPU renderer; 'blender' is an experimental offline 3D proof backend.",
    )
    render_parser.add_argument(
        "--debug-frames",
        type=Path,
        default=None,
        help="Optional directory to export QA frames.",
    )
    render_parser.add_argument(
        "--debug-labels",
        action="store_true",
        help="Draw small labels with time, mode, and weights on frames.",
    )
    render_parser.add_argument(
        "--debug-raw-frames",
        action="store_true",
        help="Also export pre-postprocessing QA frames into a sibling *_raw folder.",
    )
    render_parser.add_argument(
        "--blender-proof-stills",
        action="store_true",
        help="Blender backend only: render a tiny fixed still-frame proof into the debug-frames directory instead of a full animation/MP4.",
    )
    render_parser.add_argument(
        "--blender-smoke-scene",
        choices=("cube", "materials"),
        default=None,
        help="Blender backend only: bypass audio/timeline/world construction and render a tiny one-frame smoke-test scene into the debug-frames directory.",
    )
    render_parser.add_argument(
        "--blender-quality",
        choices=("smoke", "proof", "final"),
        default="proof",
        help="Blender backend only: quality tier. 'smoke' is the fastest clean diagnostic path, 'proof' stays realtime-friendly with slightly richer materials, and 'final' is reserved for later.",
    )
    render_parser.add_argument(
        "--blender-diagnostic-engine",
        choices=("workbench", "eevee", "opengl"),
        default=None,
        help="Blender backend only: cube diagnostic engine override for one-frame smoke testing.",
    )

    sheet_parser = subparsers.add_parser("contact-sheet", help="Create a PNG contact sheet from debug frames")
    sheet_parser.add_argument("input_dir", type=Path, help="Directory containing exported debug frames")
    sheet_parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    sheet_parser.add_argument("--columns", type=int, default=4, help="Number of columns in the sheet")
    sheet_parser.add_argument("--padding", type=int, default=16, help="Padding between frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "render":
        render_video(
            input_path=args.input,
            output_path=args.output,
            duration=args.duration,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
            preset=args.preset,
            crf=args.crf,
            audio_bitrate=args.audio_bitrate,
            analysis_backend=args.analysis_backend,
            render_scale=args.render_scale,
            bloom_strength=args.bloom_strength,
            fog_strength=args.fog_strength,
            exposure=args.exposure,
            ps2_jitter=args.ps2_jitter,
            line_scale=args.line_scale,
            render_profile=args.render_profile,
            aesthetic=args.aesthetic,
            scene_grammar=args.scene_grammar,
            render_engine=args.render_engine,
            debug_frames_dir=args.debug_frames,
            debug_labels=args.debug_labels,
            debug_raw_frames=args.debug_raw_frames,
            blender_proof_stills=args.blender_proof_stills,
            blender_smoke_scene=args.blender_smoke_scene,
            blender_quality=args.blender_quality,
            blender_diagnostic_engine=args.blender_diagnostic_engine,
        )
        return 0
    if args.command == "contact-sheet":
        render_contact_sheet(
            input_dir=args.input_dir,
            output_path=args.output,
            columns=args.columns,
            padding=args.padding,
        )
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2
