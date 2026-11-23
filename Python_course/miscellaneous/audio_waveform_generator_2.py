"""
Neon Mirrored Waveform Visualizer (Fast / OpenCV version)
-----------------------------------------------------------
Same visual style as the matplotlib version (glowing neon-cyan waveform,
mirrored 60%-opacity reflection, pure black background) but renders with
OpenCV and streams raw frames straight into ffmpeg via a pipe -- no
per-frame matplotlib figure redraw, no temp silent video file, no second
encode pass. This is typically 10-30x faster than the matplotlib version
on long tracks.

Dependencies:
    pip install librosa numpy opencv-python

You also need ffmpeg installed and on your PATH:
    - Ubuntu/Debian: sudo apt install ffmpeg
    - macOS (brew):  brew install ffmpeg
    - Windows:       download from ffmpeg.org and add to PATH

Usage:
    python neon_waveform_fast.py input.mp3
    python neon_waveform_fast.py input.mp3 -o output.mp4 --fps 30 --points 300
"""

import argparse
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import librosa


def compute_frame_curves(y, sr, fps, n_points=300, window_seconds=0.08):
    """
    For every video frame, extract a short window of audio centered on that
    moment in time, downsample it to n_points, and normalize it to [-1, 1].
    Returns an array of shape (n_frames, n_points), float32.
    """
    duration = len(y) / sr
    n_frames = int(duration * fps)
    window_samples = max(2, int(window_seconds * sr))
    half = window_samples // 2

    peak = np.percentile(np.abs(y), 99.5)
    if peak <= 0:
        peak = 1.0

    x_tgt = np.linspace(0.0, 1.0, num=n_points)
    curves = np.zeros((n_frames, n_points), dtype=np.float32)

    for i in range(n_frames):
        center = int((i / fps) * sr)
        lo = max(0, center - half)
        hi = min(len(y), center + half)
        segment = y[lo:hi]
        if len(segment) < 2:
            continue
        x_src = np.linspace(0.0, 1.0, num=len(segment))
        curve = np.interp(x_tgt, x_src, segment)
        curve = curve / peak
        curves[i] = np.tanh(curve * 1.4)

    kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)
    for i in range(n_frames):
        curves[i] = np.convolve(curves[i], kernel, mode="same")

    return curves


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR


def draw_glow_frame(curve, width, height, color_bgr,
                     main_alpha=1.0, reflect_alpha=0.6):
    """
    Draw the main waveform + mirrored reflection with a soft neon glow,
    using a single Gaussian blur pass (fast) instead of many overlaid
    translucent lines.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    cx0, cx1 = int(width * 0.04), int(width * 0.96)
    xs = np.linspace(cx0, cx1, num=len(curve)).astype(np.int32)
    cy = height // 2
    amp = height * 0.42

    main_ys = (cy - curve * amp).astype(np.int32)
    reflect_ys = (cy + curve * amp).astype(np.int32)

    # Faint permanent center line
    cv2.line(canvas, (cx0, cy), (cx1, cy), color_bgr, 1, cv2.LINE_AA)
    canvas = (canvas.astype(np.float32) * 0.25).astype(np.uint8)

    main_pts = np.stack([xs, main_ys], axis=1).reshape(-1, 1, 2)
    reflect_pts = np.stack([xs, reflect_ys], axis=1).reshape(-1, 1, 2)

    main_color = tuple(int(c * main_alpha) for c in color_bgr)
    reflect_color = tuple(int(c * reflect_alpha) for c in color_bgr)

    cv2.polylines(canvas, [reflect_pts], False, reflect_color, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [main_pts], False, main_color, 2, cv2.LINE_AA)

    # Bloom: blur a copy and screen-blend it back onto the crisp lines
    blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=6, sigmaY=6)
    blurred = cv2.addWeighted(blurred, 1.6, blurred, 0, 0)  # brighten the halo
    glow = cv2.max(canvas, blurred)

    # Wider, fainter second blur pass for a softer outer bloom
    wide_blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=16, sigmaY=16)
    wide_blur = cv2.addWeighted(wide_blur, 0.7, wide_blur, 0, 0)
    frame = cv2.max(glow, wide_blur)

    return frame


def make_waveform_video(input_mp3, output_path="waveform.mp4",
                         fps=30, n_points=300, window_seconds=0.08,
                         width=1280, height=720,
                         bar_color="#00E5FF", bg_color="#000000"):
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it and try again.")
        sys.exit(1)

    print(f"Loading audio: {input_mp3}")
    y, sr = librosa.load(input_mp3, sr=None, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s, sample rate: {sr}")

    print("Computing per-frame waveform curves...")
    t0 = time.time()
    curves = compute_frame_curves(y, sr, fps, n_points=n_points,
                                   window_seconds=window_seconds)
    n_frames = curves.shape[0]
    print(f"  done in {time.time() - t0:.1f}s ({n_frames} frames)")

    color_bgr = hex_to_bgr(bar_color)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "-",
        "-i", input_mp3,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        output_path,
    ]

    print(f"Rendering {n_frames} frames at {fps} fps ({width}x{height})...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0 = time.time()
    report_every = max(1, n_frames // 20)
    for i in range(n_frames):
        frame = draw_glow_frame(curves[i], width, height, color_bgr)
        proc.stdin.write(frame.tobytes())
        if i % report_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  frame {i}/{n_frames}  ({rate:.1f} fps render speed)")

    proc.stdin.close()
    proc.wait()

    total_time = time.time() - t0
    print(f"Rendered {n_frames} frames in {total_time:.1f}s "
          f"({n_frames / total_time:.1f} fps average)")
    print(f"Done! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a neon mirrored waveform video from an MP3 file (fast OpenCV renderer)."
    )
    parser.add_argument("input", help="Path to input .mp3 file")
    parser.add_argument("-o", "--output", default="waveform.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--points", type=int, default=300, help="Points along the waveform line")
    parser.add_argument("--window", type=float, default=0.08,
                         help="Seconds of audio shown across the waveform at once")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--color", default="#00E5FF", help="Waveform color (hex)")
    parser.add_argument("--bg", default="#000000", help="Background color (hex, currently always black)")
    args = parser.parse_args()

    make_waveform_video(
        args.input,
        output_path=args.output,
        fps=args.fps,
        n_points=args.points,
        window_seconds=args.window,
        width=args.width,
        height=args.height,
        bar_color=args.color,
        bg_color=args.bg,
    )


if __name__ == "__main__":
    main()