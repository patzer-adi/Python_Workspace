"""
Neon Mirrored Waveform Visualizer -- Enhanced Edition
--------------------------------------------------------
Same fast OpenCV -> ffmpeg pipe rendering as neon_waveform_fast.py, with
three visual upgrades:

  1. COLOR GRADIENT   - the line sweeps across a cyan -> blue -> violet
                         hue range left to right, with a slow drifting
                         phase so the gradient gently animates over time.
  2. BASS PULSE        - low-frequency energy is tracked per frame with
                         an attack/release envelope; the line gets
                         thicker and brighter on bass hits.
  3. SPARKLE PARTICLES - small glowing dots are placed on the wave's
                         peaks, with a subtle per-particle twinkle.

Dependencies:
    pip install librosa numpy opencv-python scipy

You also need ffmpeg installed and on your PATH:
    - Ubuntu/Debian: sudo apt install ffmpeg
    - macOS (brew):  brew install ffmpeg
    - Windows:       download from ffmpeg.org and add to PATH

Usage:
    python neon_waveform_beautiful.py input.mp3
    python neon_waveform_beautiful.py input.mp3 -o output.mp4 --fps 30
"""

import argparse
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks


def compute_frame_data(y, sr, fps, n_points=280, window_seconds=0.08,
                        bass_cutoff_hz=150.0):
    """
    For every video frame, extract:
      - a windowed, normalized waveform curve (n_points long)
      - a bass-energy scalar (RMS of a low-passed version of the same window)
    Returns (curves, bass_energy) where curves.shape == (n_frames, n_points)
    and bass_energy.shape == (n_frames,), both in roughly [0, 1] after
    normalization.
    """
    duration = len(y) / sr
    n_frames = int(duration * fps)
    window_samples = max(2, int(window_seconds * sr))
    half = window_samples // 2

    peak = np.percentile(np.abs(y), 99.5)
    if peak <= 0:
        peak = 1.0

    # Low-pass the whole track once for bass tracking
    nyquist = sr / 2.0
    b, a = butter(4, min(0.99, bass_cutoff_hz / nyquist), btype="low")
    y_bass = filtfilt(b, a, y)

    x_tgt = np.linspace(0.0, 1.0, num=n_points)
    curves = np.zeros((n_frames, n_points), dtype=np.float32)
    bass_raw = np.zeros(n_frames, dtype=np.float32)

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

        bass_segment = y_bass[lo:hi]
        bass_raw[i] = np.sqrt(np.mean(bass_segment ** 2)) if len(bass_segment) else 0.0

    kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)
    for i in range(n_frames):
        curves[i] = np.convolve(curves[i], kernel, mode="same")

    # Normalize bass energy to [0, 1] using a robust upper percentile
    bass_peak = np.percentile(bass_raw, 97) or 1.0
    bass_norm = np.clip(bass_raw / bass_peak, 0.0, 1.0)

    # Attack/release envelope so pulses feel punchy but not jittery
    env = np.zeros_like(bass_norm)
    release = 0.85
    current = 0.0
    for i, v in enumerate(bass_norm):
        current = v if v > current else current * release
        env[i] = current

    return curves, env


def build_gradient_hues(n_points, hue_min=88, hue_max=150):
    """Precompute a hue value (OpenCV HSV scale 0-179) for each x position."""
    return np.linspace(hue_min, hue_max, num=n_points)


def hues_to_bgr(hues, saturation=255, value=255):
    hsv = np.zeros((len(hues), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.clip(hues, 0, 179).astype(np.uint8)
    hsv[:, 0, 1] = saturation
    hsv[:, 0, 2] = value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr[:, 0, :]  # (n_points, 3) uint8 BGR


def draw_beautiful_frame(curve, bass_level, gradient_hues, phase,
                          width, height, main_alpha=1.0, reflect_alpha=0.6,
                          sparkles=True, twinkle_frame=0):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    cx0, cx1 = int(width * 0.04), int(width * 0.96)
    xs = np.linspace(cx0, cx1, num=len(curve)).astype(np.int32)
    cy = height // 2
    amp = height * 0.42

    main_ys = (cy - curve * amp).astype(np.int32)
    reflect_ys = (cy + curve * amp).astype(np.int32)

    # Drifting hue gradient + slight brightening on bass hits
    hues = (gradient_hues + phase) % 179
    value = int(np.clip(210 + bass_level * 45, 0, 255))
    colors = hues_to_bgr(hues, saturation=235, value=value)

    # Faint permanent center line
    cv2.line(canvas, (cx0, cy), (cx1, cy), (int(colors[:, 0].mean()),
                                             int(colors[:, 1].mean()),
                                             int(colors[:, 2].mean())),
              1, cv2.LINE_AA)
    canvas = (canvas.astype(np.float32) * 0.20).astype(np.uint8)

    thickness = 2 + int(round(bass_level * 3))  # pulses 2 -> 5px

    # Draw per-segment colored polylines (gradient across the wave)
    n = len(curve)
    for i in range(n - 1):
        col = tuple(int(c) for c in colors[i])
        main_col = tuple(int(c * main_alpha) for c in col)
        reflect_col = tuple(int(c * reflect_alpha) for c in col)
        cv2.line(canvas, (xs[i], reflect_ys[i]), (xs[i + 1], reflect_ys[i + 1]),
                  reflect_col, max(1, thickness - 1), cv2.LINE_AA)
    for i in range(n - 1):
        col = tuple(int(c) for c in colors[i])
        main_col = tuple(int(c * main_alpha) for c in col)
        cv2.line(canvas, (xs[i], main_ys[i]), (xs[i + 1], main_ys[i + 1]),
                  main_col, thickness, cv2.LINE_AA)

    # Sparkle particles at wave peaks
    if sparkles:
        peak_idx, props = find_peaks(np.abs(curve), height=0.45, distance=6)
        for p in peak_idx:
            twinkle = 0.6 + 0.4 * np.sin(twinkle_frame * 0.25 + p * 1.7)
            radius = int(2 + 3 * abs(curve[p]) * twinkle)
            col = tuple(int(min(255, c * 1.3)) for c in colors[p])
            py = main_ys[p] if curve[p] >= 0 else reflect_ys[p]
            cv2.circle(canvas, (xs[p], py), max(1, radius), col, -1, cv2.LINE_AA)

    # Bloom passes (glow), bass hits get a slightly hotter/wider bloom
    blur1 = cv2.GaussianBlur(canvas, (0, 0), sigmaX=6, sigmaY=6)
    blur1 = cv2.addWeighted(blur1, 1.5 + bass_level * 0.5, blur1, 0, 0)
    glow = cv2.max(canvas, blur1)

    blur2 = cv2.GaussianBlur(canvas, (0, 0), sigmaX=16 + bass_level * 6, sigmaY=16 + bass_level * 6)
    blur2 = cv2.addWeighted(blur2, 0.7 + bass_level * 0.3, blur2, 0, 0)
    frame = cv2.max(glow, blur2)

    return frame


def make_waveform_video(input_mp3, output_path="waveform.mp4",
                         fps=30, n_points=280, window_seconds=0.08,
                         width=1280, height=720,
                         hue_min=88, hue_max=150, gradient_speed=0.05,
                         sparkles=True):
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it and try again.")
        sys.exit(1)

    print(f"Loading audio: {input_mp3}")
    
    # Handle MP3 and other formats using librosa with audioread backend
    try:
        import audioread
        y, sr = librosa.load(input_mp3, sr=None, mono=True)
    except:
        # Fallback: use pydub to load the audio
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_mp3)
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            sr = audio.frame_rate
            y = samples / (2**15)  # Normalize to [-1, 1]
        except Exception as e:
            print(f"ERROR: Could not load audio file: {e}")
            sys.exit(1)
    
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s, sample rate: {sr}")

    print("Computing per-frame waveform curves + bass envelope...")
    t0 = time.time()
    curves, bass_env = compute_frame_data(y, sr, fps, n_points=n_points,
                                           window_seconds=window_seconds)
    n_frames = curves.shape[0]
    print(f"  done in {time.time() - t0:.1f}s ({n_frames} frames)")

    gradient_hues = build_gradient_hues(n_points, hue_min=hue_min, hue_max=hue_max)

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
        phase = (i * gradient_speed) % 179
        frame = draw_beautiful_frame(
            curves[i], bass_env[i], gradient_hues, phase,
            width, height, sparkles=sparkles, twinkle_frame=i
        )
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
        description="Generate an enhanced neon mirrored waveform video from an MP3 file."
    )
    parser.add_argument("input", help="Path to input .mp3 file")
    parser.add_argument("-o", "--output", default="waveform.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--points", type=int, default=280, help="Points along the waveform line")
    parser.add_argument("--window", type=float, default=0.08,
                         help="Seconds of audio shown across the waveform at once")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--hue-min", type=int, default=88,
                         help="Start hue (OpenCV HSV 0-179; ~88 = cyan)")
    parser.add_argument("--hue-max", type=int, default=150,
                         help="End hue (OpenCV HSV 0-179; ~150 = violet)")
    parser.add_argument("--gradient-speed", type=float, default=0.05,
                         help="How fast the color gradient drifts per frame")
    parser.add_argument("--no-sparkles", action="store_true",
                         help="Disable sparkle particles at wave peaks")
    args = parser.parse_args()

    make_waveform_video(
        args.input,
        output_path=args.output,
        fps=args.fps,
        n_points=args.points,
        window_seconds=args.window,
        width=args.width,
        height=args.height,
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        gradient_speed=args.gradient_speed,
        sparkles=not args.no_sparkles,
    )


if __name__ == "__main__":
    main()