"""
Neon Mirrored Waveform Visualizer
----------------------------------
Takes an MP3 file and produces an MP4 video: a glowing neon-cyan waveform
with a mirrored reflection below the center line, moving in real time with
the audio's amplitude (oscilloscope-style, matching a reference design with
pure black background, neon glow, and a 60%-opacity mirrored reflection).

Dependencies:
    pip install librosa numpy matplotlib moviepy

You also need ffmpeg installed on your system:
    - Ubuntu/Debian: sudo apt install ffmpeg
    - macOS (brew):  brew install ffmpeg
    - Windows:       download from ffmpeg.org and add to PATH

Usage:
    python neon_waveform_visualizer.py input.mp3
    python neon_waveform_visualizer.py input.mp3 -o output.mp4 --fps 30 --points 300
"""

import argparse
import os

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy import VideoFileClip, AudioFileClip


def compute_frame_curves(y, sr, fps, n_points=300, window_seconds=0.08):
    """
    For every video frame, extract a short window of audio centered on that
    moment in time, downsample it to n_points, and normalize it to [-1, 1].
    Returns an array of shape (n_frames, n_points).
    """
    duration = len(y) / sr
    n_frames = int(duration * fps)
    window_samples = max(2, int(window_seconds * sr))
    half = window_samples // 2

    # Global normalization so loud/quiet sections stay proportionate
    peak = np.percentile(np.abs(y), 99.5)
    if peak <= 0:
        peak = 1.0

    curves = np.zeros((n_frames, n_points), dtype=np.float32)

    for i in range(n_frames):
        center = int((i / fps) * sr)
        lo = max(0, center - half)
        hi = min(len(y), center + half)
        segment = y[lo:hi]

        if len(segment) < 2:
            continue

        x_src = np.linspace(0.0, 1.0, num=len(segment))
        x_tgt = np.linspace(0.0, 1.0, num=n_points)
        curve = np.interp(x_tgt, x_src, segment)

        curve = curve / peak
        curve = np.tanh(curve * 1.4)  # soft-clip so peaks look natural
        curves[i] = curve

    # Light smoothing across neighboring points to keep the line elegant
    kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)
    for i in range(n_frames):
        curves[i] = np.convolve(curves[i], kernel, mode="same")

    return curves


def make_waveform_video(input_mp3, output_path="waveform.mp4",
                         fps=30, n_points=300, window_seconds=0.08,
                         bar_color="#00E5FF", bg_color="#000000",
                         figsize=(12, 6.75)):
    print(f"Loading audio: {input_mp3}")
    y, sr = librosa.load(input_mp3, sr=None, mono=True)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s, sample rate: {sr}")

    print("Computing per-frame waveform curves...")
    curves = compute_frame_curves(y, sr, fps, n_points=n_points,
                                   window_seconds=window_seconds)
    n_frames = curves.shape[0]
    print(f"Rendering {n_frames} frames at {fps} fps...")

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)

    x = np.linspace(0.0, 1.0, num=n_points)

    # Faint permanent center line (visible even at silence, like the reference)
    ax.plot(x, np.zeros(n_points), color=bar_color, linewidth=1.2, alpha=0.25)

    # Glow effect = several overlapping lines of decreasing width, increasing alpha.
    glow_layers = [(9.0, 0.05), (6.0, 0.10), (3.5, 0.18), (1.6, 0.95)]

    main_lines = [ax.plot([], [], color=bar_color, linewidth=lw, alpha=a,
                           solid_capstyle="round")[0] for lw, a in glow_layers]
    reflect_lines = [ax.plot([], [], color=bar_color, linewidth=lw * 0.9,
                              alpha=a * 0.6, solid_capstyle="round")[0]
                      for lw, a in glow_layers]

    def init():
        for line in main_lines + reflect_lines:
            line.set_data([], [])
        return main_lines + reflect_lines

    def update(frame_idx):
        curve = curves[frame_idx] * 0.9
        for line in main_lines:
            line.set_data(x, curve)
        for line in reflect_lines:
            line.set_data(x, -curve)
        return main_lines + reflect_lines

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 / fps, blit=True
    )

    silent_video_path = "_silent_waveform_temp.mp4"
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    anim.save(silent_video_path, writer=writer, savefig_kwargs={"facecolor": bg_color})
    plt.close(fig)

    print("Merging video with audio...")
    video_clip = VideoFileClip(silent_video_path)
    audio_clip = AudioFileClip(input_mp3)

    final_duration = min(video_clip.duration, audio_clip.duration)
    video_clip = video_clip.subclipped(0, final_duration)
    audio_clip = audio_clip.subclipped(0, final_duration)

    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264",
                                audio_codec="aac", fps=fps)

    video_clip.close()
    audio_clip.close()
    final_clip.close()

    if os.path.exists(silent_video_path):
        os.remove(silent_video_path)

    print(f"Done! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a neon mirrored waveform video from an MP3 file."
    )
    parser.add_argument("input", help="Path to input .mp3 file")
    parser.add_argument("-o", "--output", default="waveform.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--points", type=int, default=300, help="Points along the waveform line")
    parser.add_argument("--window", type=float, default=0.08,
                         help="Seconds of audio shown across the waveform at once")
    parser.add_argument("--color", default="#00E5FF", help="Waveform color (hex)")
    parser.add_argument("--bg", default="#000000", help="Background color (hex)")
    args = parser.parse_args()

    make_waveform_video(
        args.input,
        output_path=args.output,
        fps=args.fps,
        n_points=args.points,
        window_seconds=args.window,
        bar_color=args.color,
        bg_color=args.bg,
    )


if __name__ == "__main__":
    main()