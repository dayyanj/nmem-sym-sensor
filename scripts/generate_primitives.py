#!/usr/bin/env python3
"""
Generate clean primitive training videos — shapes and colors with TTS audio.

Creates Rosetta Stone-style flashcards:
- Simple shape/color on screen for 3 seconds
- Clear TTS voice says the word
- Multiple speakers for speaker-invariant learning
- No music, no narration filler, no effects

Usage:
    python scripts/generate_primitives.py --output videos/primitives/ [--voices 4]
"""
import argparse
import asyncio
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path

import edge_tts
from PIL import Image, ImageDraw

# ── Curriculum ───────────────────────────────────────────

COLORS = {
    # name: list of (R, G, B) shades
    "red": [
        (220, 20, 20), (180, 30, 30), (255, 60, 60),
        (150, 10, 10), (255, 100, 100),
    ],
    "blue": [
        (30, 60, 220), (20, 20, 180), (70, 100, 255),
        (10, 10, 150), (100, 140, 255),
    ],
    "green": [
        (30, 180, 30), (20, 140, 20), (60, 220, 60),
        (10, 100, 10), (100, 255, 100),
    ],
    "yellow": [
        (240, 220, 20), (200, 180, 10), (255, 255, 60),
        (180, 160, 10), (255, 240, 100),
    ],
    "orange": [
        (240, 140, 20), (220, 120, 10), (255, 170, 50),
        (200, 100, 10), (255, 180, 80),
    ],
    "purple": [
        (140, 30, 200), (120, 20, 180), (170, 60, 230),
        (100, 10, 150), (190, 100, 255),
    ],
    "pink": [
        (240, 100, 160), (220, 80, 140), (255, 140, 190),
        (200, 60, 120), (255, 160, 200),
    ],
    "brown": [
        (140, 80, 30), (120, 60, 20), (170, 100, 50),
        (100, 50, 10), (180, 120, 70),
    ],
    "black": [
        (20, 20, 20), (40, 40, 40), (10, 10, 10),
        (30, 30, 35), (50, 50, 50),
    ],
    "white": [
        (240, 240, 240), (250, 250, 250), (230, 230, 230),
        (245, 245, 250), (235, 235, 240),
    ],
    "gray": [
        (128, 128, 128), (100, 100, 100), (160, 160, 160),
        (80, 80, 80), (180, 180, 180),
    ],
    "cyan": [
        (20, 200, 220), (10, 180, 200), (60, 230, 240),
        (10, 160, 180), (100, 240, 250),
    ],
}

SHAPES = [
    "circle", "square", "triangle", "rectangle",
    "pentagon", "hexagon", "oval", "diamond",
    "star", "ring", "semicircle", "cross",
]

# Voices — diverse speakers for invariance
VOICES = [
    "en-US-JennyNeural",       # Female US
    "en-US-GuyNeural",         # Male US
    "en-GB-SoniaNeural",       # Female UK
    "en-GB-RyanNeural",        # Male UK
    "en-AU-NatashaNeural",     # Female AU
    "en-IN-PrabhatNeural",     # Male IN
    "en-CA-ClaraNeural",       # Female CA
    "en-IE-ConnorNeural",      # Male IE
]

CANVAS_SIZE = 640
BG_COLORS = [(30, 30, 30), (240, 240, 240), (50, 50, 70), (245, 240, 230)]


# ── Shape Drawing ────────────────────────────────────────

def draw_shape(shape: str, color: tuple, bg_color: tuple, size: int = CANVAS_SIZE, rotation: int = 0) -> Image.Image:
    """Draw a shape on a solid background, optionally rotated."""
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = int(size * 0.3)  # radius

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)

    elif shape == "rectangle":
        draw.rectangle([cx - int(r * 1.4), cy - int(r * 0.7),
                        cx + int(r * 1.4), cy + int(r * 0.7)], fill=color)

    elif shape == "triangle":
        pts = [
            (cx, cy - r),
            (cx - int(r * 0.87), cy + r // 2),
            (cx + int(r * 0.87), cy + r // 2),
        ]
        draw.polygon(pts, fill=color)

    elif shape == "pentagon":
        pts = _regular_polygon(cx, cy, r, 5, rotation=-90)
        draw.polygon(pts, fill=color)

    elif shape == "hexagon":
        pts = _regular_polygon(cx, cy, r, 6, rotation=0)
        draw.polygon(pts, fill=color)

    elif shape == "oval":
        draw.ellipse([cx - int(r * 1.3), cy - int(r * 0.7),
                      cx + int(r * 1.3), cy + int(r * 0.7)], fill=color)

    elif shape == "diamond":
        pts = [(cx, cy - r), (cx + int(r * 0.7), cy),
               (cx, cy + r), (cx - int(r * 0.7), cy)]
        draw.polygon(pts, fill=color)

    elif shape == "star":
        pts = _star_polygon(cx, cy, r, r // 2, 5)
        draw.polygon(pts, fill=color)

    elif shape == "ring":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        inner = int(r * 0.55)
        draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=bg_color)

    elif shape == "semicircle":
        draw.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 0, fill=color)

    elif shape == "cross":
        w = r // 3
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=color)
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=color)

    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, fillcolor=bg_color)

    return img


def draw_outline_shape(
    shape: str, color: tuple, bg_color: tuple,
    size: int = CANVAS_SIZE, rotation: int = 0, width: int = 6,
) -> Image.Image:
    """Draw an outline-only shape (no fill) with specified border width."""
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = int(size * 0.3)

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)

    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)

    elif shape == "rectangle":
        draw.rectangle([cx - int(r * 1.4), cy - int(r * 0.7),
                        cx + int(r * 1.4), cy + int(r * 0.7)], outline=color, width=width)

    elif shape == "triangle":
        pts = [
            (cx, cy - r),
            (cx - int(r * 0.87), cy + r // 2),
            (cx + int(r * 0.87), cy + r // 2),
        ]
        draw.polygon(pts, outline=color, width=width)

    elif shape == "pentagon":
        pts = _regular_polygon(cx, cy, r, 5, rotation=-90)
        draw.polygon(pts, outline=color, width=width)

    elif shape == "hexagon":
        pts = _regular_polygon(cx, cy, r, 6, rotation=0)
        draw.polygon(pts, outline=color, width=width)

    elif shape == "oval":
        draw.ellipse([cx - int(r * 1.3), cy - int(r * 0.7),
                      cx + int(r * 1.3), cy + int(r * 0.7)], outline=color, width=width)

    elif shape == "diamond":
        pts = [(cx, cy - r), (cx + int(r * 0.7), cy),
               (cx, cy + r), (cx - int(r * 0.7), cy)]
        draw.polygon(pts, outline=color, width=width)

    elif shape == "star":
        pts = _star_polygon(cx, cy, r, r // 2, 5)
        draw.polygon(pts, outline=color, width=width)

    elif shape == "semicircle":
        draw.arc([cx - r, cy - r, cx + r, cy + r], 180, 0, fill=color, width=width)

    elif shape == "cross":
        w = r // 3
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], outline=color, width=width)
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], outline=color, width=width)

    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, fillcolor=bg_color)

    return img


def draw_color_swatch(color: tuple, bg_color: tuple, size: int = CANVAS_SIZE) -> Image.Image:
    """Draw a solid color swatch filling most of the canvas."""
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    margin = size // 6
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=size // 20,
        fill=color,
    )
    return img


def _regular_polygon(cx, cy, r, n, rotation=0):
    """Generate vertices of a regular n-gon."""
    pts = []
    for i in range(n):
        angle = math.radians(rotation + 360 * i / n)
        pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
    return pts


def _star_polygon(cx, cy, r_outer, r_inner, n):
    """Generate vertices of a star polygon."""
    pts = []
    for i in range(2 * n):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.radians(-90 + 360 * i / (2 * n))
        pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
    return pts


# ── TTS Audio ────────────────────────────────────────────

async def generate_tts(text: str, voice: str, output_path: str):
    """Generate TTS audio using edge-tts."""
    communicate = edge_tts.Communicate(text, voice, rate="-10%")
    await communicate.save(output_path)


# ── Video Assembly ───────────────────────────────────────

def _build_repeated_audio(audio_path: str, repetitions: int = 3, gap_seconds: float = 1.0) -> tuple[str, list[str]]:
    """Build repeated audio track. Returns (output_path, temp_files_to_clean)."""
    temps = []

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
        repeated_audio = af.name
        temps.append(repeated_audio)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as sf:
        silence_path = sf.name
        temps.append(silence_path)

    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(gap_seconds),
        "-c:a", "libmp3lame", "-b:a", "128k",
        "-loglevel", "error",
        silence_path,
    ], check=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as lf:
        concat_list = lf.name
        temps.append(concat_list)
        for i in range(repetitions):
            lf.write(f"file '{audio_path}'\n")
            if i < repetitions - 1:
                lf.write(f"file '{silence_path}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:a", "libmp3lame", "-b:a", "128k",
        "-loglevel", "error",
        repeated_audio,
    ], check=True)

    return repeated_audio, temps


def create_video(
    shape: str,
    color: tuple,
    bg_color: tuple,
    audio_path: str,
    output_path: str,
    repetitions: int = 3,
    gap_seconds: float = 1.0,
    fps: int = 15,
    is_swatch: bool = False,
):
    """Create an animated video: shape drifts and rotates gently.

    The shape moves in a slow figure-8 path and rotates continuously,
    never leaving the frame. Each frame is unique, giving the learner
    natural visual variety while the TTS audio labels the concept.
    """
    repeated_audio, temps = _build_repeated_audio(audio_path, repetitions, gap_seconds)

    # Get audio duration to determine frame count
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", repeated_audio],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except (ValueError, AttributeError):
        duration = 7.0

    total_frames = int(duration * fps)
    frame_dir = tempfile.mkdtemp()

    try:
        size = CANVAS_SIZE
        max_drift = int(size * 0.12)  # max pixels to drift from center
        rotation_speed = 40  # degrees per second

        for f_idx in range(total_frames):
            t = f_idx / fps

            # Figure-8 drift path (Lissajous)
            drift_x = int(max_drift * math.sin(t * 0.8))
            drift_y = int(max_drift * math.sin(t * 0.6) * math.cos(t * 0.4))

            # Gentle rotation
            rot = int(t * rotation_speed) % 360

            if is_swatch:
                # Color swatches: drift only, no rotation
                img = Image.new("RGB", (size, size), bg_color)
                draw = ImageDraw.Draw(img)
                margin = size // 6
                draw.rounded_rectangle(
                    [margin + drift_x, margin + drift_y,
                     size - margin + drift_x, size - margin + drift_y],
                    radius=size // 20,
                    fill=color,
                )
            else:
                # Shapes: draw centered then shift + rotate
                img = draw_shape(shape, color, bg_color, size=size, rotation=rot)
                # Apply drift by shifting the image
                shifted = Image.new("RGB", (size, size), bg_color)
                shifted.paste(img, (drift_x, drift_y))
                img = shifted

            img.save(os.path.join(frame_dir, f"frame_{f_idx:05d}.png"))

        # Combine frames + audio into video
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "frame_%05d.png"),
            "-i", repeated_audio,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-loglevel", "error",
            output_path,
        ], check=True)
    finally:
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        for f in temps:
            try:
                os.unlink(f)
            except OSError:
                pass


def create_outline_video(
    shape: str,
    color: tuple,
    bg_color: tuple,
    audio_path: str,
    output_path: str,
    repetitions: int = 3,
    gap_seconds: float = 1.0,
    fps: int = 15,
):
    """Create an animated outline video: border thickness pulses, color shifts.

    The outline width oscillates between thin (2px) and thick (12px),
    the border color shifts through related hues, and the shape drifts
    and rotates — all while the TTS says the shape name.
    """
    repeated_audio, temps = _build_repeated_audio(audio_path, repetitions, gap_seconds)

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", repeated_audio],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except (ValueError, AttributeError):
        duration = 7.0

    total_frames = int(duration * fps)
    frame_dir = tempfile.mkdtemp()

    try:
        size = CANVAS_SIZE
        max_drift = int(size * 0.10)
        r, g, b = color

        for f_idx in range(total_frames):
            t = f_idx / fps

            # Pulsing border width: oscillates between 2 and 12
            width = int(3 + 5 * (1 + math.sin(t * 2.5)))

            # Color shift: gently cycle hue around the base color
            shift = int(40 * math.sin(t * 1.2))
            frame_color = (
                max(0, min(255, r + shift)),
                max(0, min(255, g - shift // 2)),
                max(0, min(255, b + shift // 3)),
            )

            # Drift and rotation
            drift_x = int(max_drift * math.sin(t * 0.7))
            drift_y = int(max_drift * math.sin(t * 0.5) * math.cos(t * 0.3))
            rot = int(t * 30) % 360

            img = draw_outline_shape(shape, frame_color, bg_color,
                                     size=size, rotation=rot, width=width)

            # Apply drift
            shifted = Image.new("RGB", (size, size), bg_color)
            shifted.paste(img, (drift_x, drift_y))

            shifted.save(os.path.join(frame_dir, f"frame_{f_idx:05d}.png"))

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "frame_%05d.png"),
            "-i", repeated_audio,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-loglevel", "error",
            output_path,
        ], check=True)
    finally:
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        for f in temps:
            try:
                os.unlink(f)
            except OSError:
                pass


# ── Main Generator ───────────────────────────────────────

async def generate_all(output_dir: str, num_voices: int = 4):
    """Generate the full primitive training set."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    voices = VOICES[:num_voices]
    total = 0

    # 1. Color swatches — each color × each shade × each voice
    print(f"Generating color swatches ({len(COLORS)} colors, {num_voices} voices)...")
    color_dir = out / "colors"
    color_dir.mkdir(exist_ok=True)

    for color_name, shades in COLORS.items():
        for shade_idx, rgb in enumerate(shades):
            bg = random.choice(BG_COLORS)
            # Avoid same bg as swatch
            while _color_distance(bg, rgb) < 80:
                bg = random.choice(BG_COLORS)

            voice = voices[shade_idx % len(voices)]

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
                audio_path = af.name

            await generate_tts(color_name, voice, audio_path)

            video_name = f"{color_name}_shade{shade_idx}_{voice.split('-')[1].lower()}.mp4"
            video_path = str(color_dir / video_name)
            create_video("circle", rgb, bg, audio_path, video_path, is_swatch=True)
            os.unlink(audio_path)
            total += 1

        print(f"  {color_name}: {len(shades)} shades")

    # 2. Shapes — each shape × multiple colors × each voice
    print(f"\nGenerating shapes ({len(SHAPES)} shapes, {num_voices} voices)...")
    shape_dir = out / "shapes"
    shape_dir.mkdir(exist_ok=True)

    shape_colors = [
        (220, 20, 20), (30, 60, 220), (30, 180, 30),
        (240, 220, 20), (240, 140, 20), (140, 30, 200),
    ]

    for shape_name in SHAPES:
        for color_idx, color in enumerate(shape_colors[:3]):
            bg = random.choice(BG_COLORS)
            while _color_distance(bg, color) < 80:
                bg = random.choice(BG_COLORS)

            voice = voices[color_idx % len(voices)]

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
                audio_path = af.name

            await generate_tts(shape_name, voice, audio_path)

            video_name = f"{shape_name}_c{color_idx}_{voice.split('-')[1].lower()}.mp4"
            video_path = str(shape_dir / video_name)
            create_video(shape_name, color, bg, audio_path, video_path)
            os.unlink(audio_path)
            total += 1

        print(f"  {shape_name}: 3 color variants")

    # 3. Colored shapes — "red circle", "blue triangle" etc.
    # Teaches compound concept binding
    print("\nGenerating colored shapes (compound concepts)...")
    compound_dir = out / "compound"
    compound_dir.mkdir(exist_ok=True)

    compound_pairs = [
        ("red", "circle"), ("blue", "square"), ("green", "triangle"),
        ("yellow", "star"), ("orange", "hexagon"), ("purple", "diamond"),
        ("red", "triangle"), ("blue", "circle"), ("green", "square"),
        ("yellow", "pentagon"), ("pink", "oval"), ("cyan", "ring"),
        ("red", "star"), ("blue", "hexagon"), ("green", "pentagon"),
        ("brown", "rectangle"), ("orange", "circle"), ("purple", "triangle"),
    ]

    for color_name, shape_name in compound_pairs:
        rgb = random.choice(COLORS[color_name])
        bg = random.choice(BG_COLORS)
        while _color_distance(bg, rgb) < 80:
            bg = random.choice(BG_COLORS)

        voice = random.choice(voices)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
            audio_path = af.name

        # Say both the color and shape
        await generate_tts(f"{color_name} {shape_name}", voice, audio_path)

        video_name = f"{color_name}_{shape_name}_{voice.split('-')[1].lower()}.mp4"
        video_path = str(compound_dir / video_name)
        create_video(shape_name, rgb, bg, audio_path, video_path)
        os.unlink(audio_path)
        total += 1

    print(f"  {len(compound_pairs)} compound pairs")

    # 4. Naming pattern videos — "this is a [shape]", "this is [color]"
    # Teaches the system to detect naming patterns from speech
    print("\nGenerating naming patterns...")
    naming_dir = out / "naming"
    naming_dir.mkdir(exist_ok=True)

    naming_templates = [
        "this is a {}",
        "this is {}",
        "look, a {}",
        "{}",
    ]

    # Shapes with naming
    for shape_name in SHAPES:
        color = random.choice(shape_colors)
        bg = random.choice(BG_COLORS)
        while _color_distance(bg, color) < 80:
            bg = random.choice(BG_COLORS)

        template = random.choice(naming_templates)
        phrase = template.format(shape_name)
        voice = random.choice(voices)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
            audio_path = af.name

        await generate_tts(phrase, voice, audio_path)

        video_name = f"naming_{shape_name}_{voice.split('-')[1].lower()}.mp4"
        video_path = str(naming_dir / video_name)
        create_video(shape_name, color, bg, audio_path, video_path)
        os.unlink(audio_path)
        total += 1

    # Colors with naming
    for color_name in COLORS:
        rgb = random.choice(COLORS[color_name])
        bg = random.choice(BG_COLORS)
        while _color_distance(bg, rgb) < 80:
            bg = random.choice(BG_COLORS)

        template = random.choice(naming_templates)
        phrase = template.format(color_name)
        voice = random.choice(voices)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
            audio_path = af.name

        await generate_tts(phrase, voice, audio_path)

        video_name = f"naming_{color_name}_{voice.split('-')[1].lower()}.mp4"
        video_path = str(naming_dir / video_name)
        create_video("circle", rgb, bg, audio_path, video_path, is_swatch=True)
        os.unlink(audio_path)
        total += 1

    # 5. Rotated shapes — same shape at different angles
    # Teaches rotation invariance: a tilted triangle is still a triangle
    print("\nGenerating rotated shapes...")
    rotated_dir = out / "rotated"
    rotated_dir.mkdir(exist_ok=True)

    # Shapes that meaningfully rotate (circle/ring look the same rotated)
    rotatable = ["square", "triangle", "rectangle", "pentagon", "hexagon",
                 "oval", "diamond", "star", "semicircle", "cross"]
    rotations = [15, 30, 45, 60, 90, 120, 150, 200, 270, 315]
    rotated_count = 0

    for shape_name in rotatable:
        # Pick 3 random rotations per shape
        angles = random.sample(rotations, 3)
        for angle in angles:
            color = random.choice(shape_colors)
            bg = random.choice(BG_COLORS)
            while _color_distance(bg, color) < 80:
                bg = random.choice(BG_COLORS)

            voice = random.choice(voices)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
                audio_path = af.name

            await generate_tts(shape_name, voice, audio_path)

            video_name = f"{shape_name}_rot{angle}_{voice.split('-')[1].lower()}.mp4"
            video_path = str(rotated_dir / video_name)
            create_video(shape_name, color, bg, audio_path, video_path)
            os.unlink(audio_path)
            total += 1
            rotated_count += 1

    print(f"  {rotated_count} rotated variants")

    # 6. Scaled shapes — same shape at different sizes
    # Teaches scale invariance
    print("\nGenerating scaled shapes...")
    scaled_dir = out / "scaled"
    scaled_dir.mkdir(exist_ok=True)
    scaled_count = 0

    for shape_name in SHAPES:
        for scale_name, scale_size in [("small", 400), ("large", 800)]:
            color = random.choice(shape_colors)
            bg = random.choice(BG_COLORS)
            while _color_distance(bg, color) < 80:
                bg = random.choice(BG_COLORS)

            voice = random.choice(voices)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
                audio_path = af.name

            await generate_tts(shape_name, voice, audio_path)

            video_name = f"{shape_name}_{scale_name}_{voice.split('-')[1].lower()}.mp4"
            video_path = str(scaled_dir / video_name)
            create_video(shape_name, color, bg, audio_path, video_path)
            os.unlink(audio_path)
            total += 1
            scaled_count += 1

    print(f"  {scaled_count} scaled variants")

    # 7. Outline shapes — border-only with animated thickness and color
    # Teaches that a shape's identity is invariant to fill vs outline
    print("\nGenerating outline shapes...")
    outline_dir = out / "outlines"
    outline_dir.mkdir(exist_ok=True)
    outline_count = 0

    outline_shapes = ["circle", "square", "triangle", "rectangle", "pentagon",
                      "hexagon", "oval", "diamond", "star"]

    for shape_name in outline_shapes:
        for color_idx, color in enumerate(shape_colors[:2]):
            bg = random.choice(BG_COLORS)
            while _color_distance(bg, color) < 80:
                bg = random.choice(BG_COLORS)

            voice = random.choice(voices)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as af:
                audio_path = af.name

            await generate_tts(shape_name, voice, audio_path)

            video_name = f"outline_{shape_name}_c{color_idx}_{voice.split('-')[1].lower()}.mp4"
            video_path = str(outline_dir / video_name)
            create_outline_video(shape_name, color, bg, audio_path, video_path)
            os.unlink(audio_path)
            total += 1
            outline_count += 1

    print(f"  {outline_count} outline variants")

    print(f"\nDone! Generated {total} training videos in {output_dir}")
    print(f"  colors/   : {len(COLORS) * 5} videos")
    print(f"  shapes/   : {len(SHAPES) * 3} videos")
    print(f"  compound/ : {len(compound_pairs)} videos")
    print(f"  naming/   : {len(SHAPES) + len(COLORS)} videos")
    print(f"  rotated/  : {rotated_count} videos")
    print(f"  scaled/   : {scaled_count} videos")
    print(f"  outlines/ : {outline_count} videos")


def _color_distance(c1, c2):
    """Simple Euclidean RGB distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def main():
    parser = argparse.ArgumentParser(description="Generate primitive training videos")
    parser.add_argument("--output", "-o", default="videos/primitives/",
                        help="Output directory")
    parser.add_argument("--voices", "-v", type=int, default=4,
                        help="Number of different voices (max 8)")
    args = parser.parse_args()

    asyncio.run(generate_all(args.output, min(args.voices, len(VOICES))))


if __name__ == "__main__":
    main()
