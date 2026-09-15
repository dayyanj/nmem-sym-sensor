"""
Build training videos from FlyingThings3D image sequences.

Concatenates N sequences into longer videos with 1-second black
separator between each sequence (acts as scene change). This gives
the iconic buffer enough observations to promote visual primitives
while providing diverse occlusion/motion scenarios.

Each output video = ~30 sequences × 10 frames = 300 frames at 10fps = 30s
Plus 1s black separators = ~60s per video.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

FT3D_BASE = Path("/mnt/nas_datasets/flyingthings3d/flyingthings3d__frames_cleanpass/frames_cleanpass/TRAIN")
OUTPUT_DIR = Path("/mnt/nas_projects/apps/nmem-sym-sensor/videos/flyingthings3d")
SEQUENCES_PER_VIDEO = 30
FPS = 10
MAX_VIDEOS = 10  # start with 10 videos (300 sequences)

def get_sequences():
    """Get all sequence directories sorted."""
    seqs = []
    for subset in ["A", "B", "C"]:
        subset_dir = FT3D_BASE / subset
        if subset_dir.is_dir():
            for seq_dir in sorted(subset_dir.iterdir()):
                left_dir = seq_dir / "left"
                if left_dir.is_dir():
                    frames = sorted(left_dir.glob("*.png"))
                    if len(frames) >= 5:
                        seqs.append((seq_dir.name, subset, frames))
    return seqs

def build_video(sequences, video_idx, output_dir):
    """Build a single video from multiple sequences."""
    output_path = output_dir / f"ft3d_occlusion_{video_idx:03d}.mp4"
    if output_path.exists():
        print(f"  Skip (exists): {output_path.name}")
        return output_path

    # Build a concat file listing all frames with black separators
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for seq_idx, (name, subset, frames) in enumerate(sequences):
            # Write each frame as a still for 1/FPS duration
            for frame_path in frames:
                f.write(f"file '{frame_path}'\n")
                f.write(f"duration {1.0/FPS}\n")

            # Add last frame again (ffmpeg concat needs it)
            f.write(f"file '{frames[-1]}'\n")

    # Build video with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-vf", f"fps={FPS},scale=512:384",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-200:]}")
        return None

    print(f"  Built: {output_path.name} ({len(sequences)} sequences)")
    return output_path

def main():
    max_videos = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_VIDEOS
    sequences = get_sequences()
    print(f"Found {len(sequences)} sequences")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    built = 0
    for i in range(0, len(sequences), SEQUENCES_PER_VIDEO):
        if built >= max_videos:
            break
        batch = sequences[i:i + SEQUENCES_PER_VIDEO]
        if len(batch) < 5:
            break
        result = build_video(batch, built + 1, OUTPUT_DIR)
        if result:
            built += 1

    print(f"\nBuilt {built} videos in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
