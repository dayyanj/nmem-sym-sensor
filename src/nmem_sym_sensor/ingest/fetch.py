"""
Video fetcher: download learning content from YouTube via yt-dlp.

Downloads videos matching search queries or direct URLs, organizes
them by curriculum phase, and feeds them into the learning pipeline.

Filters:
  - Duration: 1-15 minutes (skip hour-long compilations)
  - Resolution: 480p max (we don't need 4K for 128x128 segment crops)
  - Audio: yes (skip silent videos)

Dependencies:
  - yt-dlp (system binary or pip package)
"""
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class FetchConfig:
    """Configuration for video fetching."""
    # Download
    output_dir: str = "videos"
    max_videos: int = 5                # per search query
    max_duration_s: int = 900          # 15 minutes
    min_duration_s: int = 60           # 1 minute
    max_resolution: int = 480          # 480p is plenty
    prefer_subtitles: bool = True      # download auto-subs if available

    # Filtering
    skip_music_only: bool = True       # skip videos tagged as music (no speech)
    prefer_educational: bool = True    # boost "education" category videos

    # Rate limiting
    sleep_between_downloads: int = 2   # seconds between downloads (be polite)


@dataclass
class FetchedVideo:
    """Metadata for a downloaded video."""
    path: Path
    title: str
    duration_s: float
    url: str
    has_subtitles: bool = False
    subtitle_path: Path | None = None
    category: str = ""
    description: str = ""


def fetch_videos(
    query: str,
    cfg: FetchConfig | None = None,
    phase: int | None = None,
) -> list[FetchedVideo]:
    """Search and download videos matching a query.

    Args:
        query: YouTube search query or URL.
        cfg: Fetch configuration.
        phase: Curriculum phase number (organizes into subdirectory).

    Returns:
        List of FetchedVideo with local paths.
    """
    cfg = cfg or FetchConfig()
    _check_ytdlp()

    # Determine output directory
    output_dir = Path(cfg.output_dir)
    if phase:
        output_dir = output_dir / f"phase_{phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if query is a URL or search term
    is_url = query.startswith(("http://", "https://", "www."))

    if is_url:
        urls = [query]
    else:
        urls = _search_videos(query, cfg)

    if not urls:
        log.warning("No videos found for query: %s", query)
        return []

    # Download each video
    fetched = []
    for url in urls:
        try:
            video = _download_video(url, output_dir, cfg)
            if video:
                fetched.append(video)
                log.info("Downloaded: %s (%.0fs) → %s",
                        video.title, video.duration_s, video.path)
        except Exception as e:
            log.warning("Failed to download %s: %s", url, e)

    log.info("Fetched %d/%d videos for '%s'", len(fetched), len(urls), query)
    return fetched


def _check_ytdlp():
    """Verify yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "yt-dlp not found. Install with: pip install yt-dlp"
        )


def _search_videos(query: str, cfg: FetchConfig) -> list[str]:
    """Search YouTube for videos matching query. Returns list of URLs."""
    cmd = [
        "yt-dlp",
        f"ytsearch{cfg.max_videos * 2}:{query}",  # search more than needed, filter later
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log.warning("Search timed out for: %s", query)
        return []

    urls = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            info = json.loads(line)
        except json.JSONDecodeError:
            continue

        duration = info.get("duration") or 0

        # Duration filter
        if duration < cfg.min_duration_s or duration > cfg.max_duration_s:
            continue

        # Skip music-only if configured
        if cfg.skip_music_only:
            categories = info.get("categories", [])
            title = (info.get("title") or "").lower()
            if "Music" in categories and "learn" not in title and "teach" not in title:
                continue

        url = info.get("webpage_url") or info.get("url")
        if url:
            urls.append(url)

        if len(urls) >= cfg.max_videos:
            break

    return urls


def _download_video(
    url: str,
    output_dir: Path,
    cfg: FetchConfig,
) -> FetchedVideo | None:
    """Download a single video with yt-dlp."""

    # First, get metadata
    meta_cmd = [
        "yt-dlp",
        url,
        "--dump-json",
        "--no-download",
        "--no-warnings",
    ]

    try:
        meta_result = subprocess.run(
            meta_cmd, capture_output=True, text=True, timeout=30,
        )
        meta = json.loads(meta_result.stdout)
    except Exception as e:
        log.warning("Metadata fetch failed for %s: %s", url, e)
        return None

    title = meta.get("title", "unknown")
    duration = meta.get("duration", 0)
    video_id = meta.get("id", "unknown")

    # Sanitize title for filename
    safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip()
    safe_title = re.sub(r'\s+', '_', safe_title)

    output_template = str(output_dir / f"{safe_title}__{video_id}.%(ext)s")

    # Check if already downloaded
    existing = list(output_dir.glob(f"*__{video_id}.*"))
    video_extensions = {".mp4", ".mkv", ".webm"}
    existing_videos = [p for p in existing if p.suffix in video_extensions]
    if existing_videos:
        log.info("Already downloaded: %s", existing_videos[0].name)
        return FetchedVideo(
            path=existing_videos[0],
            title=title,
            duration_s=duration,
            url=url,
            category=", ".join(meta.get("categories", [])),
        )

    # Download
    dl_cmd = [
        "yt-dlp",
        url,
        "-o", output_template,
        "-f", f"bestvideo[height<={cfg.max_resolution}]+bestaudio/best[height<={cfg.max_resolution}]",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--no-playlist",
    ]

    # Subtitles
    if cfg.prefer_subtitles:
        dl_cmd.extend([
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "srt",
        ])

    try:
        subprocess.run(dl_cmd, capture_output=True, text=True, timeout=300, check=True)
    except subprocess.CalledProcessError as e:
        log.warning("Download failed for %s: %s", url, e.stderr[:300] if e.stderr else str(e))
        return None
    except subprocess.TimeoutExpired:
        log.warning("Download timed out for %s", url)
        return None

    # Find downloaded file
    downloaded = list(output_dir.glob(f"*__{video_id}.mp4"))
    if not downloaded:
        # Try other extensions
        downloaded = list(output_dir.glob(f"*__{video_id}.*"))
        downloaded = [p for p in downloaded if p.suffix in video_extensions]
    if not downloaded:
        log.warning("Download completed but file not found for %s", video_id)
        return None

    video_path = downloaded[0]

    # Check for subtitle file
    sub_path = None
    sub_files = list(output_dir.glob(f"*__{video_id}*.srt"))
    if sub_files:
        sub_path = sub_files[0]

    return FetchedVideo(
        path=video_path,
        title=title,
        duration_s=duration,
        url=url,
        has_subtitles=sub_path is not None,
        subtitle_path=sub_path,
        category=", ".join(meta.get("categories", [])),
        description=(meta.get("description") or "")[:200],
    )


def fetch_phase_videos(
    phase: int,
    cfg: FetchConfig | None = None,
    max_per_query: int = 3,
) -> list[FetchedVideo]:
    """Fetch videos appropriate for a curriculum phase.

    Uses the phase's suggested search terms to find and download
    relevant learning content.

    Args:
        phase: Curriculum phase number (1-6).
        cfg: Fetch configuration.
        max_per_query: Max videos per search query.

    Returns:
        List of all fetched videos.
    """
    from nmem_sym_sensor.ingest.curriculum import PHASES

    phase_def = PHASES[min(phase - 1, len(PHASES) - 1)]

    if not phase_def.video_tags:
        log.info("Phase %d (%s) has no suggested search terms", phase, phase_def.name)
        return []

    cfg = cfg or FetchConfig()
    cfg.max_videos = max_per_query

    all_fetched = []
    for tag in phase_def.video_tags:
        log.info("Searching: '%s' (phase %d: %s)", tag, phase, phase_def.name)
        videos = fetch_videos(tag, cfg, phase=phase)
        all_fetched.extend(videos)

    log.info("Phase %d: fetched %d videos total across %d search terms",
            phase, len(all_fetched), len(phase_def.video_tags))

    return all_fetched
