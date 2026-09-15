"""
Continuous learning pipeline.

Manages a queue of videos, prioritises unwatched content over re-watches,
runs the ingestion pipeline, and derives statistical groundings after
each batch. Can run indefinitely as new videos are added.

Priority order:
  1. Unwatched videos (watch_count = 0), oldest first
  2. Under-watched videos (watch_count < max_watches), least-watched first
  3. Nothing left → wait for new content

Usage:
    # Add videos to the queue
    await learner.add_directory("videos/playlist/")

    # Run continuous learning (blocks until queue is empty)
    await learner.run()

    # Or run a single batch
    await learner.run_batch(n=5)
"""
import logging
import random
from pathlib import Path

from nmem_sym_sensor import config
from nmem_sym_sensor.ingest.text_bridge import (
    SpeechAccumulator,
    create_text_callback,
    ground_unified,
)
from nmem_sym_sensor.ingest.video import VideoConfig, ingest_video

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


class ContinuousLearner:
    """Manages a learning queue and runs videos through the pipeline.

    Prioritises breadth (new content) over depth (re-watches).
    Derives statistical groundings after each batch.
    """

    def __init__(
        self,
        sensor_graph,
        fps: float = 2.0,
        stt_model: str = "tiny",
        batch_size: int = 5,
        ground_every: int = 5,
        max_watches: int = 3,
        recognition_engine=None,
    ):
        self.sg = sensor_graph
        self.fps = fps
        self.stt_model = stt_model
        self.batch_size = batch_size
        self.ground_every = ground_every
        self.max_watches = max_watches
        self.rec_engine = recognition_engine

        self._videos_since_grounding = 0
        self._watch_dirs: list[Path] = []  # directories to rescan for new content

    async def add_directory(self, directory: str | Path) -> int:
        """Scan a directory (recursively) and add all video files to the queue.

        Skips files already in the queue. The directory is remembered for
        periodic re-scanning so new videos added while the learner runs
        are picked up automatically.

        Returns count added.
        """
        dirpath = Path(directory).resolve()
        if not dirpath.is_dir():
            log.warning("Not a directory: %s", dirpath)
            return 0

        # Remember for re-scanning (dedup by resolved path)
        if dirpath not in self._watch_dirs:
            self._watch_dirs.append(dirpath)

        videos = sorted([
            p for p in dirpath.rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
        ])

        added = 0
        for v in videos:
            path_str = str(v.resolve())
            title = v.stem.split("__")[0] if "__" in v.stem else v.stem

            result = await self.sg.pool.execute(
                """
                INSERT INTO learning_queue (video_path, title, max_watches)
                VALUES ($1, $2, $3)
                ON CONFLICT (video_path) DO NOTHING
                """,
                path_str, title, self.max_watches,
            )
            if "INSERT 0 1" in result:
                added += 1

        log.info("Added %d/%d videos from %s to learning queue", added, len(videos), dirpath)
        return added

    async def add_video(self, path: str | Path, title: str | None = None) -> bool:
        """Add a single video to the queue."""
        path_str = str(Path(path).resolve())
        title = title or Path(path).stem
        result = await self.sg.pool.execute(
            """
            INSERT INTO learning_queue (video_path, title, max_watches)
            VALUES ($1, $2, $3)
            ON CONFLICT (video_path) DO NOTHING
            """,
            path_str, title, self.max_watches,
        )
        return "INSERT 0 1" in result

    async def queue_stats(self) -> dict:
        """Return current queue statistics."""
        total = await self.sg.pool.fetchval("SELECT COUNT(*) FROM learning_queue")
        unwatched = await self.sg.pool.fetchval(
            "SELECT COUNT(*) FROM learning_queue WHERE watch_count = 0")
        rewatchable = await self.sg.pool.fetchval(
            "SELECT COUNT(*) FROM learning_queue WHERE watch_count > 0 AND watch_count < max_watches")
        completed = await self.sg.pool.fetchval(
            "SELECT COUNT(*) FROM learning_queue WHERE watch_count >= max_watches")
        total_watches = await self.sg.pool.fetchval(
            "SELECT COALESCE(SUM(watch_count), 0) FROM learning_queue")

        return {
            "total": total,
            "unwatched": unwatched,
            "rewatchable": rewatchable,
            "completed": completed,
            "total_watches": total_watches,
        }

    async def next_video(self) -> dict | None:
        """Get the next video to learn from.

        Priority: unwatched first, then least-watched, then nothing.
        """
        row = await self.sg.pool.fetchrow(
            """
            SELECT id, video_path, title, watch_count
            FROM learning_queue
            WHERE watch_count < max_watches
              AND status != 'failed'
            ORDER BY watch_count ASC, added_at ASC
            LIMIT 1
            """,
        )
        if not row:
            return None
        return dict(row)

    async def run_batch(self, n: int | None = None) -> list[dict]:
        """Run a batch of videos through the pipeline.

        Args:
            n: Number of videos to process. Defaults to self.batch_size.

        Returns:
            List of result dicts per video.
        """
        n = n or self.batch_size
        results = []

        for i in range(n):
            video = await self.next_video()
            if not video:
                log.info("Queue empty — no more videos to learn from")
                break

            result = await self._learn_one(video, i + 1, n)
            results.append(result)

            self._videos_since_grounding += 1
            if self._videos_since_grounding >= self.ground_every:
                await self._run_grounding()
                self._videos_since_grounding = 0

        # Final grounding if any videos were processed
        if results and self._videos_since_grounding > 0:
            await self._run_grounding()
            self._videos_since_grounding = 0

        return results

    async def run(self, max_videos: int | None = None) -> list[dict]:
        """Run until the queue is exhausted.

        When the queue empties, rescans all watched directories for new
        content before stopping. This means you can drop new videos into
        any of the source directories while the learner runs and they'll
        be picked up automatically.

        Args:
            max_videos: Stop after this many videos. None = run all.

        Returns:
            List of all result dicts.
        """
        all_results = []
        processed = 0

        while True:
            video = await self.next_video()

            if not video:
                # Queue empty — rescan directories for new content
                new_found = await self._rescan_directories()
                if new_found > 0:
                    log.info("Found %d new videos from directory rescan", new_found)
                    print(f"\n  Rescan found {new_found} new videos — continuing...")
                    continue  # retry with new content
                break  # truly done

            if max_videos and processed >= max_videos:
                break

            processed += 1
            result = await self._learn_one(video, processed, max_videos or "?")
            all_results.append(result)

            self._videos_since_grounding += 1
            if self._videos_since_grounding >= self.ground_every:
                await self._run_grounding()
                self._videos_since_grounding = 0

        if self._videos_since_grounding > 0:
            await self._run_grounding()

        return all_results

    async def _rescan_directories(self) -> int:
        """Rescan all watched directories for new videos.

        Called when the queue empties. Picks up any videos added to the
        source directories since the last scan.
        """
        total_new = 0
        for dirpath in self._watch_dirs:
            if dirpath.is_dir():
                added = await self.add_directory(dirpath)
                total_new += added
        return total_new

    async def _learn_one(self, video: dict, index: int, total) -> dict:
        """Process a single video."""
        path = video["video_path"]
        title = video["title"]
        watch_num = video["watch_count"] + 1
        rewatch = " (re-watch)" if watch_num > 1 else ""

        log.info("[%s/%s] %s (watch #%d)%s", index, total, title, watch_num, rewatch)
        print(f"  [{index}/{total}] {title} (watch #{watch_num}){rewatch}")

        # Mark as in-progress
        await self.sg.pool.execute(
            "UPDATE learning_queue SET status = 'learning' WHERE id = $1",
            video["id"],
        )

        # Randomize FPS between 2-5 per watch for variation —
        # same video at different frame rates produces different visual
        # samples, strengthening robust sound-visual associations.
        watch_fps = round(random.uniform(2.0, 5.0), 1)
        cfg = VideoConfig(
            fps=watch_fps,
            stt_model=self.stt_model,
            skip_similar_frames=False,  # foveal habituation handles redundancy
        )
        accumulator = SpeechAccumulator()
        text_cb = create_text_callback(
            sensor_graph=self.sg,
            accumulator=accumulator,
        )

        try:
            progress = await ingest_video(
                self.sg, path, cfg,
                text_callback=text_cb,
                recognition_engine=self.rec_engine,
            )

            accumulator.flush()

            # Update queue
            await self.sg.pool.execute(
                """
                UPDATE learning_queue
                SET watch_count = watch_count + 1,
                    last_watched = NOW(),
                    frames_processed = frames_processed + $1,
                    speech_segments = speech_segments + $2,
                    status = 'queued'
                WHERE id = $3
                """,
                progress.processed_frames,
                progress.transcript_segments,
                video["id"],
            )

            avg_surprise = round(
                progress.total_surprise / max(progress.intelligence_cycles, 1), 3
            )
            result = {
                "title": title,
                "watch": watch_num,
                "frames": progress.processed_frames,
                "visual": progress.visual_nodes_created,
                "audio": progress.audio_nodes_created,
                "speech": progress.transcript_segments,
                "bindings": progress.bindings_created,
                "intelligence_cycles": progress.intelligence_cycles,
                "avg_surprise": avg_surprise,
                "confirmed": progress.predictions_confirmed,
                "refuted": progress.predictions_refuted,
                "errors": len(progress.errors),
            }

            print(f"    Done: {progress.processed_frames} frames, "
                  f"{progress.visual_nodes_created} visual, "
                  f"{progress.transcript_segments} speech, "
                  f"surprise={avg_surprise:.2f}")

            return result

        except Exception as e:
            log.error("Failed to learn %s: %s", title, e)
            await self.sg.pool.execute(
                "UPDATE learning_queue SET status = 'failed' WHERE id = $1",
                video["id"],
            )
            print(f"    FAILED: {e}")
            return {"title": title, "error": str(e)}

    async def _run_grounding(self):
        """Derive groundings from accumulated statistics."""
        print("  → Deriving groundings from statistics...")
        groundings = await ground_unified(self.sg.pool)

        new_groundings = [g for g in groundings if g["action"] == "grounded"]
        confirmations = [g for g in groundings if g["action"] == "confirmed"]

        if new_groundings:
            for g in new_groundings:
                print(f"    NEW: \"{g['word']}\" → {g['node_type']}:{g['node']} "
                      f"(heard {g['count']}x, {g['dominance']*100:.0f}%)")
        if confirmations:
            print(f"    Confirmed {len(confirmations)} existing groundings")
        if not groundings:
            print("    No new groundings yet")

        # Run sensory dreamstate — pressure-driven if available, else full cycle
        try:
            dream_stats = await self.sg.check_and_run_pressure_dreamstate()
            if dream_stats is None and config.DREAMSTATE_EVERY_VIDEOS > 0:
                # No pressure system or no pressure — fall back to full cycle
                dream_stats = await self.sg.run_dreamstate()
            if dream_stats:
                consolidated = sum(v for v in dream_stats.values() if isinstance(v, int))
                ran = dream_stats.get("ran_functions", [])
                selective = dream_stats.get("selective", False)
                mode = f"selective: {ran}" if selective else "full"
                print(f"    Dreamstate ({mode}): {consolidated} operations in {dream_stats.get('duration_s', 0):.1f}s")
        except Exception as e:
            log.warning("Dreamstate failed: %s", e, exc_info=True)

        stats = await self.sg.stats()
        print(f"    Graph: {stats['nodes']} nodes, {stats['grounded_clusters']} grounded")
