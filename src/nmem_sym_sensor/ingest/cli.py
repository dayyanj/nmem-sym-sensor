"""
CLI for video ingestion and curriculum management.

Usage:
    # Ingest a single video
    python -m nmem_sym_sensor.ingest learn video.mp4

    # Ingest a directory of videos
    python -m nmem_sym_sensor.ingest learn-dir /path/to/videos/

    # Check curriculum status
    python -m nmem_sym_sensor.ingest status

    # Advance to next phase (if ready)
    python -m nmem_sym_sensor.ingest advance

    # Run rechallenge cycle
    python -m nmem_sym_sensor.ingest rechallenge
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sensory memory video learning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=None, help="PostgreSQL DSN (or NMEM_SENSOR_DB_DSN env)")
    parser.add_argument("--data-dir", default="sensory_data", help="Directory for curriculum state")
    parser.add_argument("--llm-url", default=None, help="vLLM endpoint URL for real-time recognition (e.g. http://<vllm-host>:8000/v1/chat/completions)")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ", help="Model name for LLM endpoint")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── Learn single video ───────────────────────────────
    p_learn = sub.add_parser("learn", help="Ingest a single video")
    p_learn.add_argument("video", help="Path to video file")
    p_learn.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate")
    p_learn.add_argument("--stt-model", default="base", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"])
    p_learn.add_argument("--no-stt", action="store_true", help="Skip speech-to-text")
    p_learn.add_argument("--max-frames", type=int, default=None)
    p_learn.add_argument("--visual-backend", default=None, choices=["edge"])
    p_learn.add_argument("--llm-url", default=None, help="vLLM endpoint for real-time recognition")
    p_learn.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ", help="Model name")

    # ── Learn directory ──────────────────────────────────
    p_dir = sub.add_parser("learn-dir", help="Ingest all videos in a directory")
    p_dir.add_argument("directory", help="Path to directory of videos")
    p_dir.add_argument("--fps", type=float, default=2.0)
    p_dir.add_argument("--stt-model", default="base", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"])
    p_dir.add_argument("--no-stt", action="store_true")
    p_dir.add_argument("--max-frames", type=int, default=None)
    p_dir.add_argument("--max-videos", type=int, default=None, help="Cap number of videos")
    p_dir.add_argument("--visual-backend", default=None, choices=["edge"])
    p_dir.add_argument("--llm-url", default=None, help="vLLM endpoint for real-time recognition")
    p_dir.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ", help="Model name")

    # ── Fetch and learn ─────────────────────────────────
    p_fetch = sub.add_parser("fetch-and-learn", help="Download videos from YouTube and learn from them")
    p_fetch.add_argument("query", help="YouTube search query or URL")
    p_fetch.add_argument("--max-videos", type=int, default=3)
    p_fetch.add_argument("--fps", type=float, default=2.0)
    p_fetch.add_argument("--stt-model", default="base", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"])
    p_fetch.add_argument("--max-frames", type=int, default=None)
    p_fetch.add_argument("--phase", type=int, default=None, help="Curriculum phase (organizes downloads)")
    p_fetch.add_argument("--output-dir", default="videos", help="Download directory")
    p_fetch.add_argument("--visual-backend", default=None, choices=["edge"])

    # ── Fetch phase videos ───────────────────────────────
    p_phase_fetch = sub.add_parser("fetch-phase", help="Download videos for current curriculum phase")
    p_phase_fetch.add_argument("--max-per-query", type=int, default=2)
    p_phase_fetch.add_argument("--output-dir", default="videos")

    # ── Continuous learner ──────────────────────────────
    p_run = sub.add_parser("run", help="Continuous learning: add videos to queue and learn")
    p_run.add_argument("directories", nargs="+", help="Directories of videos to add to queue")
    p_run.add_argument("--fps", type=float, default=2.0)
    p_run.add_argument("--stt-model", default="tiny", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"])
    p_run.add_argument("--max-videos", type=int, default=None, help="Stop after N videos")
    p_run.add_argument("--max-watches", type=int, default=3, help="Max times to re-watch each video")
    p_run.add_argument("--ground-every", type=int, default=5, help="Derive groundings every N videos")
    p_run.add_argument("--llm-url", default=None, help="vLLM endpoint for recognition")
    p_run.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ")

    sub.add_parser("queue", help="Show learning queue status")

    # ── Live streaming ───────────────────────────────────
    p_live = sub.add_parser("live", help="Real-time perception from camera + microphone")
    p_live.add_argument("--camera", default="0", help="Camera device index or RTSP URL")
    p_live.add_argument("--no-camera", action="store_true", help="Disable camera")
    p_live.add_argument("--no-mic", action="store_true", help="Disable microphone")
    p_live.add_argument("--fps", type=float, default=2.0, help="Camera FPS")
    p_live.add_argument("--stt", action="store_true", help="Enable speech-to-text")
    p_live.add_argument("--stt-model", default="tiny", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"])
    p_live.add_argument("--consolidate-every", type=float, default=30.0, help="Seconds between consolidation")
    p_live.add_argument("--llm-url", default=None, help="vLLM endpoint for recognition")
    p_live.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ")

    # ── Curriculum status ────────────────────────────────
    sub.add_parser("status", help="Show curriculum status and phase progress")

    # ── Advance phase ────────────────────────────────────
    sub.add_parser("advance", help="Advance to next curriculum phase if ready")

    # ── Graph stats ──────────────────────────────────────
    sub.add_parser("stats", help="Show sensory graph statistics")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asyncio.run(_dispatch(args))


async def _dispatch(args):
    if args.command == "learn":
        await _learn_video(args)
    elif args.command == "learn-dir":
        await _learn_directory(args)
    elif args.command == "run":
        await _run_continuous(args)
    elif args.command == "live":
        await _run_live(args)
    elif args.command == "queue":
        await _show_queue(args)
    elif args.command == "fetch-and-learn":
        await _fetch_and_learn(args)
    elif args.command == "fetch-phase":
        await _fetch_phase(args)
    elif args.command == "status":
        await _show_status(args)
    elif args.command == "advance":
        await _advance(args)
    elif args.command == "stats":
        await _show_stats(args)


async def _learn_video(args):
    from nmem_sym_sensor.api import SensorGraph
    from nmem_sym_sensor.ingest.curriculum import load_state, save_state
    from nmem_sym_sensor.ingest.text_bridge import SpeechAccumulator, create_text_callback
    from nmem_sym_sensor.ingest.video import VideoConfig, ingest_video

    db_dsn = args.db or _get_db_dsn()

    cfg = VideoConfig(
        fps=args.fps,
        max_frames=args.max_frames,
        stt_model=args.stt_model if not args.no_stt else "tiny",
    )

    async with SensorGraph(
        db_dsn=db_dsn,
        visual_backend=args.visual_backend,
    ) as sg:
        # Text bridge (no nmem connection for now — just log)
        accumulator = SpeechAccumulator()
        text_cb = create_text_callback(
            nmem_write_fn=None,  # TODO: connect to nmem when available
            sensor_graph=sg,
            accumulator=accumulator,
        ) if not args.no_stt else None

        # Recognition engine (if LLM endpoint provided)
        rec_engine = None
        if args.llm_url:
            llm_fn = _create_llm_callable(args.llm_url, args.llm_model)
            rec_engine = sg.create_recognition_engine(llm_fn)
            await rec_engine.warm_cache()
            print(f"  Recognition engine active ({rec_engine.stats()['cache_size']} cached)")

        # Progress callback
        async def on_progress(progress):
            pct = (progress.processed_frames / max(progress.total_frames, 1)) * 100
            print(f"\r  [{pct:5.1f}%] frames={progress.processed_frames}/{progress.total_frames} "
                  f"visual={progress.visual_nodes_created} audio={progress.audio_nodes_created} "
                  f"bindings={progress.bindings_created} speech={progress.groundings_from_speech}",
                  end="", flush=True)

        print(f"Learning from: {args.video}")
        progress = await ingest_video(
            sg, args.video, cfg,
            text_callback=text_cb,
            progress_callback=on_progress,
            recognition_engine=rec_engine,
        )
        print()  # newline after progress bar

        # Flush accumulator
        if accumulator:
            accumulator.flush()

        # Update curriculum state
        state = load_state(args.data_dir)
        state.total_videos_ingested += 1
        state.total_frames_processed += progress.processed_frames
        save_state(state, args.data_dir)

        # Print summary
        print("\nIngestion complete:")
        print(f"  Frames:       {progress.processed_frames}/{progress.total_frames}")
        print(f"  Visual nodes: {progress.visual_nodes_created}")
        print(f"  Audio nodes:  {progress.audio_nodes_created}")
        print(f"  Bindings:     {progress.bindings_created}")
        print(f"  Speech segs:  {progress.transcript_segments}")
        if progress.errors:
            print(f"  Errors:       {len(progress.errors)}")
            for e in progress.errors[:5]:
                print(f"    - {e}")

        # Show graph stats
        stats = await sg.stats()
        print("\nGraph state:")
        print(f"  Total nodes:  {stats['nodes']}")
        print(f"  Total edges:  {stats['edges']}")
        print(f"  Clusters:     {stats['clusters']}")
        print(f"  Grounded:     {stats['grounded_clusters']}")


async def _learn_directory(args):
    from nmem_sym_sensor.api import SensorGraph
    from nmem_sym_sensor.ingest.curriculum import load_state, save_state
    from nmem_sym_sensor.ingest.text_bridge import SpeechAccumulator, create_text_callback
    from nmem_sym_sensor.ingest.video import VideoConfig, ingest_video

    db_dsn = args.db or _get_db_dsn()
    video_dir = Path(args.directory)

    if not video_dir.is_dir():
        print(f"Not a directory: {video_dir}")
        sys.exit(1)

    # Find video files
    extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    videos = sorted([
        p for p in video_dir.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    ])

    if args.max_videos:
        videos = videos[:args.max_videos]

    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} videos in {video_dir}")

    cfg = VideoConfig(
        fps=args.fps,
        max_frames=args.max_frames,
        stt_model=args.stt_model if not args.no_stt else "tiny",
    )

    state = load_state(args.data_dir)

    async with SensorGraph(
        db_dsn=db_dsn,
        visual_backend=args.visual_backend,
    ) as sg:
        # Recognition engine
        rec_engine = None
        if args.llm_url:
            llm_fn = _create_llm_callable(args.llm_url, args.llm_model)
            rec_engine = sg.create_recognition_engine(llm_fn)
            await rec_engine.warm_cache()
            print(f"Recognition engine active ({rec_engine.stats()['cache_size']} cached)")

        for i, video_path in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] {video_path.name}")

            accumulator = SpeechAccumulator()
            text_cb = create_text_callback(
                sensor_graph=sg,
                accumulator=accumulator,
            ) if not args.no_stt else None

            try:
                progress = await ingest_video(
                    sg, video_path, cfg, text_callback=text_cb,
                    recognition_engine=rec_engine,
                )
                state.total_videos_ingested += 1
                state.total_frames_processed += progress.processed_frames
                rec_stats = rec_engine.stats() if rec_engine else {}
                print(f"  Done: {progress.processed_frames} frames, "
                      f"{progress.visual_nodes_created} visual, "
                      f"{progress.audio_nodes_created} audio"
                      + (f", recognition: {rec_stats.get('total_cache_hits',0)} cached / {rec_stats.get('total_llm_calls',0)} LLM" if rec_engine else ""))
            except Exception as e:
                print(f"  FAILED: {e}")

            if accumulator:
                accumulator.flush()

        save_state(state, args.data_dir)

        # After all videos: derive groundings from accumulated statistics
        print("\nDeriving groundings from co-occurrence statistics...")
        from nmem_sym_sensor.ingest.text_bridge import ground_from_statistics
        groundings = await ground_from_statistics(sg.pool)
        if groundings:
            for g in groundings:
                if g["action"] == "grounded":
                    print(f"  Grounded: \"{g['word']}\" → {g['node_type']}:{g['node']} "
                          f"(heard {g['count']}x, {g['dominance']*100:.0f}% dominant)")
                elif g["action"] == "confirmed":
                    print(f"  Confirmed: \"{g['word']}\" → {g['node']} (conf={g['confidence']:.3f})")
        else:
            print("  No groundings yet (need more data)")

        stats = await sg.stats()
        print(f"\nGraph: {stats['nodes']} nodes, {stats['edges']} edges, "
              f"{stats['clusters']} clusters, {stats['grounded_clusters']} grounded")


async def _run_continuous(args):
    from nmem_sym_sensor.api import SensorGraph
    from nmem_sym_sensor.ingest.learner import ContinuousLearner

    db_dsn = args.db or _get_db_dsn()

    async with SensorGraph(db_dsn=db_dsn) as sg:
        rec_engine = None
        if args.llm_url:
            llm_fn = _create_llm_callable(args.llm_url, args.llm_model)
            rec_engine = sg.create_recognition_engine(llm_fn)
            await rec_engine.warm_cache()

        learner = ContinuousLearner(
            sg,
            fps=args.fps,
            stt_model=args.stt_model,
            ground_every=args.ground_every,
            max_watches=args.max_watches,
            recognition_engine=rec_engine,
        )

        # Add all directories to queue
        for d in args.directories:
            added = await learner.add_directory(d)
            print(f"Added {added} videos from {d}")

        queue = await learner.queue_stats()
        print(f"\nQueue: {queue['unwatched']} unwatched, "
              f"{queue['rewatchable']} rewatchable, "
              f"{queue['completed']} completed\n")

        # Run
        results = await learner.run(max_videos=args.max_videos)

        # Summary
        queue = await learner.queue_stats()
        stats = await sg.stats()
        print("\n=== Learning Complete ===")
        print(f"  Videos processed: {len(results)}")
        print(f"  Total watches: {queue['total_watches']}")
        print(f"  Queue remaining: {queue['unwatched']} unwatched, {queue['rewatchable']} rewatchable")
        print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
              f"{stats['grounded_clusters']} grounded")


async def _run_live(args):
    from nmem_sym_sensor.api import SensorGraph
    from nmem_sym_sensor.sensor_bus import SensorBus
    from nmem_sym_sensor.streaming import StreamingLoop

    db_dsn = args.db or _get_db_dsn()

    async with SensorGraph(db_dsn=db_dsn) as sg:
        bus = SensorBus()

        # Register camera
        if not args.no_camera:
            from nmem_sym_sensor.sensors.camera import CameraSensor
            # Try to parse as int (device index), otherwise treat as URL
            try:
                source = int(args.camera)
            except ValueError:
                source = args.camera
            bus.register(CameraSensor(
                source=source,
                fps=args.fps,
                queue_size=64,
            ))

        # Register microphone
        if not args.no_mic:
            try:
                from nmem_sym_sensor.sensors.microphone import MicrophoneSensor
                bus.register(MicrophoneSensor(queue_size=128))
            except ImportError:
                print("sounddevice not installed, skipping microphone")

        # Recognition engine
        rec_engine = None
        if args.llm_url:
            llm_fn = _create_llm_callable(args.llm_url, args.llm_model)
            rec_engine = sg.create_recognition_engine(llm_fn)
            await rec_engine.warm_cache()

        loop = StreamingLoop(
            sg, bus,
            consolidation_interval_s=args.consolidate_every,
            stt_enabled=args.stt,
            stt_model=args.stt_model,
            recognition_engine=rec_engine,
        )

        print("Starting live perception...")
        sensors = bus.sensor_stats()
        for s in sensors:
            print(f"  {s['sensor_id']} ({s['modality']})")
        print(f"  STT: {'enabled' if args.stt else 'disabled'}")
        print(f"  Consolidation: every {args.consolidate_every}s")
        print("Press Ctrl+C to stop.\n")

        async with bus:
            try:
                await loop.run()
            except KeyboardInterrupt:
                pass

        # Final stats
        stats = loop.stats
        graph = await sg.stats()
        print("\n=== Session Complete ===")
        print(f"  Duration: {stats['elapsed_s']}s")
        print(f"  Frames: {stats['frames']} ({stats['fps']} fps)")
        print(f"  Audio windows: {stats['audio_windows']}")
        print(f"  Text events: {stats['text_events']}")
        print(f"  Graph: {graph['nodes']} nodes, {graph['edges']} edges, "
              f"{graph['grounded_clusters']} grounded")


async def _show_queue(args):
    import asyncpg

    db_dsn = args.db or _get_db_dsn()
    pool = await asyncpg.create_pool(db_dsn)
    try:
        queue = await pool.fetch("""
            SELECT title, watch_count, max_watches, frames_processed,
                   speech_segments, status, last_watched
            FROM learning_queue
            ORDER BY watch_count ASC, added_at ASC
            LIMIT 30
        """)
        total = await pool.fetchval("SELECT COUNT(*) FROM learning_queue")
        unwatched = await pool.fetchval("SELECT COUNT(*) FROM learning_queue WHERE watch_count = 0")

        print(f"=== Learning Queue ({total} videos, {unwatched} unwatched) ===\n")
        for q in queue:
            watched = f"watched {q['watch_count']}/{q['max_watches']}" if q["watch_count"] > 0 else "unwatched"
            print(f"  [{watched:>12}] {q['title'][:50]}")

        if total > 30:
            print(f"  ... and {total - 30} more")
    finally:
        await pool.close()


async def _show_status(args):
    import asyncpg

    from nmem_sym_sensor.ingest.curriculum import (
        evaluate_phase,
        format_curriculum_status,
        load_state,
    )

    db_dsn = args.db or _get_db_dsn()
    state = load_state(args.data_dir)

    pool = await asyncpg.create_pool(db_dsn)
    try:
        state = await evaluate_phase(pool, state.current_phase)
        # Preserve ingestion counts from saved state
        saved = load_state(args.data_dir)
        state.total_videos_ingested = saved.total_videos_ingested
        state.total_frames_processed = saved.total_frames_processed
        state.phase_history = saved.phase_history
        print(format_curriculum_status(state))
    finally:
        await pool.close()


async def _advance(args):
    import asyncpg

    from nmem_sym_sensor.ingest.curriculum import (
        advance_phase,
        evaluate_phase,
        format_curriculum_status,
        load_state,
    )

    db_dsn = args.db or _get_db_dsn()
    state = load_state(args.data_dir)

    pool = await asyncpg.create_pool(db_dsn)
    try:
        state = await evaluate_phase(pool, state.current_phase)
        saved = load_state(args.data_dir)
        state.total_videos_ingested = saved.total_videos_ingested
        state.total_frames_processed = saved.total_frames_processed
        state.phase_history = saved.phase_history

        if state.ready_to_advance:
            state = await advance_phase(pool, state, args.data_dir)
            print(f"Advanced to phase {state.current_phase}: {state.phase_name}")
        else:
            print("Not ready to advance.")

        print()
        print(format_curriculum_status(state))
    finally:
        await pool.close()


async def _show_stats(args):
    from nmem_sym_sensor.api import SensorGraph

    db_dsn = args.db or _get_db_dsn()
    async with SensorGraph(db_dsn=db_dsn) as sg:
        stats = await sg.stats()
        print("=== Sensory Graph Statistics ===")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")


async def _fetch_and_learn(args):
    from nmem_sym_sensor.api import SensorGraph
    from nmem_sym_sensor.ingest.curriculum import load_state, save_state
    from nmem_sym_sensor.ingest.fetch import FetchConfig, fetch_videos
    from nmem_sym_sensor.ingest.text_bridge import SpeechAccumulator, create_text_callback
    from nmem_sym_sensor.ingest.video import VideoConfig, ingest_video

    db_dsn = args.db or _get_db_dsn()

    # Fetch videos
    fetch_cfg = FetchConfig(
        output_dir=args.output_dir,
        max_videos=args.max_videos,
    )

    print(f"Fetching videos: '{args.query}'")
    videos = fetch_videos(args.query, fetch_cfg, phase=args.phase)

    if not videos:
        print("No videos downloaded.")
        return

    print(f"Downloaded {len(videos)} videos:")
    for v in videos:
        print(f"  {v.title} ({v.duration_s:.0f}s)")

    # Learn from each
    video_cfg = VideoConfig(
        fps=args.fps,
        max_frames=args.max_frames,
        stt_model=args.stt_model,
    )

    state = load_state(args.data_dir)

    async with SensorGraph(
        db_dsn=db_dsn,
        visual_backend=args.visual_backend,
    ) as sg:
        for i, video in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] Learning: {video.title}")

            accumulator = SpeechAccumulator()
            text_cb = create_text_callback(
                sensor_graph=sg,
                accumulator=accumulator,
            )

            try:
                progress = await ingest_video(sg, video.path, video_cfg, text_callback=text_cb)
                state.total_videos_ingested += 1
                state.total_frames_processed += progress.processed_frames
                print(f"  Done: {progress.processed_frames} frames, "
                      f"{progress.visual_nodes_created} visual, "
                      f"{progress.audio_nodes_created} audio, "
                      f"{progress.transcript_segments} speech segments")
            except Exception as e:
                print(f"  FAILED: {e}")

            accumulator.flush()

        save_state(state, args.data_dir)

        stats = await sg.stats()
        print("\nAll videos processed.")
        print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
              f"{stats['clusters']} clusters, {stats['grounded_clusters']} grounded")


async def _fetch_phase(args):
    from nmem_sym_sensor.ingest.curriculum import get_current_phase, load_state
    from nmem_sym_sensor.ingest.fetch import FetchConfig, fetch_phase_videos

    state = load_state(args.data_dir)
    phase = get_current_phase(state.current_phase)

    print(f"Fetching videos for phase {state.current_phase}: {phase.name}")
    print(f"  {phase.description}")

    fetch_cfg = FetchConfig(
        output_dir=args.output_dir,
        max_videos=args.max_per_query,
    )

    videos = fetch_phase_videos(state.current_phase, fetch_cfg, args.max_per_query)

    if not videos:
        print("No videos found for this phase.")
        return

    print(f"\nDownloaded {len(videos)} videos:")
    for v in videos:
        print(f"  {v.title} ({v.duration_s:.0f}s)")

    print(f"\nNow run: python -m nmem_sym_sensor.ingest learn-dir {args.output_dir}/phase_{state.current_phase}/")


def _create_llm_callable(llm_url: str, model_name: str):
    """Create an async LLM callable for the recognition engine."""
    import httpx

    async def ask_llm(prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(llm_url, json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
                "max_tokens": 20,
                "temperature": 0.1,
            })
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content is None:
                content = resp.json()["choices"][0]["message"].get("reasoning", "")
            return content.strip() if content else ""

    return ask_llm


def _get_db_dsn() -> str:
    import os
    dsn = os.environ.get("NMEM_SENSOR_DB_DSN")
    if not dsn:
        print("Error: --db or NMEM_SENSOR_DB_DSN required")
        sys.exit(1)
    return dsn


if __name__ == "__main__":
    main()
