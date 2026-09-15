"""
Sensory Mind Dashboard — monitor what the system is learning.

A lightweight web UI to inspect:
- Node/edge/co-occurrence stats (overview)
- Sound-visual binding with myelination status
- Word-visual co-occurrences (diagnostic text path)
- Concept links (sound units that mean the same thing)
- Scene model (chronoception, active set, spatial graph)
- Dream inspector (mind's eye — what does it "see" when it hears a word?)
- Recall test (interactive word→visual and visual→word)
- Familiarity stats (processing reduction)

Run:
    python -m nmem_sym_sensor.dashboard [--port 9500]
"""
import logging
import os
from io import BytesIO

import asyncpg
import numpy as np
import onnxruntime as ort
from PIL import Image

log = logging.getLogger(__name__)

DB_DSN = os.environ.get("NMEM_SENSOR_DB_DSN")  # required — no hardcoded credential

_pool: asyncpg.Pool | None = None
_decoder_sess: ort.InferenceSession | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DB_DSN:
            raise RuntimeError("NMEM_SENSOR_DB_DSN is not set — the dashboard needs the sensory DB DSN")
        _pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=3)
    return _pool


def get_decoder() -> ort.InferenceSession | None:
    global _decoder_sess
    if _decoder_sess is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "visual_decoder_v5.onnx")
        if os.path.exists(model_path):
            _decoder_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _decoder_sess


# ── Queries ──────────────────────────────────────────────

async def fetch_overview(pool: asyncpg.Pool) -> dict:
    s = {}
    s["nodes_active"] = await pool.fetchval("SELECT COUNT(*) FROM sensory_nodes WHERE NOT archived")
    s["nodes_archived"] = await pool.fetchval("SELECT COUNT(*) FROM sensory_nodes WHERE archived")
    s["edges"] = await pool.fetchval("SELECT COUNT(*) FROM sensory_edges")
    s["clusters"] = await pool.fetchval("SELECT COUNT(*) FROM sensory_clusters")

    # Unified co-occurrences (PRIMARY learning path)
    from nmem_sym_sensor.cooccurrence import cooc_store
    cooc_stats = await cooc_store.stats(pool)
    s["sv_pairs"] = cooc_stats["active_pairs"]
    s["sv_obs"] = cooc_stats["total_obs"]
    s["sv_myelinated"] = cooc_stats["myelinated_pairs"]
    s["sv_dormant"] = cooc_stats["dormant_pairs"]
    s["cooc_by_modality"] = cooc_stats["by_modality"]

    # Sound units
    s["sound_units"] = await pool.fetchval("SELECT COUNT(*) FROM sound_units")
    s["sound_voice"] = await pool.fetchval("SELECT COUNT(*) FROM sound_units WHERE modality = 'voice'")
    s["sound_env"] = await pool.fetchval("SELECT COUNT(*) FROM sound_units WHERE modality = 'environmental'")
    s["sound_labelled"] = await pool.fetchval("SELECT COUNT(*) FROM sound_units WHERE stt_label IS NOT NULL")

    # Concept links
    s["concept_links"] = await pool.fetchval("SELECT COUNT(*) FROM concept_links")
    s["sequences"] = await pool.fetchval("SELECT COUNT(*) FROM sound_sequences")

    # Word-visual (DIAGNOSTIC text path)
    s["wv_pairs"] = await pool.fetchval("SELECT COUNT(*) FROM word_visual_cooccurrences")
    s["wv_obs"] = await pool.fetchval("SELECT COALESCE(SUM(count), 0) FROM word_visual_cooccurrences")

    # Scenes
    s["scenes"] = await pool.fetchval("SELECT COUNT(*) FROM scene_snapshots")
    s["scene_members"] = await pool.fetchval("SELECT COUNT(*) FROM scene_members")
    s["scene_edges"] = await pool.fetchval("SELECT COUNT(*) FROM scene_spatial_edges")

    # Saccade patterns
    s["saccade_patterns"] = await pool.fetchval("SELECT COUNT(*) FROM saccade_patterns")

    # Dreamstate recombination
    s["recombinations"] = await pool.fetchval(
        "SELECT COUNT(*) FROM dreamstate_recombinations"
    )
    s["recombination_recent"] = await pool.fetchval(
        "SELECT COUNT(*) FROM dreamstate_recombinations WHERE created_at > NOW() - INTERVAL '24 hours'"
    )
    avg_coherence = await pool.fetchval(
        "SELECT AVG(coherence_score) FROM dreamstate_recombinations WHERE created_at > NOW() - INTERVAL '24 hours'"
    )
    s["recombination_avg_coherence"] = round(float(avg_coherence or 0), 3)

    # Context universality (from recombination outcomes)
    s["high_universality"] = await pool.fetchval(
        """SELECT COUNT(*) FROM sensory_cooccurrences
           WHERE context_independence > 0
             AND context_independence > context_dependence * 2"""
    )
    s["high_context_dependence"] = await pool.fetchval(
        """SELECT COUNT(*) FROM sensory_cooccurrences
           WHERE context_dependence > 0
             AND context_dependence > context_independence * 2"""
    )

    # Mental rotation
    s["rotations"] = await pool.fetchval(
        "SELECT COUNT(*) FROM dreamstate_rotations"
    )
    s["rotations_recent"] = await pool.fetchval(
        "SELECT COUNT(*) FROM dreamstate_rotations WHERE created_at > NOW() - INTERVAL '24 hours'"
    )
    s["rotation_invariant"] = await pool.fetchval(
        "SELECT COUNT(*) FROM dreamstate_rotations WHERE verdict = 'invariant'"
    )
    s["rotation_sensitive"] = await pool.fetchval(
        "SELECT COUNT(*) FROM dreamstate_rotations WHERE verdict = 'sensitive'"
    )
    s["high_orient_independence"] = await pool.fetchval(
        """SELECT COUNT(*) FROM sensory_cooccurrences
           WHERE orientation_independence > 0
             AND orientation_independence > orientation_dependence * 2"""
    )

    # Memory tiers
    tiers = await pool.fetch(
        "SELECT memory_tier, COUNT(*) as n FROM sensory_nodes WHERE NOT archived GROUP BY memory_tier"
    )
    s["tiers"] = {r["memory_tier"]: r["n"] for r in tiers}
    return s


async def fetch_sound_visual_top(pool: asyncpg.Pool, limit: int = 30) -> list[dict]:
    rows = await pool.fetch("""
        SELECT sc.unit_a_id, sc.modality_a, sc.unit_b_id, sc.modality_b,
               sc.count, sc.myelinated, sc.dreamstate_challenges,
               sc.self_play_confirmations, sc.last_seen
        FROM sensory_cooccurrences sc
        WHERE sc.count > 0
        ORDER BY sc.count DESC LIMIT $1
    """, limit)
    results = []
    for r in rows:
        # Resolve labels based on modality
        label_a = await _resolve_label(pool, r["unit_a_id"], r["modality_a"])
        label_b = await _resolve_label(pool, r["unit_b_id"], r["modality_b"])
        results.append({
            "stt_label": label_a, "su_mod": r["modality_a"],
            "vis_label": label_b, "vis_mod": r["modality_b"],
            "count": r["count"], "myelinated": r["myelinated"],
            "dreamstate_challenges": r["dreamstate_challenges"],
            "self_play_confirmations": r["self_play_confirmations"],
            "last_seen": r["last_seen"],
        })
    return results


async def _resolve_label(pool: asyncpg.Pool, unit_id: int, modality: str) -> str:
    """Resolve a unit ID + modality to a human-readable label."""
    if modality in ("voice", "environmental"):
        row = await pool.fetchrow("SELECT stt_label FROM sound_units WHERE id = $1", unit_id)
        return (row["stt_label"] if row and row["stt_label"] else "?") + f" ({modality})"
    elif modality == "visual":
        row = await pool.fetchrow("SELECT label FROM sensory_nodes WHERE id = $1", unit_id)
        return row["label"] if row else f"node:{unit_id}"
    elif modality == "motor":
        row = await pool.fetchrow("SELECT label FROM saccade_patterns WHERE id = $1", unit_id)
        return row["label"] if row and row["label"] else f"saccade:{unit_id}"
    elif modality == "text":
        return f"text:{unit_id}"
    return f"{modality}:{unit_id}"


async def fetch_concept_links(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch("""
        SELECT a.stt_label as label_a, b.stt_label as label_b,
               cl.overlap_score, cl.strength
        FROM concept_links cl
        JOIN sound_units a ON a.id = cl.unit_a_id
        JOIN sound_units b ON b.id = cl.unit_b_id
        ORDER BY cl.strength DESC
    """)
    return [dict(r) for r in rows]


async def fetch_scenes(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch("""
        SELECT ss.id, ss.visit_count,
               ss.first_seen, ss.last_seen,
               (SELECT COUNT(*) FROM scene_members sm WHERE sm.scene_id = ss.id) as members,
               (SELECT COUNT(*) FROM scene_spatial_edges se WHERE se.scene_id = ss.id) as spatial_edges
        FROM scene_snapshots ss
        ORDER BY ss.last_seen DESC NULLS LAST
    """)
    return [dict(r) for r in rows]


async def fetch_scene_detail(pool: asyncpg.Pool, scene_id: int) -> dict:
    members = await pool.fetch("""
        SELECT sm.cluster_id, sc.grounded_label, sm.observation_count
        FROM scene_members sm
        LEFT JOIN sensory_clusters sc ON sc.id = sm.cluster_id
        WHERE sm.scene_id = $1
        ORDER BY sm.observation_count DESC
    """, scene_id)
    edges = await pool.fetch("""
        SELECT se.cluster_a_id, se.cluster_b_id, se.relation,
               se.confidence, se.observation_count,
               a.grounded_label as label_a, b.grounded_label as label_b
        FROM scene_spatial_edges se
        LEFT JOIN sensory_clusters a ON a.id = se.cluster_a_id
        LEFT JOIN sensory_clusters b ON b.id = se.cluster_b_id
        WHERE se.scene_id = $1
        ORDER BY se.observation_count DESC
    """, scene_id)
    return {"members": [dict(r) for r in members], "edges": [dict(r) for r in edges]}


async def fetch_recall(pool: asyncpg.Pool, word: str) -> dict:
    """Word recall: what does the system "see" when it hears this word?"""
    from nmem_sym_sensor.cooccurrence import cooc_store

    # Find sound units for this word
    units = await pool.fetch(
        "SELECT id FROM sound_units WHERE modality = 'voice' AND lower(stt_label) = $1",
        word.lower(),
    )

    # Sound path — query unified table for each matching unit
    sound_results = []
    for u in units:
        visuals = await cooc_store.query(
            u["id"], "voice", pool, target_modality="visual", min_count=1, limit=10,
        )
        for v in visuals:
            label_row = await pool.fetchrow("SELECT label FROM sensory_nodes WHERE id = $1", v["unit_id"])
            sound_results.append({
                "label": label_row["label"] if label_row else f"node:{v['unit_id']}",
                "count": v["count"],
                "myelinated": v["myelinated"],
            })

    # Sort by count descending, dedup
    sound_results.sort(key=lambda x: x["count"], reverse=True)

    # Text path (diagnostic — still uses old table)
    text_rows = await pool.fetch("""
        SELECT sn.label, wv.count
        FROM word_visual_cooccurrences wv
        JOIN sensory_nodes sn ON sn.id = wv.node_id
        WHERE lower(wv.word) = $1
        ORDER BY wv.count DESC LIMIT 10
    """, word.lower())

    return {
        "word": word,
        "sound_path": sound_results[:10],
        "text_path": [dict(r) for r in text_rows],
    }


async def fetch_top_words(pool: asyncpg.Pool, limit: int = 30) -> list[dict]:
    rows = await pool.fetch("""
        SELECT word, SUM(count) as total, COUNT(DISTINCT node_id) as nodes,
               MAX(count) as strongest
        FROM word_visual_cooccurrences
        GROUP BY word ORDER BY total DESC LIMIT $1
    """, limit)
    return [dict(r) for r in rows]


async def dream_word(pool: asyncpg.Pool, word: str) -> list[dict]:
    rows = await pool.fetch("""
        SELECT wvc.node_id, wvc.count, n.label, n.visual_embedding::text as emb
        FROM word_visual_cooccurrences wvc
        JOIN sensory_nodes n ON n.id = wvc.node_id
        WHERE wvc.word = $1 AND n.visual_embedding IS NOT NULL
        ORDER BY wvc.count DESC LIMIT 6
    """, word)
    return [dict(r) for r in rows]


def decode_embedding(emb_str: str) -> bytes | None:
    sess = get_decoder()
    if sess is None:
        return None
    emb = np.array([float(x) for x in emb_str.strip("[]").split(",")], dtype=np.float32).reshape(1, -1)
    result = sess.run(None, {"embedding": emb})[0][0]
    img = np.clip(np.transpose(result, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((128, 128), Image.NEAREST)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ── HTML Templates ───────────────────────────────────────

def render_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — Sensory Mind</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e0e0e0; --muted: #888; --accent: #6c9eff;
    --green: #4ade80; --yellow: #facc15; --red: #f87171; --purple: #a78bfa;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 20px; }}
h1 {{ color: var(--accent); margin-bottom: 8px; font-size: 1.5rem; }}
h2 {{ color: var(--text); margin: 24px 0 12px; font-size: 1.1rem; }}
h3 {{ color: var(--muted); margin: 16px 0 8px; font-size: 0.9rem; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
nav {{ margin-bottom: 24px; display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.85rem; }}
nav a {{ padding: 6px 12px; background: var(--surface); border-radius: 6px; border: 1px solid var(--border); }}
nav a:hover {{ background: var(--border); text-decoration: none; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin: 16px 0; }}
.stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: center; }}
.stat .num {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
.stat .label {{ font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.82rem; }}
th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.05em; }}
tr:hover {{ background: var(--surface); }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.68rem; font-weight: 600; }}
.badge-green {{ background: rgba(74,222,128,0.15); color: var(--green); }}
.badge-yellow {{ background: rgba(250,204,21,0.15); color: var(--yellow); }}
.badge-red {{ background: rgba(248,113,113,0.15); color: var(--red); }}
.badge-purple {{ background: rgba(167,139,250,0.15); color: var(--purple); }}
.badge-muted {{ background: rgba(136,136,136,0.15); color: var(--muted); }}
.dream-grid {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }}
.dream-tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; width: 160px; }}
.dream-tile img {{ border-radius: 4px; image-rendering: pixelated; width: 128px; height: 128px; }}
.dream-tile .info {{ font-size: 0.72rem; color: var(--muted); margin-top: 6px; }}
.bar {{ display: inline-block; height: 8px; background: var(--accent); border-radius: 4px; min-width: 2px; }}
.tier-pills {{ display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap; }}
.tier-pill {{ padding: 4px 10px; border-radius: 12px; font-size: 0.72rem; background: var(--surface); border: 1px solid var(--border); }}
.section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 16px 0; }}
.recall-result {{ margin: 8px 0; padding: 8px 12px; background: var(--surface); border-radius: 6px; border-left: 3px solid var(--accent); }}
input[type=text] {{ background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; width: 200px; }}
button {{ background: var(--accent); color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }}
button:hover {{ opacity: 0.85; }}
</style>
</head>
<body>
<h1>Sensory Mind Dashboard</h1>
<nav>
    <a href="/">Overview</a>
    <a href="/bindings">Bindings</a>
    <a href="/voice">Voice</a>
    <a href="/concepts">Concepts</a>
    <a href="/scenes">Scenes</a>
    <a href="/words">Words</a>
    <a href="/recall">Recall Test</a>
    <a href="/dream">Dream</a>
    <a href="/listen">Hot Mic</a>
    <a href="/probe">Image Probe</a>
</nav>
{body}
</body>
</html>"""


# ── FastAPI App ──────────────────────────────────────────

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response

app = FastAPI(title="Sensory Mind Dashboard")


@app.get("/", response_class=HTMLResponse)
async def overview():
    pool = await get_pool()
    s = await fetch_overview(pool)
    tiers_html = "".join(
        f'<span class="tier-pill">{k}: {v}</span>' for k, v in s.get("tiers", {}).items()
    )
    body = f"""
    <h2>Core Stats</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['nodes_active']}</div><div class="label">Active Nodes</div></div>
        <div class="stat"><div class="num">{s['nodes_archived']}</div><div class="label">Archived</div></div>
        <div class="stat"><div class="num">{s['edges']}</div><div class="label">Edges</div></div>
        <div class="stat"><div class="num">{s['clusters']}</div><div class="label">Clusters</div></div>
    </div>

    <h2>Sound-Visual Binding (Primary Path)</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['sv_pairs']}</div><div class="label">SV Pairs</div></div>
        <div class="stat"><div class="num">{s['sv_obs']}</div><div class="label">SV Observations</div></div>
        <div class="stat"><div class="num">{s['sv_myelinated']}</div><div class="label">Myelinated</div></div>
        <div class="stat"><div class="num">{s['sv_dormant']}</div><div class="label">Dormant</div></div>
        <div class="stat"><div class="num">{s['concept_links']}</div><div class="label">Concept Links</div></div>
        <div class="stat"><div class="num">{s['sequences']}</div><div class="label">Sequences</div></div>
    </div>

    <h2>Sound Units</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['sound_units']}</div><div class="label">Total</div></div>
        <div class="stat"><div class="num">{s['sound_voice']}</div><div class="label">Voice</div></div>
        <div class="stat"><div class="num">{s['sound_env']}</div><div class="label">Environmental</div></div>
        <div class="stat"><div class="num">{s['sound_labelled']}</div><div class="label">STT Labelled</div></div>
    </div>

    <h2>Word-Visual (Diagnostic Text Path)</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['wv_pairs']}</div><div class="label">WV Pairs</div></div>
        <div class="stat"><div class="num">{s['wv_obs']}</div><div class="label">WV Observations</div></div>
    </div>

    <h2>Scene Model (Chronoception)</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['scenes']}</div><div class="label">Scenes</div></div>
        <div class="stat"><div class="num">{s['scene_members']}</div><div class="label">Scene Members</div></div>
        <div class="stat"><div class="num">{s['scene_edges']}</div><div class="label">Spatial Edges</div></div>
        <div class="stat"><div class="num">{s['saccade_patterns']}</div><div class="label">Saccade Patterns</div></div>
    </div>

    <h2>Dreamstate Recombination</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['recombinations']}</div><div class="label">Total Recombinations</div></div>
        <div class="stat"><div class="num">{s['recombination_recent']}</div><div class="label">Last 24h</div></div>
        <div class="stat"><div class="num">{s['recombination_avg_coherence']}</div><div class="label">Avg Coherence (24h)</div></div>
        <div class="stat"><div class="num">{s['high_universality']}</div><div class="label">Universal Assocs</div></div>
        <div class="stat"><div class="num">{s['high_context_dependence']}</div><div class="label">Context-Bound Assocs</div></div>
    </div>

    <h2>Mental Rotation</h2>
    <div class="stats">
        <div class="stat"><div class="num">{s['rotations']}</div><div class="label">Total Tests</div></div>
        <div class="stat"><div class="num">{s['rotations_recent']}</div><div class="label">Last 24h</div></div>
        <div class="stat"><div class="num">{s['rotation_invariant']}</div><div class="label">Rotation-Invariant</div></div>
        <div class="stat"><div class="num">{s['rotation_sensitive']}</div><div class="label">Orientation-Sensitive</div></div>
        <div class="stat"><div class="num">{s['high_orient_independence']}</div><div class="label">Orient-Independent Assocs</div></div>
    </div>

    <h2>Memory Tiers</h2>
    <div class="tier-pills">{tiers_html}</div>
    """
    return render_page("Overview", body)


@app.get("/bindings", response_class=HTMLResponse)
async def bindings():
    pool = await get_pool()
    rows = await fetch_sound_visual_top(pool, 50)
    if not rows:
        return render_page("Bindings", "<p>No sound-visual bindings yet.</p>")
    max_count = max(r["count"] for r in rows) if rows else 1
    table_rows = ""
    for r in rows:
        bar_w = int(r["count"] / max_count * 120)
        label = r["stt_label"] or "?"
        mod_cls = "badge-green" if r["su_mod"] == "voice" else "badge-muted"
        myel = ' <span class="badge badge-purple">myelinated</span>' if r["myelinated"] else ""
        table_rows += f"""<tr>
            <td>{label} <span class="badge {mod_cls}">{r['su_mod']}</span></td>
            <td>{r['vis_label']}</td>
            <td>{r['count']}</td>
            <td><div class="bar" style="width:{bar_w}px"></div></td>
            <td>{r['dreamstate_challenges']}</td>
            <td>{r['self_play_confirmations']}</td>
            <td>{myel}</td>
        </tr>"""
    body = f"""
    <h2>Sound-Visual Bindings</h2>
    <p>Primary learning path: sound units bound to visual nodes through co-occurrence.</p>
    <table>
        <tr><th>Sound</th><th>Visual</th><th>Count</th><th></th><th>Challenges</th><th>Self-Play</th><th>Status</th></tr>
        {table_rows}
    </table>
    """
    return render_page("Bindings", body)


@app.get("/voice", response_class=HTMLResponse)
async def voice():
    pool = await get_pool()
    from nmem_sym_sensor.cooccurrence import cooc_store

    rows = await pool.fetch("""
        SELECT su.id, su.stt_label, su.modality, su.total_observations,
               su.member_count, su.stt_confidence, su.speaker_variance
        FROM sound_units su
        ORDER BY su.total_observations DESC
    """)

    table_rows = ""
    for r in rows:
        label = r["stt_label"] or "?"
        mod = r["modality"]
        mod_badge = "badge-blue" if mod == "voice" else "badge-green"
        conf = f'{r["stt_confidence"]:.0%}' if r["stt_confidence"] else "-"

        # Get visual bindings from unified table
        bindings = await cooc_store.query(r["id"], mod, pool, target_modality="visual")
        binding_labels = []
        for b in sorted(bindings, key=lambda x: x["count"], reverse=True)[:3]:
            node = await pool.fetchrow("SELECT label FROM sensory_nodes WHERE id = $1", b["unit_id"])
            if node:
                binding_labels.append(f'{node["label"]} ({b["count"]}x)')

        bindings_str = ", ".join(binding_labels) if binding_labels else "<em>none</em>"

        table_rows += f"""<tr>
            <td>#{r['id']}</td>
            <td>{label}</td>
            <td><span class="badge {mod_badge}">{mod}</span></td>
            <td>{r['total_observations']}</td>
            <td>{r['member_count']}</td>
            <td>{conf}</td>
            <td>{bindings_str}</td>
        </tr>"""

    body = f"""
    <h2>Sound Units ({len(rows)} total)</h2>
    <p>All discovered sound units — voice syllables and environmental sounds.</p>
    <table>
        <tr><th>ID</th><th>Label</th><th>Type</th><th>Observations</th><th>Members</th><th>STT Conf</th><th>Visual Bindings</th></tr>
        {table_rows}
    </table>
    """
    return render_page("Voice", body)


@app.get("/concepts", response_class=HTMLResponse)
async def concepts():
    pool = await get_pool()
    rows = await fetch_concept_links(pool)
    if not rows:
        return render_page("Concepts", "<p>No concept links discovered yet.</p>")
    table_rows = ""
    for r in rows:
        la = r["label_a"] or "?"
        lb = r["label_b"] or "?"
        table_rows += f"""<tr>
            <td>{la}</td>
            <td>{lb}</td>
            <td>{r['overlap_score']:.3f}</td>
            <td>{r['strength']:.2f}</td>
        </tr>"""
    body = f"""
    <h2>Concept Links</h2>
    <p>Sound units that co-occur with the same visual nodes — they mean the same thing.</p>
    <table>
        <tr><th>Sound A</th><th>Sound B</th><th>Visual Overlap</th><th>Strength</th></tr>
        {table_rows}
    </table>
    """
    return render_page("Concepts", body)


@app.get("/scenes", response_class=HTMLResponse)
async def scenes():
    pool = await get_pool()
    rows = await fetch_scenes(pool)
    if not rows:
        return render_page("Scenes", "<p>No scenes recorded yet.</p>")
    table_rows = ""
    for r in rows:
        last = r["last_seen"].strftime("%Y-%m-%d %H:%M") if r["last_seen"] else ""
        table_rows += f"""<tr>
            <td><a href="/scenes/{r['id']}">Scene #{r['id']}</a></td>
            <td>{r['visit_count']}</td>
            <td>{r['members']}</td>
            <td>{r['spatial_edges']}</td>
            <td>{last}</td>
        </tr>"""
    body = f"""
    <h2>Scene Snapshots (Chronoception)</h2>
    <p>Recognised environments with their associated clusters and spatial relationships.</p>
    <table>
        <tr><th>Scene</th><th>Visits</th><th>Clusters</th><th>Spatial Edges</th><th>Last Seen</th></tr>
        {table_rows}
    </table>
    """
    return render_page("Scenes", body)


@app.get("/scenes/{scene_id}", response_class=HTMLResponse)
async def scene_detail(scene_id: int):
    pool = await get_pool()
    data = await fetch_scene_detail(pool, scene_id)

    members_html = ""
    for m in data["members"]:
        label = m["grounded_label"] or f"cluster #{m['cluster_id']}"
        members_html += f"<tr><td>{m['cluster_id']}</td><td>{label}</td><td>{m['observation_count']}</td></tr>"

    edges_html = ""
    for e in data["edges"]:
        la = e["label_a"] or f"#{e['cluster_a_id']}"
        lb = e["label_b"] or f"#{e['cluster_b_id']}"
        edges_html += f"""<tr>
            <td>{la}</td>
            <td><span class="badge badge-green">{e['relation']}</span></td>
            <td>{lb}</td>
            <td>{e['confidence']:.2f}</td>
            <td>{e['observation_count']}</td>
        </tr>"""

    body = f"""
    <h2>Scene #{scene_id}</h2>
    <h3>Cluster Members ({len(data['members'])})</h3>
    <table>
        <tr><th>Cluster ID</th><th>Label</th><th>Observations</th></tr>
        {members_html or '<tr><td colspan="3">No members</td></tr>'}
    </table>
    <h3>Spatial Graph ({len(data['edges'])} allocentric edges)</h3>
    <table>
        <tr><th>Object A</th><th>Relation</th><th>Object B</th><th>Confidence</th><th>Observations</th></tr>
        {edges_html or '<tr><td colspan="5">No spatial relations</td></tr>'}
    </table>
    """
    return render_page(f"Scene #{scene_id}", body)


@app.get("/words", response_class=HTMLResponse)
async def words():
    pool = await get_pool()
    rows = await fetch_top_words(pool, 50)
    if not rows:
        return render_page("Words", "<p>No word-visual co-occurrences yet.</p>")
    max_total = max(r["total"] for r in rows) if rows else 1
    table_rows = ""
    for r in rows:
        bar_w = int(r["total"] / max_total * 150)
        table_rows += f"""<tr>
            <td><a href="/recall?word={r['word']}">{r['word']}</a></td>
            <td>{r['total']}</td>
            <td>{r['nodes']}</td>
            <td>{r['strongest']}</td>
            <td><div class="bar" style="width:{bar_w}px"></div></td>
        </tr>"""
    body = f"""
    <h2>Word-Visual Co-occurrences (Diagnostic Text Path)</h2>
    <table>
        <tr><th>Word</th><th>Total</th><th>Nodes</th><th>Strongest</th><th>Distribution</th></tr>
        {table_rows}
    </table>
    """
    return render_page("Words", body)


@app.get("/recall", response_class=HTMLResponse)
async def recall(word: str = Query(default="")):
    pool = await get_pool()

    # Available words from sound units
    sound_words = await pool.fetch("""
        SELECT DISTINCT stt_label FROM sound_units
        WHERE stt_label IS NOT NULL AND modality = 'voice'
        ORDER BY stt_label
    """)
    word_links = " ".join(
        f'<a href="/recall?word={w["stt_label"]}">{w["stt_label"]}</a>'
        for w in sound_words
    )

    if not word:
        body = f"""
        <h2>Recall Test</h2>
        <p>"I hear this word — what do I see?"</p>
        <form action="/recall" method="get" style="margin:16px 0">
            <input type="text" name="word" placeholder="Type a word..." autofocus>
            <button type="submit">Test</button>
        </form>
        <h3>Known words:</h3>
        <div style="line-height:2.2">{word_links}</div>
        """
        return render_page("Recall Test", body)

    data = await fetch_recall(pool, word)

    sound_html = ""
    if data["sound_path"]:
        total = sum(r["count"] for r in data["sound_path"])
        for r in data["sound_path"]:
            pct = r["count"] / max(1, total) * 100
            myel = ' <span class="badge badge-purple">M</span>' if r["myelinated"] else ""
            bar_w = int(pct * 1.5)
            sound_html += f"""<div class="recall-result">
                {r['label']} — {r['count']}x ({pct:.0f}%)
                <div class="bar" style="width:{bar_w}px"></div>{myel}
            </div>"""
    else:
        sound_html = '<p style="color:var(--muted)">No sound-visual binding for this word.</p>'

    text_html = ""
    if data["text_path"]:
        for r in data["text_path"][:5]:
            text_html += f'<div class="recall-result">{r["label"]} — {r["count"]}x</div>'
    else:
        text_html = '<p style="color:var(--muted)">No text-visual binding.</p>'

    body = f"""
    <h2>Recall: "{word}"</h2>

    <div class="section">
        <h3>Sound Path (what the ears learned)</h3>
        {sound_html}
    </div>

    <div class="section">
        <h3>Text Path (diagnostic — Whisper transcription)</h3>
        {text_html}
    </div>

    <form action="/recall" method="get" style="margin:16px 0">
        <input type="text" name="word" placeholder="Try another..." value="">
        <button type="submit">Test</button>
    </form>
    <h3>Known words:</h3>
    <div style="line-height:2.2">{word_links}</div>
    """
    return render_page(f"Recall: {word}", body)


@app.get("/dream", response_class=HTMLResponse)
async def dream(word: str = Query(default="")):
    pool = await get_pool()
    top_words = await fetch_top_words(pool, 20)
    word_links = " ".join(
        f'<a href="/dream?word={w["word"]}">{w["word"]}</a>'
        for w in top_words
    )

    if not word:
        body = f"""
        <h2>Dream Inspector — Mind's Eye</h2>
        <p>Select a word to see what the system "sees" when it hears it:</p>
        <div style="margin:16px 0; line-height:2">{word_links}</div>
        """
        return render_page("Dream", body)

    rows = await dream_word(pool, word)
    if not rows:
        body = f"""
        <h2>Dream: "{word}"</h2>
        <p>No visual associations for "{word}" yet.</p>
        <div style="margin:16px 0; line-height:2">{word_links}</div>
        """
        return render_page("Dream", body)

    tiles = ""
    for r in rows:
        if r["emb"]:
            tiles += f"""<div class="dream-tile">
                <img src="/dream/image/{r['node_id']}" alt="{r['label']}">
                <div class="info">{r['label']}<br>co-occur: {r['count']}</div>
            </div>"""

    body = f"""
    <h2>Dream: "{word}"</h2>
    <p>The system's visual associations when it hears "{word}":</p>
    <div class="dream-grid">{tiles}</div>
    <h2>Try another word</h2>
    <div style="line-height:2">{word_links}</div>
    """
    return render_page(f"Dream: {word}", body)


@app.get("/dream/image/{node_id}")
async def dream_image(node_id: int):
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT visual_embedding::text as emb FROM sensory_nodes WHERE id = $1",
        node_id,
    )
    if not row or not row["emb"]:
        return Response(content=b"", media_type="image/png", status_code=404)
    png = decode_embedding(row["emb"])
    if png is None:
        return Response(content=b"No decoder", status_code=500)
    return Response(content=png, media_type="image/png")


# ── Hot Mic: speak and see what activates ────────────────

@app.get("/listen", response_class=HTMLResponse)
async def listen():
    body = """
    <h2>Hot Mic — Speak and See</h2>
    <p style="color:var(--muted);font-size:0.85rem;">
        Record your voice. The system encodes it through TABULA2, finds the closest
        sound units by embedding similarity, then follows co-occurrence chains to
        show what visual neurons activate. No text labels involved.
    </p>
    <div class="section">
        <button id="recordBtn" onclick="toggleRecord()">Hold to Record</button>
        <span id="status" style="margin-left:12px;color:var(--muted);font-size:0.85rem;"></span>
    </div>
    <div id="results"></div>

    <script>
    let mediaRecorder, chunks = [], recording = false;

    async function toggleRecord() {
        const btn = document.getElementById('recordBtn');
        const status = document.getElementById('status');

        if (!recording) {
            const stream = await navigator.mediaDevices.getUserMedia({audio: true});
            mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
            chunks = [];
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(t => t.stop());
                const blob = new Blob(chunks, {type: 'audio/webm'});
                status.textContent = 'Processing...';
                const formData = new FormData();
                formData.append('audio', blob, 'recording.webm');
                try {
                    const resp = await fetch('/listen/process', {method: 'POST', body: formData});
                    const data = await resp.json();
                    renderResults(data);
                    status.textContent = '';
                } catch(e) {
                    status.textContent = 'Error: ' + e.message;
                }
            };
            mediaRecorder.start();
            recording = true;
            btn.textContent = 'Stop Recording';
            btn.style.background = 'var(--red)';
            status.textContent = 'Listening...';
        } else {
            mediaRecorder.stop();
            recording = false;
            btn.textContent = 'Hold to Record';
            btn.style.background = '';
        }
    }

    function renderResults(data) {
        const div = document.getElementById('results');
        if (!data.matched_units || data.matched_units.length === 0) {
            div.innerHTML = '<p style="color:var(--muted);margin-top:16px;">No matching sound units found.</p>';
            return;
        }

        let html = '<h3>Matched Sound Units (by embedding similarity)</h3>';
        html += '<table><tr><th>Unit ID</th><th>Similarity</th><th>Observations</th><th>STT Label (diagnostic)</th></tr>';
        for (const u of data.matched_units) {
            html += `<tr><td>${u.id}</td><td>${u.similarity.toFixed(3)}</td><td>${u.observations}</td>` +
                    `<td style="color:var(--muted)">${u.stt_label || '-'}</td></tr>`;
        }
        html += '</table>';

        if (data.visual_recall && data.visual_recall.length > 0) {
            html += '<h3>Visual Recall (what it "sees" when it hears this)</h3>';
            html += '<div class="dream-grid">';
            for (const v of data.visual_recall) {
                html += `<div class="dream-tile">` +
                        `<img src="/dream/image/${v.node_id}" alt="node ${v.node_id}">` +
                        `<div class="info">node #${v.node_id}<br>strength: ${v.strength.toFixed(2)}<br>` +
                        `reinf: ${v.reinforcement_count}</div></div>`;
            }
            html += '</div>';
        } else {
            html += '<p style="color:var(--muted);margin-top:12px;">No visual associations found for these sound units.</p>';
        }
        div.innerHTML = html;
    }
    </script>
    """
    return render_page("Hot Mic", body)


@app.post("/listen/process")
async def listen_process():
    """Process uploaded audio: TABULA2 encode → match sound units → visual recall."""

    # We need to handle this differently with FastAPI
    pass


# Override with proper file upload
from fastapi import File, UploadFile


@app.post("/listen/process")
async def listen_process_upload(audio: UploadFile = File(...)):
    """Process uploaded audio: TABULA2 encode → match sound units → visual recall."""
    import subprocess
    import tempfile

    pool = await get_pool()

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Convert webm to wav using ffmpeg
    wav_path = tmp_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=10,
        )
    except Exception as e:
        os.unlink(tmp_path)
        return {"error": f"ffmpeg conversion failed: {e}"}

    # Load wav
    try:
        import soundfile as sf
        waveform, sr = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
    except Exception as e:
        return {"error": f"Audio load failed: {e}"}
    finally:
        os.unlink(tmp_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    if len(waveform) < 1600:  # < 0.1s
        return {"error": "Recording too short"}

    # TABULA2 encode
    try:

        t2 = _get_tabula2()
        if t2 is None:
            return {"error": "TABULA2 not available"}

        output = t2.process(waveform, 16000)
        voiced = output.voice_embeddings[output.vad_mask]
        if len(voiced) == 0:
            return {"error": "No voice detected in recording"}

        # Mean voice embedding
        voice_emb = voiced.mean(axis=0)
        norm = np.linalg.norm(voice_emb)
        if norm > 0:
            voice_emb = voice_emb / norm
    except Exception as e:
        return {"error": f"TABULA2 processing failed: {e}"}

    # Find closest sound units by cosine similarity
    emb_str = "[" + ",".join(str(float(x)) for x in voice_emb) + "]"
    matched = await pool.fetch("""
        SELECT id, stt_label, total_observations,
               1 - (centroid <=> $1::vector) as similarity
        FROM sound_units
        WHERE modality = 'voice' AND centroid IS NOT NULL
        ORDER BY centroid <=> $1::vector
        LIMIT 5
    """, emb_str)

    matched_units = [
        {"id": r["id"], "similarity": float(r["similarity"]),
         "observations": r["total_observations"],
         "stt_label": r["stt_label"]}
        for r in matched if r["similarity"] > 0.3
    ]

    # Visual recall: for each matched unit, find visual co-occurrences
    from nmem_sym_sensor.cooccurrence import cooc_store
    visual_recall = []
    seen_nodes = set()
    for u in matched_units[:3]:
        visuals = await cooc_store.query(
            u["id"], "voice", pool,
            target_modality="visual", min_count=1, limit=5,
        )
        for v in visuals:
            if v["unit_id"] not in seen_nodes:
                seen_nodes.add(v["unit_id"])
                visual_recall.append({
                    "node_id": v["unit_id"],
                    "strength": float(v["strength"]),
                    "reinforcement_count": v.get("reinforcement_count", v.get("count", 0)),
                })

    return {
        "matched_units": matched_units,
        "visual_recall": visual_recall[:12],
        "voice_duration_s": round(len(waveform) / 16000, 2),
        "voiced_frames": int(output.vad_mask.sum()),
    }


# Lazy TABULA2 singleton for dashboard
_dashboard_tabula2 = None

def _get_tabula2():
    global _dashboard_tabula2
    if _dashboard_tabula2 is None:
        try:
            from nmem_sym_sensor import config
            from nmem_sym_sensor.tabula2 import TABULA2Disentangler
            _dashboard_tabula2 = TABULA2Disentangler(
                checkpoint_path=config.TABULA2_CHECKPOINT,
                decoder_path=config.TABULA2_DECODER_CHECKPOINT,
                vad_threshold=config.TABULA2_VAD_THRESHOLD,
            )
        except Exception as e:
            log.warning("TABULA2 not available for dashboard: %s", e)
    return _dashboard_tabula2


# ── Image Probe: show an image and see what activates ────

@app.get("/probe", response_class=HTMLResponse)
async def probe():
    body = """
    <h2>Image Probe — Show and Recognise</h2>
    <p style="color:var(--muted);font-size:0.85rem;">
        Upload an image. The system runs edge analysis + dual-stream encoding,
        matches against existing neurons, and shows what activates.
        Decoded neuron images show what the system "recognises".
    </p>
    <div class="section">
        <input type="file" id="imageFile" accept="image/*">
        <button onclick="probeImage()" style="margin-left:8px;">Probe</button>
        <span id="probeStatus" style="margin-left:12px;color:var(--muted);font-size:0.85rem;"></span>
    </div>
    <div style="display:flex;gap:24px;margin-top:16px;">
        <div id="inputPreview" style="flex:0 0 256px;"></div>
        <div id="probeResults" style="flex:1;"></div>
    </div>

    <script>
    async function probeImage() {
        const file = document.getElementById('imageFile').files[0];
        if (!file) return;
        const status = document.getElementById('probeStatus');
        const preview = document.getElementById('inputPreview');

        // Show input preview
        const reader = new FileReader();
        reader.onload = e => {
            preview.innerHTML = '<h3>Input</h3><img src="' + e.target.result +
                '" style="max-width:256px;max-height:256px;border-radius:8px;">';
        };
        reader.readAsDataURL(file);

        status.textContent = 'Processing...';
        const formData = new FormData();
        formData.append('image', file);

        try {
            const resp = await fetch('/probe/process', {method: 'POST', body: formData});
            const data = await resp.json();
            renderProbeResults(data);
            status.textContent = '';
        } catch(e) {
            status.textContent = 'Error: ' + e.message;
        }
    }

    function renderProbeResults(data) {
        const div = document.getElementById('probeResults');
        if (data.error) {
            div.innerHTML = '<p style="color:var(--red);">' + data.error + '</p>';
            return;
        }

        let html = '<h3>Activated Neurons (' + data.activated.length + ')</h3>';
        if (data.activated.length > 0) {
            html += '<div class="dream-grid">';
            for (const n of data.activated) {
                const badge = n.is_new ? '<span class="badge badge-green">NEW</span>' :
                    '<span class="badge badge-purple">KNOWN</span>';
                html += '<div class="dream-tile">' +
                    '<img src="/dream/image/' + n.node_id + '" alt="node">' +
                    '<div class="info">' + badge + ' #' + n.node_id +
                    '<br>type: ' + n.node_type +
                    '<br>sim: ' + (n.similarity ? n.similarity.toFixed(3) : '-') +
                    '<br>obs: ' + n.observations + '</div></div>';
            }
            html += '</div>';
        }

        if (data.co_activated_sounds && data.co_activated_sounds.length > 0) {
            html += '<h3>Associated Sounds</h3>';
            html += '<table><tr><th>Unit ID</th><th>Strength</th><th>STT (diagnostic)</th></tr>';
            for (const s of data.co_activated_sounds) {
                html += '<tr><td>' + s.unit_id + '</td><td>' + s.strength.toFixed(2) +
                    '</td><td style="color:var(--muted)">' + (s.stt_label || '-') + '</td></tr>';
            }
            html += '</table>';
        }

        html += '<div style="margin-top:12px;color:var(--muted);font-size:0.75rem;">' +
            'Primitives detected: ' + data.primitives_count +
            ' | Processing: ' + data.processing_ms + 'ms</div>';
        div.innerHTML = html;
    }
    </script>
    """
    return render_page("Image Probe", body)


@app.post("/probe/process")
async def probe_process(image: UploadFile = File(...)):
    """Process uploaded image: edge analysis → match neurons → decode back."""
    import time as _time
    t0 = _time.monotonic()

    pool = await get_pool()

    # Load image
    content = await image.read()
    import cv2
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    # Run edge analysis
    from nmem_sym_sensor.visual import analyze_frame
    analysis = analyze_frame(img, "probe")

    activated = []
    for prim in analysis.primitives:
        if prim.embedding is None:
            continue

        emb_str = "[" + ",".join(str(float(x)) for x in prim.embedding) + "]"
        emb_col = "visual_embedding"

        # Find nearest neuron
        match = await pool.fetchrow(f"""
            SELECT id, node_type, observation_count,
                   1 - ({emb_col} <=> $1::vector) as similarity
            FROM sensory_nodes
            WHERE node_type = $2 AND modality = 'visual'
              AND {emb_col} IS NOT NULL
            ORDER BY {emb_col} <=> $1::vector
            LIMIT 1
        """, emb_str, prim.node_type)

        if match:
            activated.append({
                "node_id": match["id"],
                "node_type": match["node_type"],
                "similarity": float(match["similarity"]),
                "observations": match["observation_count"],
                "is_new": match["similarity"] < 0.95,
            })

    # Deduplicate by node_id, keep highest similarity
    seen = {}
    for a in activated:
        nid = a["node_id"]
        if nid not in seen or a["similarity"] > seen[nid]["similarity"]:
            seen[nid] = a
    activated = sorted(seen.values(), key=lambda x: -x["similarity"])

    # For activated visual neurons, find associated sounds
    from nmem_sym_sensor.cooccurrence import cooc_store
    co_sounds = []
    seen_sounds = set()
    for a in activated[:5]:
        sounds = await cooc_store.query(
            a["node_id"], "visual", pool,
            target_modality="voice", min_count=1, limit=3,
        )
        for s in sounds:
            if s["unit_id"] not in seen_sounds:
                seen_sounds.add(s["unit_id"])
                label = await pool.fetchval(
                    "SELECT stt_label FROM sound_units WHERE id = $1", s["unit_id"])
                co_sounds.append({
                    "unit_id": s["unit_id"],
                    "strength": float(s["strength"]),
                    "stt_label": label,
                })

    elapsed_ms = int((_time.monotonic() - t0) * 1000)

    return {
        "activated": activated[:20],
        "co_activated_sounds": co_sounds[:10],
        "primitives_count": len(analysis.primitives),
        "processing_ms": elapsed_ms,
    }


# ── CLI entry point ──────────────────────────────────────

def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Sensory Mind Dashboard")
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Sensory Mind Dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
