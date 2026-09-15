"""Embedder-space partition discipline: neurons only ever match WITHIN one
``embedder_id`` vector space.

A tag matches the same tag (and legacy NULL matches NULL); a differently-tagged
neuron is never a valid neighbour even if cosine-identical, because vectors from
different embedders (JEPA on/off, model swaps) aren't comparable. Enforced by the
``embedder_id IS NOT DISTINCT FROM $4`` clause in ``activate_or_create_neuron``.
"""
from __future__ import annotations

import pytest
from nmem.migrate import MigrationRunner

from nmem_sym_sensor import graph
from tests.conftest import MIGRATIONS_DIR

pytestmark = pytest.mark.integration


async def _migrate(pool) -> None:
    await MigrationRunner(
        project="nmem-sym-sensor", migrations_dir=MIGRATIONS_DIR, pool=pool
    ).run()


def _unit_vec() -> list[float]:
    v = [0.0] * 512
    v[0] = 1.0
    return v


async def test_same_embedding_different_space_does_not_match(clean_pool):
    await _migrate(clean_pool)
    emb = _unit_vec()

    id_a, new_a = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id="spaceA"
    )
    assert new_a is True

    # Cosine-identical, but a DIFFERENT embedder space → must create, not activate.
    id_b, new_b = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id="spaceB"
    )
    assert new_b is True
    assert id_b != id_a


async def test_same_embedding_same_space_activates(clean_pool):
    await _migrate(clean_pool)
    emb = _unit_vec()

    id_a, new_a = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id="spaceA"
    )
    assert new_a is True

    # Same space, identical embedding (sim=1.0 ≥ threshold) → activate the SAME neuron.
    id_again, new_again = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id="spaceA"
    )
    assert new_again is False
    assert id_again == id_a

    count = await clean_pool.fetchval(
        "SELECT observation_count FROM sensory_nodes WHERE id = $1", id_a
    )
    assert count == 2  # created (1) then activated (+1)


async def test_null_embedder_id_only_matches_null(clean_pool):
    await _migrate(clean_pool)
    emb = _unit_vec()

    id_null, _ = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id=None
    )
    # A tagged query with the same vector must not match the legacy-NULL neuron.
    id_tagged, new_tagged = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id="spaceA"
    )
    assert new_tagged is True
    assert id_tagged != id_null

    # And a NULL query re-activates the NULL neuron.
    id_null2, new_null2 = await graph.activate_or_create_neuron(
        clean_pool, emb, "visual", "object", "thing", embedder_id=None
    )
    assert new_null2 is False
    assert id_null2 == id_null
