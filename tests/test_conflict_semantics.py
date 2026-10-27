# SPDX-License-Identifier: BSD-3-Clause

import json

from lost_update_env.server.game import LostUpdateGame
from lost_update_env.server.tasks import generate


def fresh(tier="T0", seed=0, **kw):
    ep = generate(seed, tier)
    return ep, LostUpdateGame(ep, **kw)


def test_update_with_current_revision_succeeds_and_bumps_revision():
    ep, game = fresh()
    doc = game.read(ep.uri)
    result = game.update(ep.uri, doc["content"], doc["revision"])
    assert result["ok"] and result["revision"] != doc["revision"]
    assert game.stat(ep.uri)["revision"] == result["revision"]


def test_update_with_stale_revision_is_412_with_current_revision():
    ep, game = fresh()
    doc = game.read(ep.uri)
    first = game.update(ep.uri, doc["content"], doc["revision"])
    second = game.update(ep.uri, doc["content"], doc["revision"])
    assert second["ok"] is False and second["status"] == 412
    assert second["error"] == "precondition_failed"
    assert second["current_revision"] == first["revision"]
    assert game.stats["conflicts_seen"] == 1


def test_failed_update_writes_nothing():
    ep, game = fresh("T1")
    doc = game.read(ep.uri)
    before_rival = json.loads(doc["content"])
    result = game.update(ep.uri, json.dumps({"x": 1}), doc["revision"])
    assert result["status"] == 412
    after = json.loads(game.read(ep.uri)["content"])
    assert after != before_rival  # the rival's write is there
    assert "x" not in after  # ours is not


def test_rival_lands_before_each_write_attempt_up_to_schedule():
    for seed in range(30):
        ep, game = fresh("T2", seed)
        k = len(ep.rival)
        assert 2 <= k <= 4
        for i in range(k):
            doc = game.read(ep.uri)
            assert game.update(ep.uri, doc["content"], doc["revision"])["status"] == 412
            assert len(game.rival_landed) == i + 1
        doc = game.read(ep.uri)
        assert game.update(ep.uri, doc["content"], doc["revision"])["ok"]
        assert len(game.rival_landed) == k


def test_no_rival_on_t0_and_reads_do_not_trigger_rival():
    ep, game = fresh("T0")
    rev = game.stat(ep.uri)["revision"]
    for _ in range(3):
        assert game.read(ep.uri)["revision"] == rev
    ep, game = fresh("T1")
    rev = game.stat(ep.uri)["revision"]
    for _ in range(3):
        assert game.read(ep.uri)["revision"] == rev
    assert game.rival_landed == []


def test_write_is_unconditional_and_counts_blind_overwrites():
    ep, game = fresh("T1")
    doc = game.read(ep.uri)
    assert game.write(ep.uri, doc["content"])["ok"]  # rival landed first, we clobbered it
    assert game.stats["blind_overwrites"] == 1
    doc = game.read(ep.uri)
    assert game.write(ep.uri, doc["content"])["ok"]
    assert game.stats["blind_overwrites"] == 1  # seen revision, not blind


def test_create_and_delete():
    ep, game = fresh()
    assert game.create(ep.uri, "{}")["status"] == 409
    assert game.create("mem://scratch.json", '{"a": 1}')["ok"]
    rev = game.stat("mem://scratch.json")["revision"]
    assert game.delete("mem://scratch.json", "nope")["status"] == 412
    assert game.delete("mem://scratch.json", rev)["ok"]
    assert game.read("mem://scratch.json")["status"] == 404


def test_invalid_content_is_400():
    ep, game = fresh()
    doc = game.read(ep.uri)
    assert game.update(ep.uri, "not json", doc["revision"])["status"] == 400
    assert game.update(ep.uri, "[1, 2]", doc["revision"])["status"] == 400


def test_deleting_the_document_scores_zero():
    ep, game = fresh("T0")
    doc = game.read(ep.uri)
    game.delete(ep.uri, doc["revision"])
    game.submit()
    assert game.reward == 0.0
