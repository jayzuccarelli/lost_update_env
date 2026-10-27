# SPDX-License-Identifier: BSD-3-Clause

from lost_update_env.server.grader import grade

START = {"v": 1, "api": {"limit": 10, "retry": 100, "timeout": 500}, "tags": ["a", "b"], "flag": False}


def test_merged_result_scores_one_and_list_order_is_ignored():
    rival = [{"op": "set", "path": ["flag"], "value": True}]
    intent = [{"op": "append", "path": ["tags"], "item": "c"}]
    final = {**START, "flag": True, "tags": ["c", "a", "b"]}
    assert grade(START, rival, intent, final)["reward"] == 1.0


def test_lost_rival_write_zeroes_preserve():
    rival = [{"op": "set", "path": ["flag"], "value": True}]
    intent = [{"op": "set", "path": ["api", "limit"], "value": 20}]
    final = {**START, "api": {**START["api"], "limit": 20}}
    out = grade(START, rival, intent, final)
    assert out["task"] == 1.0 and out["rivals_preserved"] == 0.0 and out["reward"] == 0.0


def test_partial_credit_across_two_rival_writes():
    rival = [
        {"op": "set", "path": ["flag"], "value": True},
        {"op": "incr", "path": ["v"], "by": 1},
    ]
    intent = [{"op": "set", "path": ["api", "limit"], "value": 20}]
    final = {**START, "flag": True, "api": {**START["api"], "limit": 20}}
    assert grade(START, rival, intent, final)["reward"] == 0.5


def test_derived_value_must_use_the_rivals_new_input():
    rival = [{"op": "set", "path": ["api", "retry"], "value": 300}]
    intent = [{"op": "derive", "path": ["api", "timeout"], "source": ["api", "retry"], "factor": 3}]
    stale = {**START, "api": {**START["api"], "retry": 300, "timeout": 300}}
    good = {**START, "api": {**START["api"], "retry": 300, "timeout": 900}}
    assert grade(START, rival, intent, stale)["task"] == 0.0
    assert grade(START, rival, intent, good)["reward"] == 1.0


def test_rename_requires_setting_the_new_key():
    rival = [{"op": "rename", "path": ["api", "limit"], "to": "max_limit"}]
    intent = [{"op": "set_any", "paths": [["api", "limit"], ["api", "max_limit"]], "value": 99, "label": "limit"}]
    recreated_old = {**START, "api": {"limit": 99, "retry": 100, "timeout": 500}}
    merged = {**START, "api": {"max_limit": 99, "retry": 100, "timeout": 500}}
    assert grade(START, rival, intent, recreated_old)["reward"] == 0.0
    assert grade(START, rival, intent, merged)["reward"] == 1.0


def test_missing_document_scores_zero():
    intent = [{"op": "set", "path": ["flag"], "value": True}]
    assert grade(START, [], intent, None)["reward"] == 0.0


def test_integer_valued_floats_match_ints():
    intent = [{"op": "set", "path": ["v"], "value": 2}]
    assert grade(START, [], intent, {**START, "v": 2.0})["reward"] == 1.0
