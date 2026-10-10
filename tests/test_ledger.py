from __future__ import annotations

from pocket_agent.ledger import CallLedger, SemanticDeduper, canonical_json


def test_canonical_json_sorts_keys_and_is_compact() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json({"city": "北京"}) == '{"city":"北京"}'


def test_fingerprint_stable_under_key_order() -> None:
    ledger = CallLedger()
    fp1 = ledger.fingerprint("get_weather", {"city": "北京", "unit": "c"})
    fp2 = ledger.fingerprint("get_weather", {"unit": "c", "city": "北京"})
    assert fp1 == fp2
    assert len(fp1) == 32


def test_fingerprint_differs_for_different_args_or_tools() -> None:
    ledger = CallLedger()
    a = ledger.fingerprint("get_weather", {"city": "北京"})
    b = ledger.fingerprint("get_weather", {"city": "上海"})
    c = ledger.fingerprint("calculator", {"city": "北京"})
    assert a != b
    assert a != c


def test_check_miss_then_save_then_hit() -> None:
    ledger = CallLedger()
    args = {"city": "上海"}
    assert ledger.check("get_weather", args) is None
    assert ledger.hit_count == 0
    assert ledger.real_call_count == 0

    ledger.save("get_weather", args, "多云 18℃")
    assert ledger.real_call_count == 1
    assert ledger.hit_count == 0

    cached = ledger.check("get_weather", {"city": "上海"})
    assert cached == "多云 18℃"
    assert ledger.hit_count == 1
    assert ledger.real_call_count == 1

    # 键序不同仍命中
    cached2 = ledger.check("get_weather", {"city": "上海"})
    assert cached2 == "多云 18℃"
    assert ledger.hit_count == 2


def test_save_overwrites_same_fingerprint() -> None:
    ledger = CallLedger()
    ledger.save("t", {"x": 1}, "first")
    ledger.save("t", {"x": 1}, "second")
    assert ledger.real_call_count == 2
    assert ledger.check("t", {"x": 1}) == "second"


def test_semantic_deduper_detects_near_duplicate() -> None:
    deduper = SemanticDeduper()
    history = [{"city": "北京天气"}]
    assert deduper.is_duplicate({"city": "北京 天气"}, history, threshold=0.95) is True
    assert deduper.is_duplicate({"city": "上海"}, history, threshold=0.95) is False


def test_semantic_deduper_normalize_case_and_whitespace() -> None:
    deduper = SemanticDeduper()
    history = [{"q": "Hello World"}]
    assert deduper.is_duplicate({"q": "hello   world"}, history) is True
