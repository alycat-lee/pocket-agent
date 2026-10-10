from __future__ import annotations

from pathlib import Path

from pocket_agent.events import EventStore


def test_events_pagination_from_tail(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events")
    for i in range(1, 11):
        store.append("default", "step.start", {"step": i})

    page = store.list_before("default", before=None, limit=3)
    ids = [e["id"] for e in page["events"]]
    assert ids == [8, 9, 10]
    assert page["next_before"] == 8

    page2 = store.list_before("default", before=page["next_before"], limit=3)
    assert [e["id"] for e in page2["events"]] == [5, 6, 7]
    assert page2["next_before"] == 5

    # 边界：before=1 → 空页
    empty = store.list_before("default", before=1, limit=5)
    assert empty["events"] == []
    assert empty["next_before"] is None

    # limit=0
    zero = store.list_before("default", before=None, limit=0)
    assert zero["events"] == []


def test_events_order_and_export(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events")
    store.append("s1", "step.start", {"step": 1})
    store.append("s1", "answer.done", {"answer": "ok"})
    all_ev = store.export_all("s1")
    assert len(all_ev) == 2
    assert all_ev[0]["id"] < all_ev[1]["id"]
    assert all_ev[0]["event"] == "step.start"
