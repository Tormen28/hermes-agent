"""The durable detached-admission queue (B2): a result that arrives while the destination turn is
live waits for the turn loop's next drain instead of raising, and lands exactly once."""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from hermes_state_detached_results import QueuedDetachedAdmission
from hermes_state_errors import CompressionSessionClosedError


@pytest.fixture
def db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    try:
        yield db
    finally:
        db.close()


def _hold_turn(db, holder="client-turn"):
    assert db.acquire_session_turn_lease("s1", holder, wait_seconds=0)


def _deliver(db, delegation_id="d1", content="DELIVERY"):
    return db.append_delegation_delivery(
        "s1", content, {"delegation_id": delegation_id})


def test_no_live_turn_takes_the_direct_path(db):
    """Without a live turn nothing changes: the row lands in the transcript and the queue is empty."""
    message_id = _deliver(db)
    assert isinstance(message_id, int)
    rows = db.get_messages("s1")
    assert len(rows) == 1 and rows[0]["content"] == "DELIVERY"
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is None


def test_live_turn_queues_instead_of_raising(db):
    """A live turn queues the delivery (no exception, no transcript row) and a retry with the same
    key returns the same queued row instead of a second one."""
    _hold_turn(db)
    try:
        queued = _deliver(db)
        assert isinstance(queued, QueuedDetachedAdmission)
        assert (queued.session_id, queued.kind) == ("s1", "delegation_delivery")
        assert isinstance(queued.message_uid, str) and queued.message_uid
        assert db.get_messages("s1") == []
        again = _deliver(db)
        assert isinstance(again, QueuedDetachedAdmission)
        assert (again.queue_id, again.message_uid) == (queued.queue_id, queued.message_uid)
    finally:
        db.release_session_turn_lease("s1", "client-turn")
    claimed = db.claim_pending_detached_admission("s1", live_message_uids=frozenset())
    assert claimed is not None and claimed["delegation_id"] == "d1"
    assert claimed["content"] == "DELIVERY"
    assert claimed["display_kind"] == "async_delegation_complete"
    # Still in the live turn's messages: the claim survives, and a second claim finds nothing else.
    assert db.reconcile_detached_admissions(
        "s1", live_message_uids={claimed["message_uid"]}) == {"deleted": 0, "reset": 0}
    assert db.claim_pending_detached_admission(
        "s1", live_message_uids={claimed["message_uid"]}) is None


def test_cap_one_admission_per_drain_call(db):
    """Two pending rows are admitted one per claim: each call returns exactly one row, and a third
    call with both rows still in flight finds nothing."""
    _hold_turn(db)
    try:
        _deliver(db, "d1")
        _deliver(db, "d2", "SECOND")
    finally:
        db.release_session_turn_lease("s1", "client-turn")
    first = db.claim_pending_detached_admission("s1", live_message_uids=frozenset())
    second = db.claim_pending_detached_admission(
        "s1", live_message_uids={first["message_uid"]})
    assert first is not None and second is not None
    assert {first["delegation_id"], second["delegation_id"]} == {"d1", "d2"}
    assert db.claim_pending_detached_admission(
        "s1", live_message_uids={first["message_uid"], second["message_uid"]}) is None


def test_reconcile_returns_a_dead_claim_to_pending(db):
    """A claim whose turn died before the flush goes back to pending and is admitted again."""
    _hold_turn(db)
    try:
        queued = _deliver(db)
    finally:
        db.release_session_turn_lease("s1", "client-turn")
    claimed = db.claim_pending_detached_admission("s1", live_message_uids=frozenset())
    assert claimed is not None and claimed["message_uid"] == queued.message_uid
    assert db.reconcile_detached_admissions(
        "s1", live_message_uids=frozenset()) == {"deleted": 0, "reset": 1}
    again = db.claim_pending_detached_admission("s1", live_message_uids=frozenset())
    assert again is not None and again["message_uid"] == queued.message_uid


def test_retry_that_lands_directly_drops_the_queued_row(db):
    """A queued row whose retry recorded directly is deleted at claim time, never injected twice."""
    _hold_turn(db)
    try:
        _deliver(db)
    finally:
        db.release_session_turn_lease("s1", "client-turn")
    assert isinstance(_deliver(db), int)
    assert len(db.get_messages("s1")) == 1
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is None
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is None


def test_completion_note_queues_and_drains_by_admission_id(db):
    """Notes queue the same way, keyed by admission_id, and keep their display kind."""
    _hold_turn(db)
    try:
        queued = db.append_completion_note(
            "s1", "adm-1", "NOTE", "internal_notification", {})
        again = db.append_completion_note(
            "s1", "adm-1", "NOTE", "internal_notification", {})
        assert isinstance(queued, QueuedDetachedAdmission) and queued.kind == "completion_note"
        assert (again.queue_id, again.message_uid) == (queued.queue_id, queued.message_uid)
    finally:
        db.release_session_turn_lease("s1", "client-turn")
    claimed = db.claim_pending_detached_admission("s1", live_message_uids=frozenset())
    assert claimed is not None and claimed["admission_id"] == "adm-1"
    row = db.admission_message_row(claimed)
    assert row["role"] == "user" and row["content"] == "NOTE"
    assert row["display_kind"] == "internal_notification"
    assert row["message_uid"] == queued.message_uid


def test_compression_closed_session_receives_no_rows(db):
    """A parent closed by compression still raises (nothing is recorded) and nothing is queued."""
    db.create_session("parent", source="api_server")
    db.end_session("parent", "compression")
    db.create_session("child", source="api_server", parent_session_id="parent")
    with pytest.raises(CompressionSessionClosedError):
        db.append_delegation_delivery("parent", "DELIVERY", {"delegation_id": "d-closed"})
    assert db.claim_pending_detached_admission("parent", live_message_uids=frozenset()) is None
    assert db.claim_pending_detached_admission("child", live_message_uids=frozenset()) is None
    assert db.get_messages("child") == []
