"""The turn loop's detached-admission drain (B2): queued results land as ordinary user rows —
mid-turn after the newest tool result, or at the turn boundary after the final assistant message —
and the queue empties only once the transcript row is durable."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.detached_admission_drain import (
    drain_pending_detached_admission,
    drain_pending_detached_admission_at_turn_end,
    reconcile_detached_admissions,
)
from agent.message_metadata import MESSAGE_UID
from agent.turn_finalizer import finalize_turn
from agent.turn_iteration_prep import prepare_iteration
from hermes_state import SessionDB
from hermes_state_detached_results import QueuedDetachedAdmission


class _FinalizeFakeAgent:
    """The established ``finalize_turn`` fixture shape (see
    ``test_turn_finalizer_final_response_persistence.py``), plus the drain's inputs."""

    def __init__(self, db):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "s1"
        self._session_db = db
        self._persist_disabled = False
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages: list[dict] | None = None
        self.persist_calls = 0
        self._persist_user_message_idx: int | None = None
        self._persist_user_message_override = None
        self._persist_user_message_timestamp: float | None = None
        self._db = db

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persist_calls += 1
        self._db.replace_messages(self.session_id, messages)
        self.persisted_messages = [dict(message) for message in messages]

    def _apply_persist_user_message_override(self, messages):
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if idx is not None and override is not None:
            messages[idx]["content"] = override

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(agent, messages, monkeypatch, *, interrupted=False):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    return finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )


def test_finalize_turn_drains_at_the_turn_boundary(db, monkeypatch):
    """The turn-boundary hook is wired into the real ``finalize_turn``: the queued delivery is
    appended after the final assistant message, persisted again, and the queue row is deleted once
    its transcript row is durable."""
    _queue_delivery(db)
    agent = _FinalizeFakeAgent(db)
    messages = [{"role": "user", "content": "do it"}]
    result = _finalize(agent, messages, monkeypatch)
    assert result["messages"][-1]["role"] == "user"
    assert result["messages"][-1]["content"] == "DELIVERY"
    assert agent.persist_calls == 2  # the turn, then again after the drain
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is None
    assert any(m["content"] == "DELIVERY" for m in db.get_messages("s1"))


def test_finalize_turn_skips_the_drain_when_interrupted(db, monkeypatch):
    """An interrupted turn leaves the row queued for the next turn instead of appending after a
    tail the failure path owns."""
    _queue_delivery(db)
    agent = _FinalizeFakeAgent(db)
    messages = [{"role": "user", "content": "do it"}]
    _finalize(agent, messages, monkeypatch, interrupted=True)
    assert all(m["content"] != "DELIVERY" for m in messages)
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is not None


@pytest.fixture
def db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    try:
        yield db
    finally:
        db.close()


def _queue_delivery(db, delegation_id="d1", content="DELIVERY"):
    """Enqueue a delivery the way a live turn does, then release the turn."""
    assert db.acquire_session_turn_lease("s1", "client-turn", wait_seconds=0)
    try:
        outcome = db.append_delegation_delivery(
            "s1", content, {"delegation_id": delegation_id})
    finally:
        db.release_session_turn_lease("s1", "client-turn")
    assert isinstance(outcome, QueuedDetachedAdmission)
    return outcome


def _agent(db, **overrides):
    agent = SimpleNamespace(_session_db=db, session_id="s1", _persist_disabled=False)
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _tool_messages():
    return [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call-1", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "output"},
    ]


def _roles(messages):
    return [m["role"] for m in messages]


def test_mid_turn_drain_injects_after_the_newest_tool_result(db):
    """The queued row lands as an ordinary user row after the tool result; the prefix up to that
    tool result is untouched, so the cached prompt prefix survives the admission."""
    queued = _queue_delivery(db)
    messages = _tool_messages()
    head = messages[:]
    row = drain_pending_detached_admission(_agent(db), messages)
    assert row is not None and row["role"] == "user" and row["content"] == "DELIVERY"
    assert row[MESSAGE_UID] == queued.message_uid
    assert messages[:3] == head and messages[3] is row
    assert _roles(messages) == ["user", "assistant", "tool", "user"]


def test_mid_turn_drain_admits_at_most_one_per_call(db):
    """One drain call admits exactly one row even with two pending; the second stays queued."""
    _queue_delivery(db, "d1")
    _queue_delivery(db, "d2", "SECOND")
    messages = _tool_messages()
    row = drain_pending_detached_admission(_agent(db), messages)
    assert row is not None and len(messages) == 4
    second = db.claim_pending_detached_admission(
        "s1", live_message_uids={row[MESSAGE_UID]})
    assert second is not None and second["delegation_id"] == "d2"


def test_mid_turn_drain_without_a_tool_result_leaves_the_row_queued(db):
    """No tool result yet this iteration: nothing is injected and the claim is released, so the row
    stays queued for a later iteration instead of being lost."""
    _queue_delivery(db)
    messages = [{"role": "user", "content": "hi"}]
    assert drain_pending_detached_admission(_agent(db), messages) is None
    assert _roles(messages) == ["user"]
    claimed = db.claim_pending_detached_admission("s1", live_message_uids=frozenset())
    assert claimed is not None and claimed["delegation_id"] == "d1"


def test_end_to_end_inject_flush_reconcile_empties_the_queue(db):
    """Inject → flush the row under the turn's own lease → reconcile: the queue row is deleted once
    its transcript row is durable, and a re-drain finds nothing to inject twice."""
    _queue_delivery(db)
    agent = _agent(db)
    messages = _tool_messages()
    row = drain_pending_detached_admission(agent, messages)
    assert row is not None
    assert db.acquire_session_turn_lease("s1", "test-turn", wait_seconds=0)
    try:
        assert db.append_messages_batch("s1", [row], turn_lease_holder="test-turn") == 1
    finally:
        db.release_session_turn_lease("s1", "test-turn")
    assert reconcile_detached_admissions(agent, messages) == {"deleted": 1, "reset": 0}
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is None
    rows = db.get_messages("s1")
    assert len(rows) == 1 and rows[0]["content"] == "DELIVERY"
    assert drain_pending_detached_admission(agent, messages) is None


def test_turn_end_drain_appends_after_the_final_assistant_message(db):
    """At the turn boundary the row is appended after the final assistant message, so the next turn
    reloads it as an ordinary user row."""
    _queue_delivery(db)
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]
    row = drain_pending_detached_admission_at_turn_end(_agent(db), messages)
    assert row is not None and row["content"] == "DELIVERY"
    assert messages[-1] is row and _roles(messages) == ["user", "assistant", "user"]


def test_turn_end_drain_skips_a_user_tailed_turn(db):
    """A turn whose tail is a user row (vetoed, interrupted) leaves the row queued for the next turn
    instead of breaking role alternation."""
    _queue_delivery(db)
    messages = [{"role": "user", "content": "hi"}]
    assert drain_pending_detached_admission_at_turn_end(_agent(db), messages) is None
    assert _roles(messages) == ["user"]
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is not None


def test_drain_fails_open_without_a_session_db(db):
    """No session db, no session id, or persistence disabled: the drain does nothing and the row
    stays queued — the turn is unaffected."""
    _queue_delivery(db)
    messages = _tool_messages()
    assert drain_pending_detached_admission(_agent(db, _session_db=None), messages) is None
    assert drain_pending_detached_admission(_agent(db, session_id=None), messages) is None
    assert drain_pending_detached_admission(_agent(db, _persist_disabled=True), messages) is None
    assert drain_pending_detached_admission_at_turn_end(_agent(db, _session_db=None), messages) is None
    assert reconcile_detached_admissions(_agent(db, _session_db=None), messages) == {
        "deleted": 0, "reset": 0}
    assert len(messages) == 3
    assert db.claim_pending_detached_admission("s1", live_message_uids=frozenset()) is not None


def test_prepare_iteration_admits_the_queued_delivery(db):
    """The mid-turn hook is wired into the real ``prepare_iteration``: the queued delivery lands in
    the iteration's messages as an ordinary user row after the newest tool result."""
    _queue_delivery(db)
    agent = SimpleNamespace(
        _session_db=db, session_id="s1", _persist_disabled=False,
        step_callback=None, _skill_nudge_interval=0,
        _adopt_nous_key_before_expiry=lambda: None,
        _drain_pending_steer=lambda: None, run_budget_seconds=None,
        _sanitize_tool_call_arguments=lambda messages, logger, session_id, cursor: 0,
        logger=None,
    )
    prep = prepare_iteration(agent, messages=_tool_messages(), api_call_count=1)
    assert prep.action == "fallthrough"
    assert _roles(prep.messages) == ["user", "assistant", "tool", "user"]
    assert prep.messages[-1]["content"] == "DELIVERY"


def test_prepare_iteration_without_queued_rows_changes_nothing(db):
    """With an empty queue the hook is a no-op: the iteration's messages come back untouched."""
    agent = SimpleNamespace(
        _session_db=db, session_id="s1", _persist_disabled=False,
        step_callback=None, _skill_nudge_interval=0,
        _adopt_nous_key_before_expiry=lambda: None,
        _drain_pending_steer=lambda: None, run_budget_seconds=None,
        _sanitize_tool_call_arguments=lambda messages, logger, session_id, cursor: 0,
        logger=None,
    )
    messages = _tool_messages()
    prep = prepare_iteration(agent, messages=messages, api_call_count=1)
    assert prep.action == "fallthrough"
    assert prep.messages == messages and len(prep.messages) == 3
