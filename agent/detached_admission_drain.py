"""Drain the durable detached-admission queue into the live turn (B2).

A detached result that arrives while the destination turn is live waits in
``pending_detached_admissions`` (see ``hermes_state_detached_results``) instead of raising to the
producer. The turn loop admits it at two points that already drain steer text: mid-turn in
``prepare_iteration`` and at the turn boundary in ``finalize_turn``. Both inject the queued row as
an ordinary user row — the same shape the direct write would have produced — never by rewriting a
persisted row, so the prompt cache keeps its contract (the new row costs only its own tokens).

The queue row is claimed (CAS) before injection and deleted only once its transcript row is
durable: the drain reconciles claimed rows after each flush, so a crash between injection and
flush re-admits the row, and a crash between flush and delete drops the queue row instead of
duplicating it. Everything here fails open: an agent with no session db, no session id, or
persistence disabled simply never drains.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.message_metadata import message_uid_or_none

logger = logging.getLogger(__name__)


def _live_message_uids(messages: list[dict]) -> frozenset:
    """The uids of the live list, so the reconcile can tell an in-flight claim from a dead one."""
    return frozenset(
        uid for msg in messages
        if isinstance(msg, dict) and (uid := message_uid_or_none(msg)) is not None
    )


def _drainable(agent: Any) -> tuple | None:
    """The ``(db, session_id)`` the drain can run against, or None when the agent cannot drain:
    no session db, no session id, or a persistence-disabled agent (its rows would never flush, so
    claiming here would only strand the row until the next real turn)."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id or getattr(agent, "_persist_disabled", False):
        return None
    return db, session_id


def drain_pending_detached_admission(agent: Any, messages: list[dict]) -> dict | None:
    """Mid-turn drain (``prepare_iteration``): claim the session's oldest queued admission and inject
    it as an ordinary user row after the newest tool result. At most one row per call; the rest wait
    for the next iteration. Returns the injected row dict, or None when nothing was admitted. A drain
    that cannot inject yet — no tool result this iteration — releases the claim, so the row stays
    queued for a later iteration instead of being lost."""
    drained = _drainable(agent)
    if drained is None:
        return None
    db, session_id = drained
    try:
        claimed = db.claim_pending_detached_admission(
            session_id, live_message_uids=_live_message_uids(messages))
    except Exception:
        logger.debug("detached admission drain unavailable (session=%s)", session_id, exc_info=True)
        return None
    if claimed is None:
        return None
    row = db.admission_message_row(claimed)
    if not _inject_after_newest_tool_result(messages, row):
        reconcile_detached_admissions(agent, messages)
        return None
    return row


def drain_pending_detached_admission_at_turn_end(agent: Any, messages: list[dict]) -> dict | None:
    """Turn-boundary drain (``finalize_turn``): claim the oldest queued admission and append it after
    the final assistant message, so the next turn reloads it as an ordinary user row. Only when the
    tail is an assistant message — appending after a user row would break role alternation, so a
    vetoed or user-tailed turn leaves the row queued for the next turn."""
    drained = _drainable(agent)
    if drained is None:
        return None
    db, session_id = drained
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        return None
    try:
        claimed = db.claim_pending_detached_admission(
            session_id, live_message_uids=_live_message_uids(messages))
    except Exception:
        logger.debug("detached admission turn-end drain unavailable (session=%s)", session_id, exc_info=True)
        return None
    if claimed is None:
        return None
    row = db.admission_message_row(claimed)
    messages.append(row)
    return row


def reconcile_detached_admissions(agent: Any, messages: list[dict]) -> dict[str, int]:
    """Settle this session's claimed rows after a flush (see
    ``SessionDB.reconcile_detached_admissions``): delete every claimed row whose transcript row is
    durable, keep the ones still in the live turn's messages, return the rest to pending. Fail-open:
    ``{"deleted": 0, "reset": 0}`` when the drain cannot run."""
    drained = _drainable(agent)
    if drained is None:
        return {"deleted": 0, "reset": 0}
    db, session_id = drained
    try:
        return db.reconcile_detached_admissions(
            session_id, live_message_uids=_live_message_uids(messages))
    except Exception:
        logger.debug("detached admission reconcile unavailable (session=%s)", session_id, exc_info=True)
        return {"deleted": 0, "reset": 0}


def _inject_after_newest_tool_result(messages: list[dict], row: dict) -> bool:
    """Insert the row after the newest tool message (the steer injection point); False when the turn
    has no tool result yet, so the caller leaves the row queued for a later iteration."""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            messages.insert(idx + 1, row)
            return True
    return False
