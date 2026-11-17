"""Results that arrive between turns, recorded once as user rows: a detached API delegation's delivery and
an unviewed local completion's note (#134889 LC-7). Split out of ``hermes_state_messages`` and bound onto
SessionDB through ``SessionMessagesMixin``; built on its _message_row_params/_execute_write primitives.

B2: a result that arrives while the destination turn is live no longer raises to the producer. It waits
in the durable ``pending_detached_admissions`` queue ("admitted for the next drain") and the turn loop
drains it — mid-turn in ``prepare_iteration`` or at the turn boundary in ``finalize_turn`` — as an
ordinary user row, identical to the row the direct write would have produced. A queue row leaves only
once its transcript row is durable: the drain reconciles claimed rows against the compression lineage,
so a crash between injection and flush re-admits the row instead of losing it, and a crash between
flush and delete drops the queue row instead of duplicating it. The lease guard is untouched: direct
transcript writes during a live turn are still refused; only the turn owner ever writes the row.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agent.message_metadata import MESSAGE_UID, message_uid_or_none, stamp_message_uid
from hermes_state_errors import SessionTurnLeaseLostError

# Rows of a session and of the parents it was compressed from; a detached result's match condition follows.
_LINEAGE_MATCH_SQL = """WITH RECURSIVE lineage(id) AS (SELECT ? UNION
    SELECT s.parent_session_id FROM sessions s JOIN lineage l ON s.id = l.id
    JOIN sessions p ON p.id = s.parent_session_id WHERE p.end_reason = 'compression'
) SELECT m.id FROM messages m JOIN lineage l ON m.session_id = l.id WHERE """
_DELEGATION_DELIVERY_MATCH = ("m.display_kind IN ('async_delegation_complete', 'hidden') AND json_extract(m.display_metadata, "
                              "'$.delegation_id') = ? AND coalesce(json_extract(m.display_metadata, '$.delivery_notice'), '') = ?")
_COMPLETION_NOTE_MATCH = "json_extract(m.display_metadata, '$.completion_admission') = ?"

# Candidates examined per drain: the queue is keyed by the tip at enqueue time, so a row queued for a
# parent closed by compression since then is found by resolving its session id, not by an index scan.
_DRAIN_CANDIDATE_LIMIT = 32


@dataclass(frozen=True)
class QueuedDetachedAdmission:
    """A detached result that could not land in the transcript because the destination turn is live:
    it waits in the durable queue and the turn loop drains it as an ordinary user row. The producer
    must not retry blindly — the row is already recorded once, keyed by its stable id."""

    queue_id: int
    session_id: str
    kind: str  # "delegation_delivery" | "completion_note"
    message_uid: str


class SessionDetachedResultsMixin:
    """Detached API delegation deliveries and unviewed completion notes."""

    def append_delegation_delivery(self, session_id: str, content: str, metadata: dict[str, Any]) -> int | QueuedDetachedAdmission:
        """Record a detached API result once, keyed by the event's unit id (not its text or active flag).
        Returns the message id of the recorded (or already-recorded) row, or a
        :class:`QueuedDetachedAdmission` when the destination turn is live — the result waits in the
        durable queue for the turn loop's next drain instead of raising."""
        delegation_id = metadata.get("delegation_id")
        if not delegation_id:
            raise ValueError("Delegation delivery requires a stable delegation_id")
        kind = "hidden" if metadata.get("presentation_suppressed") else "async_delegation_complete"
        return self._append_detached_result(session_id, content, kind, metadata, _DELEGATION_DELIVERY_MATCH,
                                            (delegation_id, metadata.get("delivery_notice", "")),
                                            "delegation_delivery")

    def append_completion_note(self, session_id: str, admission_id: str, content: str, display_kind: str,
                               metadata: dict[str, Any]) -> int | QueuedDetachedAdmission:
        """Record an unviewed completion's note once, keyed by its admission (stored as ``completion_admission``).
        Returns the message id of the recorded (or already-recorded) row, or a
        :class:`QueuedDetachedAdmission` when the destination turn is live."""
        return self._append_detached_result(
            session_id, content, display_kind, {**metadata, "completion_admission": admission_id},
            _COMPLETION_NOTE_MATCH, (admission_id,), "completion_note")

    def _append_detached_result(self, session_id: str, content: str, display_kind: str, metadata: dict[str, Any],
                                match_sql: str, params: tuple, queue_kind: str) -> int | QueuedDetachedAdmission:
        """Record a result that arrived between turns as a user row once, including replay after rotation: a row
        ``match_sql`` (a condition on ``m``) finds in the compression lineage is returned instead. Check and insert
        share the writer transaction, so independent gateway processes cannot duplicate it. A live turn no
        longer refuses the write: the result is enqueued ("admitted for the next drain") in the same
        transaction and the turn loop drains it; only a compression-closed destination still raises."""
        from hermes_state_messages import _INSERT_MESSAGE_SQL  # that module binds this mixin
        msg = {"content": content, "display_kind": display_kind, "display_metadata": metadata}
        stamp_message_uid(msg)  # a fresh occurrence: mint its durable id
        row = self._message_row_params(session_id, "user", msg, None, time.time(), keep_reasoning=True)

        def _do(conn):
            existing = conn.execute(_LINEAGE_MATCH_SQL + match_sql + " LIMIT 1", (session_id, *params)).fetchone()
            if existing is not None:
                return existing[0]
            try:
                self._check_transcript_write_guards(conn, session_id, None, reject_active_turn_lease=True)
            except SessionTurnLeaseLostError:
                # Live turn: the row cannot land now, but the result is already accepted — queue it
                # durably in this same transaction instead of dropping it or making the producer retry.
                return self._enqueue_detached_admission(
                    conn, session_id, content, display_kind, metadata, queue_kind, message_uid_or_none(msg))
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, row).lastrowid
            self._bump_session_counters(conn, session_id, 1, 0, unit=True)
            return msg_id

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def _enqueue_detached_admission(self, conn, session_id: str, content: str, display_kind: str,
                                    metadata: dict[str, Any], queue_kind: str,
                                    message_uid: str | None) -> QueuedDetachedAdmission:
        """Insert the result into the durable admission queue (or return the row a retry already queued):
        idempotent by the stable key — ``(delegation_id, delivery_notice)`` for deliveries,
        ``admission_id`` for notes. A note with ``admission_id=None`` has no key, so each call queues
        its own row, exactly like the direct path records its own row."""
        meta_json = self._encode_display_metadata(metadata) or "{}"
        if queue_kind == "delegation_delivery":
            delegation_id = metadata.get("delegation_id")
            delivery_notice = metadata.get("delivery_notice", "") or ""
            conn.execute(
                "INSERT OR IGNORE INTO pending_detached_admissions "
                "(session_id, kind, delegation_id, delivery_notice, admission_id, content, display_kind, "
                "display_metadata, message_uid, status, created_at, attempts) "
                "VALUES (?, 'delegation_delivery', ?, ?, NULL, ?, ?, ?, ?, 'pending', ?, 0)",
                (session_id, delegation_id, delivery_notice, content, display_kind, meta_json,
                 message_uid, time.time()))
            found = conn.execute(
                "SELECT id, message_uid FROM pending_detached_admissions "
                "WHERE session_id = ? AND delegation_id = ? AND delivery_notice = ?",
                (session_id, delegation_id, delivery_notice)).fetchone()
        else:
            admission_id = metadata.get("completion_admission")
            conn.execute(
                "INSERT OR IGNORE INTO pending_detached_admissions "
                "(session_id, kind, delegation_id, delivery_notice, admission_id, content, display_kind, "
                "display_metadata, message_uid, status, created_at, attempts) "
                "VALUES (?, 'completion_note', NULL, '', ?, ?, ?, ?, ?, 'pending', ?, 0)",
                (session_id, admission_id, content, display_kind, meta_json, message_uid, time.time()))
            if admission_id is None:
                found = conn.execute(
                    "SELECT id, message_uid FROM pending_detached_admissions "
                    "WHERE session_id = ? AND message_uid = ?", (session_id, message_uid)).fetchone()
            else:
                found = conn.execute(
                    "SELECT id, message_uid FROM pending_detached_admissions "
                    "WHERE session_id = ? AND admission_id = ?", (session_id, admission_id)).fetchone()
        return QueuedDetachedAdmission(queue_id=found["id"], session_id=session_id,
                                       kind=queue_kind, message_uid=found["message_uid"])

    # ── Turn-loop drain (B2) ─────────────────────────────────────────────────

    def claim_pending_detached_admission(self, session_id: str, *,
                                         live_message_uids: frozenset = frozenset()) -> dict | None:
        """Claim this session's oldest queued admission for the turn loop to inject as an ordinary user
        row. Returns the claimed row as a plain dict (``content`` / ``display_kind`` / ``display_metadata``
        as JSON text / ``message_uid``), or None when nothing is admissible right now.

        At most one row per call (the per-drain cap); the rest wait for the next drain. A row whose
        session has rotated since it was queued is found by resolving its stored session id to the live
        tip — a parent closed by compression never receives rows. Before claiming, this reconciles the
        session's claimed rows: a claimed row whose transcript row is already durable is deleted, one
        still in the live turn's messages stays claimed, and one whose turn died before the flush goes
        back to pending. With nothing queued anywhere this is a single indexed read and no write txn.
        """
        with self._read_ctx() as conn:
            if conn.execute(
                    "SELECT 1 FROM pending_detached_admissions "
                    "WHERE status IN ('pending', 'claimed') LIMIT 1").fetchone() is None:
                return None
            pending = conn.execute(
                "SELECT * FROM pending_detached_admissions WHERE status = 'pending' "
                "ORDER BY created_at ASC, id ASC LIMIT ?", (_DRAIN_CANDIDATE_LIMIT,)).fetchall()
            claimed = conn.execute(
                "SELECT * FROM pending_detached_admissions WHERE status = 'claimed' "
                "ORDER BY created_at ASC, id ASC LIMIT ?", (_DRAIN_CANDIDATE_LIMIT,)).fetchall()
        # The stored session id is the tip at enqueue time; resolve each distinct one once and keep
        # the first row that belongs to this session.
        resolved: dict[str, str] = {}
        for row in (*pending, *claimed):
            stored = row["session_id"]
            if stored not in resolved:
                try:
                    resolved[stored] = self.resolve_resume_session_id(stored)
                except Exception:
                    resolved[stored] = stored
        mine_claimed = [dict(r) for r in claimed if resolved[r["session_id"]] == session_id]
        candidate = next((dict(r) for r in pending if resolved[r["session_id"]] == session_id), None)

        def _do(conn):
            self._reconcile_claimed_admissions_on(conn, mine_claimed, session_id, live_message_uids)
            if candidate is None:
                return None
            if self._detached_admission_recorded_on(conn, session_id, candidate):
                # A retry landed the row directly while this one waited: drop the queue row, don't inject.
                conn.execute("DELETE FROM pending_detached_admissions WHERE id = ?", (candidate["id"],))
                return None
            cur = conn.execute(
                "UPDATE pending_detached_admissions SET status = 'claimed', claimed_at = ? "
                "WHERE id = ? AND status = 'pending'", (time.time(), candidate["id"]))
            if cur.rowcount == 0:
                return None  # another drainer claimed it first (CAS)
            return candidate

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def reconcile_detached_admissions(self, session_id: str, *,
                                      live_message_uids: frozenset = frozenset()) -> dict[str, int]:
        """Settle this session's claimed rows after a flush: delete every claimed row whose transcript
        row is durable, keep the ones still in the live turn's messages, and return the rest to pending
        (their turn died or dropped the row before the flush, so the next drain re-admits them).
        Returns ``{"deleted": n, "reset": n}``."""
        with self._read_ctx() as conn:
            claimed = conn.execute(
                "SELECT * FROM pending_detached_admissions WHERE status = 'claimed' "
                "ORDER BY created_at ASC, id ASC LIMIT ?", (_DRAIN_CANDIDATE_LIMIT,)).fetchall()
        if not claimed:
            return {"deleted": 0, "reset": 0}
        resolved: dict[str, str] = {}
        for row in claimed:
            stored = row["session_id"]
            if stored not in resolved:
                try:
                    resolved[stored] = self.resolve_resume_session_id(stored)
                except Exception:
                    resolved[stored] = stored
        mine = [dict(r) for r in claimed if resolved[r["session_id"]] == session_id]
        if not mine:
            return {"deleted": 0, "reset": 0}

        def _do(conn):
            return self._reconcile_claimed_admissions_on(conn, mine, session_id, live_message_uids)

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def _reconcile_claimed_admissions_on(self, conn, claimed_rows: list[dict], session_id: str,
                                         live_message_uids: frozenset) -> dict[str, int]:
        """The reconcile itself, inside the caller's write txn: for each claimed row of this session,
        in order — transcript row durable means done (delete), still in the live messages means in
        flight (keep claimed), otherwise the turn died before the flush (back to pending)."""
        deleted = reset = 0
        for row in claimed_rows:
            if self._message_uid_in_lineage_on(conn, session_id, row["message_uid"]):
                conn.execute("DELETE FROM pending_detached_admissions WHERE id = ?", (row["id"],))
                deleted += 1
            elif row["message_uid"] in live_message_uids:
                continue  # in flight in this turn's live list: the flush has not run yet
            else:
                conn.execute(
                    "UPDATE pending_detached_admissions SET status = 'pending', claimed_at = NULL "
                    "WHERE id = ?", (row["id"],))
                reset += 1
        return {"deleted": deleted, "reset": reset}

    def _message_uid_in_lineage_on(self, conn, session_id: str, message_uid: str | None) -> bool:
        """True when a message with this uid is already durable in the session's compression lineage."""
        if not message_uid:
            return False
        return conn.execute(
            _LINEAGE_MATCH_SQL + "m.message_uid = ? LIMIT 1", (session_id, message_uid)).fetchone() is not None

    def _detached_admission_recorded_on(self, conn, session_id: str, row: dict) -> bool:
        """True when the queued row's key already has a transcript row in the session's lineage —
        a producer retry that landed directly while this row waited. Notes with no admission id
        have no key to match, so they are never considered recorded."""
        if row["kind"] == "delegation_delivery":
            match, params = _DELEGATION_DELIVERY_MATCH, (row["delegation_id"], row["delivery_notice"] or "")
        elif row["kind"] == "completion_note" and row["admission_id"]:
            match, params = _COMPLETION_NOTE_MATCH, (row["admission_id"],)
        else:
            return False
        return conn.execute(
            _LINEAGE_MATCH_SQL + match + " LIMIT 1", (session_id, *params)).fetchone() is not None

    @staticmethod
    def admission_message_row(claimed: dict) -> dict[str, Any]:
        """The live message dict for a claimed queue row: the same shape as the row the direct write
        would have produced — an ordinary ``user`` row with the delivery's own ``display_kind`` and
        ``display_metadata``, stamped with the message uid minted at enqueue time so a re-admission
        after a crashed turn keeps the same identity."""
        return {
            "role": "user",
            "content": claimed["content"],
            "display_kind": claimed["display_kind"],
            "display_metadata": json.loads(claimed["display_metadata"] or "{}"),
            MESSAGE_UID: claimed["message_uid"],
        }
