"""Cross-Process Durable Gatekeeper for Autonomous MCP Operations.

Provides the single authoritative pipeline state for all MCP servers:
- Filesystem MCP (isl-filesystem)
- Linter & Test MCP (isl-linter-test)
- Kaggle Manager MCP (isl-kaggle-manager)
- GitHub MCP (isl-github)

Backed by SQLite with WAL mode and transaction serialization (BEGIN IMMEDIATE).
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable

from src.orchestrator.state_machine import (
    HumanGateException,
    PipelineState,
    PipelineStateMachine,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


class GateViolation(Exception):
    """Raised when an MCP operation is attempted outside permitted pipeline states."""

    pass


class PipelineGate:
    """Authoritative cross-process gate and synchronization point for all MCP tool operations."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.state_dir = self.project_root / ".state"
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.state_dir / "pipeline.db"
        except OSError:
            # Fallback for sealed read-only container root
            self.state_dir = Path("/tmp/.state")
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.state_dir / "pipeline.db"

        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    current_retries INTEGER NOT NULL DEFAULT 0,
                    current_config_mutations INTEGER NOT NULL DEFAULT 0,
                    max_total_retries INTEGER NOT NULL DEFAULT 5,
                    max_config_mutations INTEGER NOT NULL DEFAULT 3,
                    failure_signatures TEXT NOT NULL DEFAULT '{}',
                    tree_baseline TEXT,
                    last_updated REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            row = conn.execute("SELECT state FROM pipeline_state WHERE id = 1").fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO pipeline_state (
                        id, state, current_retries, current_config_mutations,
                        max_total_retries, max_config_mutations, failure_signatures,
                        tree_baseline, last_updated
                    ) VALUES (1, ?, 0, 0, 5, 3, '{}', NULL, ?)
                    """,
                    (PipelineState.IDLE.name, time.time()),
                )

    def current_state(self) -> PipelineState:
        """Return the current authoritative state from the durable store."""
        with self._connection() as conn:
            row = conn.execute("SELECT state FROM pipeline_state WHERE id = 1").fetchone()
        if not row:
            return PipelineState.IDLE
        return PipelineState[row[0]]

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return complete snapshot of the state and retry counters."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT state, current_retries, current_config_mutations,
                       max_total_retries, max_config_mutations, failure_signatures,
                       tree_baseline, last_updated
                FROM pipeline_state WHERE id = 1
                """
            ).fetchone()
        if not row:
            return {"state": PipelineState.IDLE.name}
        return {
            "state": row[0],
            "current_retries": row[1],
            "current_config_mutations": row[2],
            "max_total_retries": row[3],
            "max_config_mutations": row[4],
            "failure_signatures": json.loads(row[5]) if row[5] else {},
            "tree_baseline": json.loads(row[6]) if row[6] else None,
            "last_updated": row[7],
        }

    def require_state(self, operation: str, allowed_states: Iterable[PipelineState]) -> PipelineState:
        """Gate an MCP operation. Raises GateViolation if current state is not permitted."""
        state = self.current_state()
        allowed = set(allowed_states)
        if state not in allowed:
            allowed_names = ", ".join(s.name for s in sorted(allowed, key=lambda x: x.value))
            raise GateViolation(
                f"MCP operation '{operation}' is forbidden in pipeline state '{state.name}'. "
                f"Permitted state(s): {allowed_names}"
            )
        return state

    def transition(self, target: PipelineState, evidence: Dict[str, Any], agent: str) -> PipelineState:
        """Execute an authoritative transition under SQLite BEGIN IMMEDIATE lock."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT state, current_retries, current_config_mutations,
                           max_total_retries, max_config_mutations, failure_signatures,
                           tree_baseline
                    FROM pipeline_state WHERE id = 1
                    """
                ).fetchone()

                current_state_enum = PipelineState[row[0]]
                current_retries = row[1]
                current_config_mutations = row[2]
                max_total_retries = row[3]
                max_config_mutations = row[4]
                failure_sigs = json.loads(row[5]) if row[5] else {}
                tree_base = json.loads(row[6]) if row[6] else None

                # Reconstruct state machine instance with exact durable state
                sm = PipelineStateMachine(
                    project_root=str(self.project_root),
                    retry_policy=RetryPolicy(
                        max_total_retries=max_total_retries,
                        max_config_mutations=max_config_mutations,
                        current_retries=current_retries,
                        current_config_mutations=current_config_mutations,
                        failure_signatures=failure_sigs,
                    ),
                )
                sm.current_state = current_state_enum
                sm._tree_baseline = tree_base

                # Perform verified transition
                sm.transition(target=target, evidence=evidence, agent=agent)

                new_state = sm.get_state()
                now = time.time()

                # Update durable state
                conn.execute(
                    """
                    UPDATE pipeline_state SET
                        state = ?,
                        current_retries = ?,
                        current_config_mutations = ?,
                        failure_signatures = ?,
                        tree_baseline = ?,
                        last_updated = ?
                    WHERE id = 1
                    """,
                    (
                        new_state.name,
                        sm.retry_policy.current_retries,
                        sm.retry_policy.current_config_mutations,
                        json.dumps(sm.retry_policy.failure_signatures),
                        json.dumps(sm._tree_baseline) if sm._tree_baseline else None,
                        now,
                    ),
                )

                # Record history entry
                conn.execute(
                    """
                    INSERT INTO pipeline_history (
                        from_state, to_state, evidence, agent, timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        current_state_enum.name,
                        new_state.name,
                        json.dumps(evidence),
                        agent,
                        now,
                    ),
                )

                conn.execute("COMMIT")
                return new_state
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def mutate_config(
        self,
        config_path: str,
        mutation_data: Dict[str, Any],
        reason: str,
        agent: str = "ml-ops",
    ) -> Dict[str, Any]:
        """Apply configuration mutation with atomic limit checking."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT state, current_retries, current_config_mutations,
                           max_total_retries, max_config_mutations, failure_signatures
                    FROM pipeline_state WHERE id = 1
                    """
                ).fetchone()

                current_state_enum = PipelineState[row[0]]
                current_mutations = row[2]
                max_mutations = row[4]

                if current_mutations >= max_mutations:
                    conn.execute(
                        "UPDATE pipeline_state SET state = ?, last_updated = ? WHERE id = 1",
                        (PipelineState.HUMAN_GATE.name, time.time()),
                    )
                    conn.execute("COMMIT")
                    raise HumanGateException(f"Config mutation limit exceeded ({current_mutations}/{max_mutations})")

                sm = PipelineStateMachine(
                    project_root=str(self.project_root),
                    retry_policy=RetryPolicy(
                        max_config_mutations=max_mutations,
                        current_config_mutations=current_mutations,
                    ),
                )
                sm.current_state = current_state_enum

                result = sm.mutate_config(config_path, mutation_data, reason=reason, agent=agent)

                conn.execute(
                    "UPDATE pipeline_state SET current_config_mutations = ?, last_updated = ? WHERE id = 1",
                    (sm.retry_policy.current_config_mutations, time.time()),
                )
                conn.execute("COMMIT")
                return result
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def reset(self) -> None:
        """Reset the durable pipeline state to IDLE."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE pipeline_state SET
                    state = ?,
                    current_retries = 0,
                    current_config_mutations = 0,
                    failure_signatures = '{}',
                    tree_baseline = NULL,
                    last_updated = ?
                WHERE id = 1
                """,
                (PipelineState.IDLE.name, time.time()),
            )
