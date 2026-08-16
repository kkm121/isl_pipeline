"""Durable SQLite State Store for Kaggle Lifecycle Management.

Persists account statuses, active kernel mappings, job execution lifecycle,
and rotation counters across MCP server restarts.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any
import time
import logging

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"complete", "error", "cancelled", "timeout", "failed"}


class KaggleStateStore:
    def __init__(self, db_path: str = "credentials/kaggle_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'available',
                    last_used REAL NOT NULL DEFAULT 0.0,
                    active_kernels INTEGER NOT NULL DEFAULT 0,
                    max_concurrent INTEGER NOT NULL DEFAULT 2,
                    total_submissions INTEGER NOT NULL DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kernel_jobs (
                    kernel_ref TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    submit_time REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    last_polled REAL NOT NULL DEFAULT 0.0,
                    output_dir TEXT,
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rotation_policy (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    rotation_permitted INTEGER NOT NULL DEFAULT 1,
                    current_rotations INTEGER NOT NULL DEFAULT 0,
                    max_rotations INTEGER NOT NULL DEFAULT 3,
                    require_human_approval_after INTEGER NOT NULL DEFAULT 2
                )
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO rotation_policy (id, rotation_permitted, current_rotations, max_rotations, require_human_approval_after)
                VALUES (1, 1, 0, 3, 2)
            """)
            conn.commit()

    def sync_accounts(self, accounts_metadata: List[Dict[str, Any]]) -> None:
        """Register configured accounts while preserving dynamic state."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for acc in accounts_metadata:
                cursor.execute("""
                    INSERT INTO accounts (account_id, username, status, last_used, active_kernels, max_concurrent, total_submissions)
                    VALUES (?, ?, 'available', 0.0, 0, ?, 0)
                    ON CONFLICT(account_id) DO UPDATE SET
                        username = excluded.username,
                        max_concurrent = excluded.max_concurrent
                """, (acc["account_id"], acc["username"], acc.get("max_concurrent", 2)))
            conn.commit()

    def get_accounts(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accounts")
            return [dict(row) for row in cursor.fetchall()]

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_account_usage(self, account_id: str, delta_kernels: int = 0, delta_submissions: int = 0, status: Optional[str] = None) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = time.time()
            if status:
                cursor.execute("""
                    UPDATE accounts
                    SET active_kernels = MAX(0, active_kernels + ?),
                        total_submissions = total_submissions + ?,
                        status = ?,
                        last_used = ?
                    WHERE account_id = ?
                """, (delta_kernels, delta_submissions, status, now, account_id))
            else:
                cursor.execute("""
                    UPDATE accounts
                    SET active_kernels = MAX(0, active_kernels + ?),
                        total_submissions = total_submissions + ?,
                        last_used = ?
                    WHERE account_id = ?
                """, (delta_kernels, delta_submissions, now, account_id))
            conn.commit()

    def set_account_status(self, account_id: str, status: str) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE accounts SET status = ? WHERE account_id = ?", (status, account_id))
            conn.commit()

    def record_job_submission(self, kernel_ref: str, account_id: str, output_dir: Optional[str] = None) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = time.time()
            cursor.execute("""
                INSERT INTO kernel_jobs (kernel_ref, account_id, submit_time, status, last_polled, output_dir)
                VALUES (?, ?, ?, 'queued', ?, ?)
                ON CONFLICT(kernel_ref) DO UPDATE SET
                    account_id = excluded.account_id,
                    submit_time = excluded.submit_time,
                    status = 'queued',
                    last_polled = excluded.last_polled,
                    output_dir = excluded.output_dir
            """, (kernel_ref, account_id, now, now, output_dir))
            # Update account active kernels and total submissions
            cursor.execute("""
                UPDATE accounts
                SET active_kernels = active_kernels + 1,
                    total_submissions = total_submissions + 1,
                    status = 'in_use',
                    last_used = ?
                WHERE account_id = ?
            """, (now, account_id))
            conn.commit()
        logger.info("Recorded Kaggle job submission in SQLite: %s (account: %s)", kernel_ref, account_id)

    def update_job_status(self, kernel_ref: str, status: str) -> Optional[str]:
        """Update job status and automatically release account if status is terminal."""
        account_to_release = None
        clean_status = status.lower().strip()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT account_id, status FROM kernel_jobs WHERE kernel_ref = ?", (kernel_ref,))
            row = cursor.fetchone()
            if not row:
                logger.warning("No tracked job found for kernel_ref: %s", kernel_ref)
                return None

            account_id = row["account_id"]
            prev_status = row["status"].lower()

            cursor.execute("""
                UPDATE kernel_jobs
                SET status = ?, last_polled = ?
                WHERE kernel_ref = ?
            """, (clean_status, time.time(), kernel_ref))

            # If transitioning to a terminal state from a non-terminal state, decrement active count
            if clean_status in TERMINAL_STATES and prev_status not in TERMINAL_STATES:
                cursor.execute("""
                    UPDATE accounts
                    SET active_kernels = MAX(0, active_kernels - 1)
                    WHERE account_id = ?
                """, (account_id,))
                
                # Check if account has no active kernels left
                cursor.execute("SELECT active_kernels, status FROM accounts WHERE account_id = ?", (account_id,))
                acc_row = cursor.fetchone()
                if acc_row and acc_row["active_kernels"] == 0 and acc_row["status"] == 'in_use':
                    cursor.execute("UPDATE accounts SET status = 'available' WHERE account_id = ?", (account_id,))
                
                account_to_release = account_id
                logger.info("Guaranteed account release on terminal state '%s': %s (account: %s)", clean_status, kernel_ref, account_id)

            conn.commit()
        return account_to_release

    def get_job(self, kernel_ref: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kernel_jobs WHERE kernel_ref = ?", (kernel_ref,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_active_jobs(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kernel_jobs WHERE status NOT IN ('complete', 'error', 'cancelled', 'timeout', 'failed')")
            return [dict(row) for row in cursor.fetchall()]

    def recover_orphaned_jobs(self) -> int:
        """Startup recovery: reconciles account active counters against non-terminal jobs."""
        recovered = 0
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Calculate actual active jobs per account
            cursor.execute("""
                SELECT account_id, COUNT(*) as real_active
                FROM kernel_jobs
                WHERE status NOT IN ('complete', 'error', 'cancelled', 'timeout', 'failed')
                GROUP BY account_id
            """)
            active_counts = {row["account_id"]: row["real_active"] for row in cursor.fetchall()}

            # Reconcile accounts table
            cursor.execute("SELECT account_id, active_kernels, status FROM accounts")
            accounts = cursor.fetchall()
            for acc in accounts:
                acc_id = acc["account_id"]
                real_active = active_counts.get(acc_id, 0)
                if acc["active_kernels"] != real_active:
                    new_status = 'in_use' if real_active > 0 else ('available' if acc["status"] == 'in_use' else acc["status"])
                    cursor.execute("""
                        UPDATE accounts
                        SET active_kernels = ?, status = ?
                        WHERE account_id = ?
                    """, (real_active, new_status, acc_id))
                    recovered += 1
            conn.commit()
        if recovered > 0:
            logger.info("Recovered %d orphaned account states from SQLite state store on startup.", recovered)
        return recovered

    def get_rotation_policy(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rotation_policy WHERE id = 1")
            row = cursor.fetchone()
            return dict(row) if row else {}

    def update_rotation_count(self, current_rotations: int) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE rotation_policy SET current_rotations = ? WHERE id = 1", (current_rotations,))
            conn.commit()
