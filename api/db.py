"""
GyanMitra — Database Helper
SQLite-backed persistence layer. Replaces the browser localStorage in the
prototype with server-side storage that survives clearing cookies / switching
devices and gives teachers a real escalation queue to inspect.

Tables
------
students        — one row per student_id (created on first visit)
sessions        — one row per session_id; links to student; tracks subject,
                  turn count, and per-topic confusion counts as JSON
escalations     — one row each time confusion_count for a topic crosses the
                  threshold; readable by a teacher dashboard
turn_log        — every NLU/dialog/action event written here for audit
"""

import sqlite3
import json
import os
import threading
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gyanmitra.db")
_local = threading.local()   # thread-local connections (safe for Flask)


def _get_conn():
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return _local.conn


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id   TEXT PRIMARY KEY,
            name         TEXT,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            student_id   TEXT NOT NULL REFERENCES students(student_id),
            subject      TEXT,
            turns        INTEGER NOT NULL DEFAULT 0,
            confusion    TEXT NOT NULL DEFAULT '{}',
            last_active  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS escalations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            topic        TEXT NOT NULL,
            count        INTEGER NOT NULL,
            flagged_at   TEXT NOT NULL DEFAULT (datetime('now')),
            resolved     INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS turn_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            turn_number  INTEGER NOT NULL,
            user_text    TEXT NOT NULL,
            intent       TEXT,
            confidence   REAL,
            subject      TEXT,
            topic        TEXT,
            bot_reply    TEXT,
            escalated    INTEGER NOT NULL DEFAULT 0,
            ts           TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


# ── students ──────────────────────────────────────────────────────────────

def get_or_create_student(student_id: str, name: str = None) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            (student_id, name)
        )
        conn.commit()
        return {"student_id": student_id, "name": name, "created_at": datetime.utcnow().isoformat()}
    return dict(row)


def update_student_name(student_id: str, name: str):
    conn = _get_conn()
    conn.execute("UPDATE students SET name = ? WHERE student_id = ?", (name, student_id))
    conn.commit()


# ── sessions ──────────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["confusion"] = json.loads(d["confusion"])
    return d


def create_session(session_id: str, student_id: str) -> dict:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (session_id, student_id) VALUES (?, ?)",
        (session_id, student_id)
    )
    conn.commit()
    return {"session_id": session_id, "student_id": student_id,
            "subject": None, "turns": 0, "confusion": {}}


def get_or_create_session(session_id: str, student_id: str) -> dict:
    s = get_session(session_id)
    if s is None:
        get_or_create_student(student_id)
        s = create_session(session_id, student_id)
    return s


def save_session(session: dict):
    conn = _get_conn()
    conn.execute("""
        UPDATE sessions
        SET subject = ?, turns = ?, confusion = ?, last_active = datetime('now')
        WHERE session_id = ?
    """, (
        session["subject"],
        session["turns"],
        json.dumps(session["confusion"]),
        session["session_id"],
    ))
    conn.commit()


def get_student_sessions(student_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE student_id = ? ORDER BY last_active DESC",
        (student_id,)
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["confusion"] = json.loads(d["confusion"])
        result.append(d)
    return result


# ── escalations ───────────────────────────────────────────────────────────

def create_escalation(student_id: str, session_id: str, topic: str, count: int):
    conn = _get_conn()
    # Only create one open escalation per student/topic
    existing = conn.execute("""
        SELECT id FROM escalations
        WHERE student_id = ? AND topic = ? AND resolved = 0
    """, (student_id, topic)).fetchone()
    if existing is None:
        conn.execute("""
            INSERT INTO escalations (student_id, session_id, topic, count)
            VALUES (?, ?, ?, ?)
        """, (student_id, session_id, topic, count))
        conn.commit()


def get_open_escalations() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT e.*, s.name as student_name
        FROM escalations e
        LEFT JOIN students s ON e.student_id = s.student_id
        WHERE e.resolved = 0
        ORDER BY e.flagged_at DESC
    """).fetchall()
    return [dict(r) for r in rows]


def resolve_escalation(escalation_id: int):
    conn = _get_conn()
    conn.execute("UPDATE escalations SET resolved = 1 WHERE id = ?", (escalation_id,))
    conn.commit()


# ── turn log ──────────────────────────────────────────────────────────────

def log_turn(session_id: str, turn_number: int, user_text: str,
             intent: str, confidence: float, subject: str, topic: str,
             bot_reply: str, escalated: bool):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO turn_log
          (session_id, turn_number, user_text, intent, confidence,
           subject, topic, bot_reply, escalated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, turn_number, user_text, intent, confidence,
          subject, topic, bot_reply, int(escalated)))
    conn.commit()


def get_turn_history(session_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM turn_log WHERE session_id = ? ORDER BY turn_number",
        (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]
