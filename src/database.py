"""Phase 11 — SQLite Database and Incident History Management.

Persists session metadata and summary detection incidents inside
database/fireguard.db using parameterized SQL queries and context-managed
connections. Designed to log one record per analyzed image or video rather
than flooding storage with per-frame entries.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import config


class DatabaseOperationError(RuntimeError):
    """Raised when an incident database write, query, or migration fails."""


class IncidentDatabase:
    """Parameterized SQLite storage service for FireGuard AI."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else config.DB_PATH
        self.initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a context-managed SQLite connection with foreign keys enabled."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=15.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseOperationError(f"Unable to connect to database '{self.db_path}': {exc}") from exc

        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseOperationError(f"SQLite transaction failed: {exc}") from exc
        finally:
            conn.close()

    def initialize_database(self) -> None:
        """Create normalized sessions and incidents tables if they do not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    notes TEXT DEFAULT ''
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fire_coverage REAL NOT NULL,
                    smoke_coverage REAL NOT NULL,
                    visual_severity TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    video_timestamp REAL,
                    created_at TEXT NOT NULL,
                    output_path TEXT DEFAULT '',
                    original_path TEXT DEFAULT '',
                    growth_trend TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents (created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_detection_type ON incidents (detection_type);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents (visual_severity);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_sessions (
                    audit_session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT DEFAULT '',
                    model_identifier TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_filename TEXT DEFAULT '',
                    items_processed INTEGER DEFAULT 0,
                    fire_detections INTEGER DEFAULT 0,
                    smoke_detections INTEGER DEFAULT 0,
                    evaluation_summary TEXT DEFAULT '',
                    avg_latency_ms REAL DEFAULT 0.0,
                    status TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_health_checks (
                    check_id TEXT PRIMARY KEY,
                    checked_at TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    architecture TEXT DEFAULT '',
                    class_names TEXT DEFAULT '',
                    execution_device TEXT DEFAULT '',
                    avg_latency_ms REAL,
                    model_tier TEXT DEFAULT '',
                    details TEXT DEFAULT ''
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_audit_results (
                    result_id TEXT PRIMARY KEY,
                    audit_session_id TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    ground_truth_class TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    matched_detections INTEGER NOT NULL,
                    false_positives INTEGER NOT NULL,
                    false_negatives INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    iou REAL NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    audit_status TEXT NOT NULL,
                    snapshot_path TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (audit_session_id) REFERENCES audit_sessions (audit_session_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_events (
                    event_id TEXT PRIMARY KEY,
                    audit_session_id TEXT NOT NULL,
                    frame_or_item TEXT NOT NULL,
                    timestamp_sec REAL,
                    detection_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fire_coverage REAL DEFAULT 0.0,
                    smoke_coverage REAL DEFAULT 0.0,
                    inference_latency_ms REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (audit_session_id) REFERENCES audit_sessions (audit_session_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_incidents (
                    alert_id TEXT PRIMARY KEY,
                    audit_session_id TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    detected_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    video_timestamp REAL,
                    fire_coverage REAL DEFAULT 0.0,
                    smoke_coverage REAL DEFAULT 0.0,
                    alert_level TEXT NOT NULL,
                    snapshot_path TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (audit_session_id) REFERENCES audit_sessions (audit_session_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_evaluation_summaries (
                    eval_id TEXT PRIMARY KEY,
                    evaluated_at TEXT NOT NULL,
                    model_identifier TEXT NOT NULL,
                    dataset_split TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    map50 REAL,
                    map50_95 REAL,
                    false_alarm_rate REAL,
                    avg_inference_ms REAL,
                    avg_fps REAL,
                    details_json TEXT DEFAULT ''
                );
                """
            )

    def create_session(
        self,
        session_id: str | None = None,
        source_type: str = "dashboard",
        notes: str = "",
    ) -> str:
        """Insert a new session record (or keep existing) and return its session_id."""
        sid = session_id or f"SES-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, created_at, source_type, notes)
                VALUES (?, ?, ?, ?);
                """,
                (sid, now_str, str(source_type), str(notes)),
            )
        return sid

    def insert_incident(
        self,
        file_name: str,
        file_type: str,
        detection_type: str,
        confidence: float,
        fire_coverage: float,
        smoke_coverage: float,
        visual_severity: str,
        alert_level: str,
        session_id: str | None = None,
        incident_id: str | None = None,
        video_timestamp: float | None = None,
        created_at: str | None = None,
        output_path: str = "",
        original_path: str = "",
        growth_trend: str = "",
        notes: str = "",
    ) -> str:
        """Validate and insert one summary incident record into SQLite."""
        if not file_name or not file_type or not detection_type:
            raise DatabaseOperationError("file_name, file_type, and detection_type are required.")

        now_utc = datetime.now(timezone.utc)
        sid = session_id or f"SES-{now_utc.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        inc_id = incident_id or f"INC-{now_utc.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        timestamp_str = created_at or now_utc.strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, created_at, source_type, notes)
                VALUES (?, ?, ?, ?);
                """,
                (sid, timestamp_str, str(file_type), ""),
            )
            conn.execute(
                """
                INSERT INTO incidents (
                    incident_id, session_id, file_name, file_type, detection_type,
                    confidence, fire_coverage, smoke_coverage, visual_severity,
                    alert_level, video_timestamp, created_at, output_path,
                    original_path, growth_trend, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    inc_id,
                    sid,
                    str(file_name),
                    str(file_type),
                    str(detection_type),
                    float(confidence),
                    float(fire_coverage),
                    float(smoke_coverage),
                    str(visual_severity),
                    str(alert_level),
                    float(video_timestamp) if video_timestamp is not None else None,
                    timestamp_str,
                    str(output_path or ""),
                    str(original_path or ""),
                    str(growth_trend or ""),
                    str(notes or ""),
                ),
            )
        return inc_id

    def get_incidents(
        self,
        search_query: str | None = None,
        detection_type: str | None = None,
        severity: str | None = None,
        alert_level: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve filtered incident records sorted newest-first."""
        clauses: list[str] = []
        params: list[Any] = []

        if search_query:
            q = f"%{search_query.strip()}%"
            clauses.append("(incident_id LIKE ? OR file_name LIKE ? OR notes LIKE ?)")
            params.extend([q, q, q])
        if detection_type and detection_type != "All":
            clauses.append("detection_type = ?")
            params.append(detection_type)
        if severity and severity != "All":
            clauses.append("visual_severity = ?")
            params.append(severity)
        if alert_level and alert_level != "All":
            clauses.append("alert_level = ?")
            params.append(alert_level)
        if start_date:
            clauses.append("substr(created_at, 1, 10) >= ?")
            params.append(str(start_date))
        if end_date:
            clauses.append("substr(created_at, 1, 10) <= ?")
            params.append(str(end_date))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit is not None and limit > 0 else ""

        sql = f"""
            SELECT incident_id, session_id, file_name, file_type, detection_type,
                   confidence, fire_coverage, smoke_coverage, visual_severity,
                   alert_level, video_timestamp, created_at, output_path,
                   original_path, growth_trend, notes
            FROM incidents
            {where_sql}
            ORDER BY created_at DESC, rowid DESC
            {limit_sql};
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_incident_by_id(self, incident_id: str) -> dict[str, Any] | None:
        """Fetch a single incident record by primary ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE incident_id = ?;",
                (incident_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def delete_incidents(self, incident_ids: list[str]) -> int:
        """Delete selected incident records safely using parameterized placeholders."""
        cleaned = [str(item) for item in incident_ids if item]
        if not cleaned:
            return 0
        placeholders = ",".join("?" for _ in cleaned)
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM incidents WHERE incident_id IN ({placeholders});",
                cleaned,
            )
            return int(cursor.rowcount)

    def clear_incidents(self) -> int:
        """Delete all stored detection incident rows cleanly."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM incidents;")
            return int(cursor.rowcount)

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Compute live analytics KPIs strictly from persisted SQLite records."""
        records = self.get_incidents()
        total_files = len(records)

        fire_incidents = sum(
            1 for r in records if r["detection_type"] in {"Fire detected.", "Fire and smoke detected."}
        )
        smoke_incidents = sum(
            1 for r in records if r["detection_type"] in {"Smoke detected.", "Fire and smoke detected."}
        )
        negative_predictions = sum(
            1 for r in records if r["detection_type"] == "No fire or smoke detected."
        )

        pos_confs = [float(r["confidence"]) for r in records if float(r["confidence"]) > 0.0]
        avg_confidence = round(sum(pos_confs) / len(pos_confs), 4) if pos_confs else 0.0

        severity_counts: dict[str, int] = {}
        alert_counts: dict[str, int] = {}
        for rec in records:
            sev = str(rec["visual_severity"])
            alt = str(rec["alert_level"])
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            alert_counts[alt] = alert_counts.get(alt, 0) + 1

        return {
            "total_analyzed_files": total_files,
            "total_fire_incidents": fire_incidents,
            "total_smoke_incidents": smoke_incidents,
            "total_negative_predictions": negative_predictions,
            "average_confidence": avg_confidence,
            "severity_distribution": severity_counts,
            "alert_distribution": alert_counts,
            "recent_incidents": records[:10],
            "all_incidents": records,
        }

    def export_incidents_csv(
        self,
        records: list[dict[str, Any]] | None = None,
        output_path: Path | None = None,
    ) -> bytes:
        """Serialize incident records to UTF-8 CSV bytes (and optionally write to disk)."""
        rows = records if records is not None else self.get_incidents()
        fieldnames = [
            "incident_id",
            "session_id",
            "file_name",
            "file_type",
            "detection_type",
            "confidence",
            "fire_coverage",
            "smoke_coverage",
            "visual_severity",
            "alert_level",
            "video_timestamp",
            "created_at",
            "output_path",
            "growth_trend",
            "notes",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        csv_bytes = buf.getvalue().encode("utf-8")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(csv_bytes)
        return csv_bytes

    def start_audit_session(
        self,
        model_identifier: str,
        source_type: str,
        source_filename: str = "",
        audit_session_id: str | None = None,
    ) -> str:
        """Create and persist an audit_sessions row when a live/dataset audit begins."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        aid = audit_session_id or f"AUD-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_sessions (
                    audit_session_id, start_time, end_time, model_identifier,
                    source_type, source_filename, items_processed, fire_detections,
                    smoke_detections, evaluation_summary, avg_latency_ms, status
                )
                VALUES (?, ?, '', ?, ?, ?, 0, 0, 0, '', 0.0, 'RUNNING');
                """,
                (aid, now_str, str(model_identifier), str(source_type), str(source_filename)),
            )
        return aid

    def finish_audit_session(
        self,
        audit_session_id: str,
        items_processed: int,
        fire_detections: int,
        smoke_detections: int,
        avg_latency_ms: float,
        evaluation_summary: str = "",
        status: str = "COMPLETED",
    ) -> None:
        """Finalize an audit_sessions row with actual observed counts and latency."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE audit_sessions
                SET end_time = ?, items_processed = ?, fire_detections = ?,
                    smoke_detections = ?, evaluation_summary = ?, avg_latency_ms = ?,
                    status = ?
                WHERE audit_session_id = ?;
                """,
                (
                    now_str,
                    int(items_processed),
                    int(fire_detections),
                    int(smoke_detections),
                    str(evaluation_summary),
                    float(avg_latency_ms),
                    str(status),
                    str(audit_session_id),
                ),
            )

    def log_model_health_check(
        self,
        model_path: str,
        status: str,
        architecture: str = "",
        class_names: str = "",
        execution_device: str = "",
        avg_latency_ms: float | None = None,
        model_tier: str = "",
        details: str = "",
    ) -> str:
        """Persist a ModelHealthReport snapshot into model_health_checks."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        chk_id = f"CHK-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_health_checks (
                    check_id, checked_at, model_path, status, architecture,
                    class_names, execution_device, avg_latency_ms, model_tier, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    chk_id,
                    now_str,
                    str(model_path),
                    str(status),
                    str(architecture),
                    str(class_names),
                    str(execution_device),
                    float(avg_latency_ms) if avg_latency_ms is not None else None,
                    str(model_tier),
                    str(details),
                ),
            )
        return chk_id

    def insert_dataset_audit_results(
        self,
        audit_session_id: str,
        rows: list[dict[str, Any]],
    ) -> int:
        """Batch-insert labeled dataset audit rows for an audit session."""
        if not rows:
            return 0
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        params: list[tuple[Any, ...]] = []
        for r in rows:
            res_id = f"DAR-{uuid.uuid4().hex[:10].upper()}"
            params.append(
                (
                    res_id,
                    audit_session_id,
                    str(r.get("image_name", "")),
                    str(r.get("category", "")),
                    str(r.get("ground_truth_class", "")),
                    str(r.get("predicted_class", "")),
                    int(r.get("matched_detections", 0)),
                    int(r.get("false_positives", 0)),
                    int(r.get("false_negatives", 0)),
                    float(r.get("confidence", 0.0)),
                    float(r.get("iou", 0.0)),
                    float(r.get("inference_time_ms", 0.0)),
                    str(r.get("audit_status", "NOT EVALUATED")),
                    str(r.get("annotated_path", "")),
                    now_str,
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO dataset_audit_results (
                    result_id, audit_session_id, image_name, category,
                    ground_truth_class, predicted_class, matched_detections,
                    false_positives, false_negatives, confidence, iou,
                    inference_time_ms, audit_status, snapshot_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                params,
            )
        return len(params)

    def insert_detection_event(
        self,
        audit_session_id: str,
        frame_or_item: str,
        detection_class: str,
        confidence: float,
        timestamp_sec: float | None = None,
        fire_coverage: float = 0.0,
        smoke_coverage: float = 0.0,
        inference_latency_ms: float = 0.0,
    ) -> str:
        """Insert a representative detection event linked to an audit session."""
        ev_id = f"DET-{uuid.uuid4().hex[:10].upper()}"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO detection_events (
                    event_id, audit_session_id, frame_or_item, timestamp_sec,
                    detection_class, confidence, fire_coverage, smoke_coverage,
                    inference_latency_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    ev_id,
                    audit_session_id,
                    str(frame_or_item),
                    float(timestamp_sec) if timestamp_sec is not None else None,
                    str(detection_class),
                    float(confidence),
                    float(fire_coverage),
                    float(smoke_coverage),
                    float(inference_latency_ms),
                    now_str,
                ),
            )
        return ev_id

    def insert_alert_incident(
        self,
        audit_session_id: str,
        source_filename: str,
        detected_class: str,
        confidence: float,
        alert_level: str,
        video_timestamp: float | None = None,
        fire_coverage: float = 0.0,
        smoke_coverage: float = 0.0,
        snapshot_path: str = "",
        alert_id: str | None = None,
    ) -> str:
        """Persist a cooldown-verified alert incident with optional snapshot path."""
        now_utc = datetime.now(timezone.utc)
        aid = alert_id or f"ALT-{now_utc.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_incidents (
                    alert_id, audit_session_id, source_filename, detected_class,
                    confidence, video_timestamp, fire_coverage, smoke_coverage,
                    alert_level, snapshot_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    aid,
                    audit_session_id,
                    str(source_filename),
                    str(detected_class),
                    float(confidence),
                    float(video_timestamp) if video_timestamp is not None else None,
                    float(fire_coverage),
                    float(smoke_coverage),
                    str(alert_level),
                    str(snapshot_path or ""),
                    now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        return aid

    def insert_model_evaluation_summary(
        self,
        model_identifier: str,
        dataset_split: str,
        sample_count: int,
        precision: float | None,
        recall: float | None,
        f1_score: float | None,
        map50: float | None = None,
        map50_95: float | None = None,
        false_alarm_rate: float | None = None,
        avg_inference_ms: float | None = None,
        avg_fps: float | None = None,
        details_json: str = "",
    ) -> str:
        """Store a standardized test/audit evaluation summary row."""
        now_utc = datetime.now(timezone.utc)
        eval_id = f"EVL-{now_utc.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_evaluation_summaries (
                    eval_id, evaluated_at, model_identifier, dataset_split,
                    sample_count, precision, recall, f1_score, map50, map50_95,
                    false_alarm_rate, avg_inference_ms, avg_fps, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    eval_id,
                    now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    str(model_identifier),
                    str(dataset_split),
                    int(sample_count),
                    precision,
                    recall,
                    f1_score,
                    map50,
                    map50_95,
                    false_alarm_rate,
                    avg_inference_ms,
                    avg_fps,
                    str(details_json),
                ),
            )
        return eval_id

    def get_audit_sessions(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query persisted audit_sessions ordered newest-first."""
        clauses: list[str] = []
        params: list[Any] = []
        if source_type and source_type != "All":
            clauses.append("source_type = ?")
            params.append(source_type)
        if status and status != "All":
            clauses.append("status = ?")
            params.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM audit_sessions
            {where_sql}
            ORDER BY start_time DESC, rowid DESC
            LIMIT {int(limit)};
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_model_health_checks(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return recent model health diagnostics records."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM model_health_checks ORDER BY checked_at DESC, rowid DESC LIMIT {int(limit)};"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_dataset_audit_results(self, audit_session_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Retrieve stored per-image dataset audit rows."""
        with self._connect() as conn:
            if audit_session_id:
                rows = conn.execute(
                    f"SELECT * FROM dataset_audit_results WHERE audit_session_id = ? ORDER BY rowid ASC LIMIT {int(limit)};",
                    (audit_session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM dataset_audit_results ORDER BY created_at DESC, rowid DESC LIMIT {int(limit)};"
                ).fetchall()
        return [dict(row) for row in rows]

    def get_alert_incidents(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve persisted alert_incidents ordered newest-first."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM alert_incidents ORDER BY created_at DESC, rowid DESC LIMIT {int(limit)};"
            ).fetchall()
        return [dict(row) for row in rows]

    def export_audit_sessions_csv(self, records: list[dict[str, Any]] | None = None) -> bytes:
        """Export audit_sessions records as UTF-8 CSV bytes."""
        rows = records if records is not None else self.get_audit_sessions(limit=1000)
        fieldnames = [
            "audit_session_id",
            "start_time",
            "end_time",
            "model_identifier",
            "source_type",
            "source_filename",
            "items_processed",
            "fire_detections",
            "smoke_detections",
            "evaluation_summary",
            "avg_latency_ms",
            "status",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue().encode("utf-8")
