"""Step 7 — Bounded In-Memory Live Audit Console Logger.

Captures real execution events across model diagnostics, live video monitoring,
dataset auditing, and alert triggers without unbounded memory growth.
"""

from __future__ import annotations

import csv
import io
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

LOG_LEVELS: tuple[str, ...] = ("INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


@dataclass
class AuditLogEntry:
    """Single structured event recorded in the Live Audit log stream."""

    timestamp: str
    level: str
    event_type: str
    source_item: str
    detection_class: str = "-"
    confidence: float | None = None
    inference_latency_ms: float | None = None
    alert_level: str = "-"
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveAuditLogger:
    """Bounded circular buffer of operational audit log events."""

    def __init__(self, max_entries: int = 500) -> None:
        self.max_entries = max(50, int(max_entries))
        self._entries: deque[AuditLogEntry] = deque(maxlen=self.max_entries)

    def log(
        self,
        level: str,
        event_type: str,
        source_item: str = "-",
        detection_class: str = "-",
        confidence: float | None = None,
        inference_latency_ms: float | None = None,
        alert_level: str = "-",
        details: str = "",
    ) -> AuditLogEntry:
        """Append a real event entry into the bounded log queue."""
        clean_level = level.upper() if level.upper() in LOG_LEVELS else "INFO"
        entry = AuditLogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            level=clean_level,
            event_type=str(event_type),
            source_item=str(source_item),
            detection_class=str(detection_class),
            confidence=round(float(confidence), 4) if confidence is not None else None,
            inference_latency_ms=(
                round(float(inference_latency_ms), 2) if inference_latency_ms is not None else None
            ),
            alert_level=str(alert_level),
            details=str(details),
        )
        self._entries.append(entry)
        return entry

    def get_entries(
        self,
        search_text: str | None = None,
        event_type: str | None = None,
        level: str | None = None,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Filter retained audit logs by substring, event type, or severity level."""
        items = list(self._entries)
        if newest_first:
            items.reverse()

        results: list[dict[str, Any]] = []
        q = search_text.strip().lower() if search_text else ""
        for item in items:
            if level and level != "All" and item.level != level:
                continue
            if event_type and event_type != "All" and item.event_type != event_type:
                continue
            if q:
                blob = (
                    f"{item.event_type} {item.source_item} {item.detection_class} "
                    f"{item.alert_level} {item.details}"
                ).lower()
                if q not in blob:
                    continue
            results.append(item.to_dict())
        return results

    def unique_event_types(self) -> list[str]:
        """Return sorted unique event types currently retained."""
        return sorted({entry.event_type for entry in self._entries})

    def clear(self) -> None:
        """Clear all retained in-memory audit log entries."""
        self._entries.clear()

    def export_csv_bytes(self, entries: list[dict[str, Any]] | None = None) -> bytes:
        """Serialize audit log records to UTF-8 CSV bytes."""
        rows = entries if entries is not None else self.get_entries(newest_first=False)
        fieldnames = [
            "timestamp",
            "level",
            "event_type",
            "source_item",
            "detection_class",
            "confidence",
            "inference_latency_ms",
            "alert_level",
            "details",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue().encode("utf-8")

    def __len__(self) -> int:
        return len(self._entries)
