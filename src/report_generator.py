"""Phase 13 — Automated PDF and CSV Incident Report Generation.

Uses ReportLab to generate formatted academic incident reports including
metadata, original and annotated images (when available), video growth graphs,
incident narrative summaries, and explicit academic prototype disclaimers.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config


def _fit_reportlab_image(image_path: Path, max_w: float, max_h: float) -> RLImage | None:
    """Create a proportionally scaled ReportLab Image flowable if file is valid."""
    if not image_path.is_file():
        return None
    try:
        with PILImage.open(image_path) as img:
            w, h = img.size
        if w <= 0 or h <= 0:
            return None
        scale = min(max_w / w, max_h / h)
        return RLImage(str(image_path), width=w * scale, height=h * scale)
    except Exception:
        return None


def generate_pdf_report(
    incident: dict[str, Any],
    output_path: Path | None = None,
    growth_chart_path: Path | None = None,
) -> tuple[bytes, Path]:
    """Generate a formatted ReportLab PDF report for a single incident record."""
    inc_id = str(incident.get("incident_id", "INC-UNKNOWN"))
    target_dir = config.REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(output_path) if output_path is not None else (target_dir / f"{inc_id}_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"FireGuard AI Incident Report - {inc_id}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FireGuardTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#B91C1C"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "FireGuardSub",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "FireGuardSection",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "FireGuardBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
    )
    disclaimer_style = ParagraphStyle(
        "FireGuardDisclaimer",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#7F1D1D"),
        backColor=colors.HexColor("#FEF2F2"),
        borderPadding=8,
    )

    elements: list[Any] = [
        Paragraph(f"{config.PROJECT_NAME} — Incident Analysis Report", title_style),
        Paragraph(config.PROJECT_SUBTITLE, subtitle_style),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#EA580C")),
        Spacer(1, 10),
        Paragraph("1. Incident Metadata", section_style),
    ]

    video_ts = incident.get("video_timestamp")
    ts_display = f"{float(video_ts):.2f} s" if video_ts is not None else "N/A (Single Image)"
    growth_display = str(incident.get("growth_trend") or "N/A")

    meta_rows = [
        ["Incident ID", inc_id, "Session ID", str(incident.get("session_id", "N/A"))],
        ["File Name", str(incident.get("file_name", "N/A")), "File Type", str(incident.get("file_type", "N/A"))],
        ["Date & Time", str(incident.get("created_at", "N/A")), "Alert Level", str(incident.get("alert_level", "N/A"))],
        ["Detection Category", str(incident.get("detection_type", "N/A")), "Prediction Confidence", f"{float(incident.get('confidence', 0.0)):.2%}"],
        ["Visual Severity", str(incident.get("visual_severity", "N/A")), "Video Timestamp", ts_display],
        ["Fire Coverage", f"{float(incident.get('fire_coverage', 0.0)):.2f}%", "Smoke Coverage", f"{float(incident.get('smoke_coverage', 0.0)):.2f}%"],
        ["Growth Trend", growth_display, "Prototype Version", config.PROJECT_VERSION],
    ]

    meta_table = Table(meta_rows, colWidths=[1.35 * inch, 2.0 * inch, 1.45 * inch, 2.0 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(meta_table)

    # Visual Evidence Section (Original + Detection Snapshot)
    orig_path_str = str(incident.get("original_path") or "")
    out_path_str = str(incident.get("output_path") or "")
    orig_rl = _fit_reportlab_image(Path(orig_path_str), 3.1 * inch, 2.3 * inch) if orig_path_str else None
    snap_rl = _fit_reportlab_image(Path(out_path_str), 3.1 * inch, 2.3 * inch) if out_path_str else None

    if orig_rl is not None or snap_rl is not None:
        elements.append(Paragraph("2. Visual Evidence", section_style))
        img_headers: list[str] = []
        img_cells: list[Any] = []
        if orig_rl is not None:
            img_headers.append("Original Input")
            img_cells.append(orig_rl)
        if snap_rl is not None:
            img_headers.append("Detection Snapshot (Fire: Red, Smoke: Orange)")
            img_cells.append(snap_rl)
        col_w = [3.35 * inch] * len(img_cells)
        img_table = Table([img_headers, img_cells], colWidths=col_w)
        img_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 1), (-1, 1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(img_table)

    # Video Growth Graph if applicable
    if growth_chart_path is not None:
        growth_rl = _fit_reportlab_image(Path(growth_chart_path), 6.6 * inch, 2.8 * inch)
        if growth_rl is not None:
            elements.append(Paragraph("3. Temporal Fire & Smoke Coverage Progression", section_style))
            elements.append(growth_rl)

    elements.append(Paragraph("4. Incident Summary", section_style))
    summary_notes = str(incident.get("notes") or "").strip()
    summary_text = (
        f"Analysis of <b>{incident.get('file_name', 'input')}</b> ({incident.get('file_type', 'media')}) "
        f"resulted in status: <b>{incident.get('detection_type', 'N/A')}</b> with maximum confidence "
        f"<b>{float(incident.get('confidence', 0.0)):.2%}</b>. Approximate 2D visual coverage was estimated at "
        f"<b>{float(incident.get('fire_coverage', 0.0)):.2f}%</b> for fire and "
        f"<b>{float(incident.get('smoke_coverage', 0.0)):.2f}%</b> for smoke "
        f"(Visual Severity: <b>{incident.get('visual_severity', 'N/A')}</b>, "
        f"Alert Level: <b>{incident.get('alert_level', 'N/A')}</b>)."
    )
    if summary_notes:
        summary_text += f"<br/><br/><b>Additional Observations:</b> {summary_notes}"
    elements.append(Paragraph(summary_text, body_style))

    elements.append(Paragraph("5. Prototype Limitations & Disclaimer", section_style))
    limitations_text = (
        f"{config.ACADEMIC_DISCLAIMER} Visual coverage percentages are derived from 2D bounding-box union "
        "areas and do NOT measure burned area, temperature, heat release rate, or physical fire spread. "
        "A negative detection does NOT guarantee that an environment is safe."
    )
    elements.append(Paragraph(limitations_text, disclaimer_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    pdf_path.write_bytes(pdf_bytes)
    return pdf_bytes, pdf_path


def generate_csv_report(
    records: list[dict[str, Any]],
    output_path: Path | None = None,
) -> tuple[bytes, Path]:
    """Export incident history records as UTF-8 CSV bytes and file."""
    target_path = (
        Path(output_path)
        if output_path is not None
        else (config.REPORTS_DIR / "fireguard_incident_history.csv")
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
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
    for rec in records:
        writer.writerow(rec)
    csv_bytes = buf.getvalue().encode("utf-8")
    target_path.write_bytes(csv_bytes)
    return csv_bytes, target_path


def generate_audit_pdf_report(
    audit_session: dict[str, Any],
    health_report: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    detection_examples: list[Path] | None = None,
    failed_examples: list[Path] | None = None,
    alert_history: list[dict[str, Any]] | None = None,
    confusion_matrix_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[bytes, Path]:
    """Step 14 — Generate comprehensive PDF Audit Report distinguishing software checks,
    AI model evaluation, and non-certification prototype limitations.
    """
    session_id = str(audit_session.get("audit_session_id", "AUDIT-SESSION"))
    target_dir = config.REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(output_path) if output_path is not None else (target_dir / f"{session_id}_audit_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"FireGuard AI Audit Report - {session_id}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("AudTitle", parent=styles["Heading1"], fontSize=17, textColor=colors.HexColor("#B91C1C"), spaceAfter=4)
    sub_style = ParagraphStyle("AudSub", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#475569"), spaceAfter=8)
    sec_style = ParagraphStyle("AudSec", parent=styles["Heading2"], fontSize=11.5, textColor=colors.HexColor("#0F172A"), spaceBefore=9, spaceAfter=5)
    body_style = ParagraphStyle("AudBody", parent=styles["Normal"], fontSize=9.2, leading=13, textColor=colors.HexColor("#1E293B"))
    warn_style = ParagraphStyle("AudWarn", parent=styles["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#7F1D1D"), backColor=colors.HexColor("#FEF2F2"), borderPadding=8)

    elements: list[Any] = [
        Paragraph(f"{config.PROJECT_NAME} — System & Model Audit Report", title_style),
        Paragraph(config.PROJECT_SUBTITLE, sub_style),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#EA580C")),
        Spacer(1, 8),
        Paragraph("1. Verification vs. Evaluation vs. Safety Scope", sec_style),
        Paragraph(
            "<b>Software Functionality Verification:</b> Confirms input validation, YOLO weight loading, "
            "live video frame streaming, SQLite persistence, alert cooldowns, and PDF/CSV export.<br/>"
            "<b>AI Model Performance Evaluation:</b> Reports actual precision, recall, F1, IoU matching, "
            "and latency measured on labeled datasets.<br/>"
            "<b>Safety Certification Notice:</b> This prototype is NOT a certified fire safety or life-safety "
            "system and provides zero real-world emergency certification.",
            warn_style,
        ),
        Paragraph("2. Audit Session & Model Health Summary", sec_style),
    ]

    hr = health_report or {}
    ds = dataset_info or {}
    met = metrics or {}

    sess_rows = [
        ["Audit Session ID", session_id, "Source Type", str(audit_session.get("source_type", "N/A"))],
        ["Start Time", str(audit_session.get("start_time", "N/A")), "Status", str(audit_session.get("status", "N/A"))],
        ["Model Health Status", str(hr.get("status", "Unknown")), "Model Tier", str(hr.get("model_tier", "Unknown"))],
        ["Model Architecture", str(hr.get("architecture", "N/A")), "Execution Device", str(hr.get("execution_device", "N/A"))],
        ["Dataset Source", str(ds.get("source", "D-Fire / Local Dataset")), "Samples Evaluated", str(met.get("evaluated_sample_count", audit_session.get("items_processed", 0)))],
        ["Avg Inference Latency", f"{met.get('avg_inference_ms', audit_session.get('avg_latency_ms', 0.0))} ms", "Avg Inference FPS", str(met.get("avg_inference_fps", met.get("fps", "N/A")))],
    ]
    t_sess = Table(sess_rows, colWidths=[1.4 * inch, 2.0 * inch, 1.45 * inch, 2.05 * inch])
    t_sess.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(t_sess)

    elements.append(Paragraph("3. Actual Observed Performance Metrics", sec_style))
    if met:
        met_rows = [
            ["Metric", "Observed Value", "Metric", "Observed Value"],
            ["Precision", str(met.get("precision", "N/A")), "Recall", str(met.get("recall", "N/A"))],
            ["F1-Score", str(met.get("f1_score", "N/A")), "False Alarm Rate (Neg)", str(met.get("negative_false_alarm_rate", met.get("false_alarm_evaluation", {}).get("enhanced_fpr", "N/A")))],
            ["mAP@50", str(met.get("map50", "N/A")), "mAP@50-95", str(met.get("map50_95", "N/A"))],
            ["True Positives / PASS", str(met.get("true_positives", met.get("pass_count", "N/A"))), "FP / FN / FAIL", f"{met.get('false_positives', 'N/A')} / {met.get('false_negatives', 'N/A')} / {met.get('fail_count', 'N/A')}"],
        ]
        t_met = Table(met_rows, colWidths=[1.6 * inch, 1.8 * inch, 1.65 * inch, 1.85 * inch])
        t_met.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(t_met)
    else:
        elements.append(Paragraph("No labeled evaluation metrics recorded in this audit session.", body_style))

    if detection_examples:
        valid_ex = [p for p in detection_examples if Path(p).is_file()][:2]
        if valid_ex:
            elements.append(Paragraph("4. Detection Examples", sec_style))
            for img_p in valid_ex:
                rl_img = _fit_reportlab_image(Path(img_p), 6.2 * inch, 2.2 * inch)
                if rl_img is not None:
                    elements.append(rl_img)
                    elements.append(Spacer(1, 4))

    if failed_examples:
        valid_fail = [p for p in failed_examples if Path(p).is_file()][:2]
        if valid_fail:
            elements.append(Paragraph("5. Failed / Misclassified Examples", sec_style))
            for img_p in valid_fail:
                rl_img = _fit_reportlab_image(Path(img_p), 6.2 * inch, 2.2 * inch)
                if rl_img is not None:
                    elements.append(rl_img)
                    elements.append(Spacer(1, 4))

    if confusion_matrix_path is not None and Path(confusion_matrix_path).is_file():
        cm_img = _fit_reportlab_image(Path(confusion_matrix_path), 5.4 * inch, 3.4 * inch)
        if cm_img is not None:
            elements.append(Paragraph("6. Confusion Matrix", sec_style))
            elements.append(cm_img)

    if alert_history:
        elements.append(Paragraph("7. Alert History", sec_style))
        alt_rows = [["Alert ID", "Level", "Class", "Confidence", "Timestamp"]]
        for al in alert_history[:12]:
            alt_rows.append(
                [
                    str(al.get("alert_id", "")),
                    str(al.get("alert_level", "")),
                    str(al.get("detected_class", al.get("alert_type", ""))),
                    f"{float(al.get('confidence', 0.0)):.1%}",
                    str(al.get("created_at", al.get("timestamp", ""))),
                ]
            )
        t_alt = Table(alt_rows, colWidths=[1.6 * inch, 0.95 * inch, 1.3 * inch, 0.95 * inch, 2.1 * inch])
        t_alt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7F1D1D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(t_alt)

    elements.append(Paragraph("8. Final Audit Summary & Model Limitations", sec_style))
    eval_summary_txt = str(audit_session.get("evaluation_summary") or "Audit completed using actual trained YOLO11 inference.")
    elements.append(
        Paragraph(
            f"{eval_summary_txt}<br/><br/>"
            f"<b>Model Limitations:</b> {config.ACADEMIC_DISCLAIMER} Smoke-test models trained on small "
            "subsets or CPU require full 50-epoch GPU training on the complete D-Fire dataset for production-grade generalization.",
            body_style,
        )
    )

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    pdf_path.write_bytes(pdf_bytes)
    return pdf_bytes, pdf_path
