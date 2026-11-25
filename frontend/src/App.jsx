import React, { useEffect, useRef, useState } from "react";
import {
  analyzeMedia,
  clearHistory,
  fetchAbout,
  fetchHealth,
  fetchHistory,
  resolveMediaUrl,
} from "./api.js";

const PAGES = [
  { id: "home", label: "Home" },
  { id: "live-audit", label: "Live Audit" },
  { id: "history", label: "Detection History" },
  { id: "about", label: "About Project" },
];

const ACCEPTED_EXTS = [".jpg", ".jpeg", ".png", ".mp4"];

function getExt(filename) {
  const idx = filename.lastIndexOf(".");
  return idx >= 0 ? filename.slice(idx).toLowerCase() : "";
}

function severityCssClass(severity, finalResult) {
  if ((finalResult || "").toLowerCase() === "analysis failed") return "sev-red";
  const key = (severity || "").trim().toLowerCase();
  if (key === "low") return "sev-yellow";
  if (key === "medium") return "sev-orange";
  if (key === "high") return "sev-red";
  return "sev-green";
}

function CoverageCard({ title, percentage, color, subtitle }) {
  const safePct = Math.max(0, Math.min(100, Number(percentage || 0)));
  return (
    <div className="panel-card">
      <div className="coverage-label">{title}</div>
      <div className="coverage-value">{safePct.toFixed(2)}%</div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${safePct}%`, backgroundColor: color }}
        />
      </div>
      {subtitle ? <div className="disclaimer-note">{subtitle}</div> : null}
    </div>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState("home");
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState("");

  // Live Audit state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [auditResult, setAuditResult] = useState(null);
  const [auditError, setAuditError] = useState("");
  const abortRef = useRef(null);
  const fileInputRef = useRef(null);

  // History state
  const [historyRows, setHistoryRows] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyMsg, setHistoryMsg] = useState("");
  const [confirmClear, setConfirmClear] = useState(false);

  // About state
  const [aboutData, setAboutData] = useState(null);

  async function loadHealth(forceRefresh = false) {
    setHealthLoading(true);
    setHealthError("");
    try {
      const data = await fetchHealth(forceRefresh);
      setHealth(data);
    } catch (err) {
      setHealth(null);
      setHealthError(err.message || "Could not reach backend API.");
    } finally {
      setHealthLoading(false);
    }
  }

  async function loadHistory() {
    setHistoryLoading(true);
    setHistoryMsg("");
    try {
      const data = await fetchHistory();
      setHistoryRows(data.records || []);
    } catch (err) {
      setHistoryMsg(err.message || "Failed to load detection history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadAbout() {
    try {
      const data = await fetchAbout();
      setAboutData(data);
    } catch {
      // Non-fatal; About page still shows baseline info
    }
  }

  useEffect(() => {
    loadHealth(false);
    loadAbout();
  }, []);

  useEffect(() => {
    if (activePage === "history") {
      loadHistory();
    } else if (activePage === "about" && !aboutData) {
      loadAbout();
    }
  }, [activePage]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleFileChange(file) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (!file) {
      setSelectedFile(null);
      setPreviewUrl("");
      setAuditResult(null);
      setAuditError("");
      return;
    }

    const ext = getExt(file.name);
    if (!ACCEPTED_EXTS.includes(ext)) {
      setSelectedFile(null);
      setPreviewUrl("");
      setAuditResult({
        status: "failed",
        final_result: "Analysis Failed",
        visual_severity: "Analysis Failed",
      });
      setAuditError("Unsupported file format. Please upload JPG, JPEG, PNG, or MP4.");
      return;
    }

    // Clear stale results immediately whenever a new file is selected
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAuditResult(null);
    setAuditError("");
  }

  async function handleAnalyze() {
    if (!selectedFile || analyzing) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setAnalyzing(true);
    setAuditError("");
    setAuditResult(null);

    try {
      const res = await analyzeMedia(selectedFile, controller.signal);
      setAuditResult(res);
    } catch (err) {
      if (err.name === "AbortError") {
        setAuditError("Analysis request cancelled.");
      } else {
        setAuditError(err.message || "Could not complete detection.");
        setAuditResult({
          status: "failed",
          final_result: "Analysis Failed",
          visual_severity: "Analysis Failed",
        });
      }
    } finally {
      setAnalyzing(false);
      abortRef.current = null;
    }
  }

  function handleStop() {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }

  async function handleClearHistory() {
    if (!confirmClear) return;
    try {
      const res = await clearHistory();
      setHistoryRows([]);
      setConfirmClear(false);
      setHistoryMsg(`Cleared ${res.deleted || 0} detection record(s).`);
    } catch (err) {
      setHistoryMsg(err.message || "Failed to clear detection history.");
    }
  }

  const isModelReady = Boolean(health?.model_ready);
  const showSubsetNote =
    isModelReady &&
    ["SMOKE_TEST", "SUBSET_CPU_16EP"].includes(health?.model_tier || "");

  const isVideoSelected =
    selectedFile && getExt(selectedFile.name) === ".mp4";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-label">Navigation</div>
        {PAGES.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`nav-btn ${activePage === p.id ? "active" : ""}`}
            onClick={() => setActivePage(p.id)}
          >
            {p.label}
          </button>
        ))}

        <div className="sidebar-footer">
          <strong>FireGuard AI</strong>
          <div>Vercel React Frontend + Render FastAPI YOLO11n Backend</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <div className="brand-title">&#128293; FireGuard AI</div>
            <div className="brand-sub">AI-Based Fire and Smoke Detection</div>
          </div>

          <div className="pill-group">
            {healthLoading ? (
              <span className="model-pill pill-loading">Checking backend...</span>
            ) : isModelReady ? (
              <span className="model-pill pill-ready">Model ready</span>
            ) : (
              <>
                <span className="model-pill pill-unavailable">Model unavailable</span>
                <button
                  type="button"
                  className="btn btn-secondary btn-small"
                  onClick={() => loadHealth(true)}
                >
                  Retry
                </button>
              </>
            )}
          </div>
        </header>

        {healthError ? (
          <div className="error-box">
            <strong>Backend Connection Issue:</strong> {healthError}. On Render&apos;s
            free tier, the API service may take ~30–50 seconds to wake up from inactivity.
          </div>
        ) : null}

        {showSubsetNote ? (
          <div className="smoke-note">
            Note: Balanced D-Fire subset CPU YOLO11n weights are active for academic
            demonstration. Train on the complete 21,527-image D-Fire dataset in{" "}
            <code>notebooks/model_training.ipynb</code> on Google Colab GPU for full benchmark
            accuracy.
          </div>
        ) : null}

        {activePage === "home" ? (
          <section className="page-header">
            <h1>FireGuard AI</h1>
            <p className="page-sub">
              AI-Based Fire and Smoke Detection — Intelligent Software Prototype
            </p>
            <p style={{ lineHeight: 1.6, color: "#cbd5e1", maxWidth: "840px" }}>
              FireGuard AI uses a trained <strong>YOLO11n</strong> deep learning object
              detection model to identify visible fire and smoke from uploaded images and
              prerecorded MP4 videos. It runs purely in software on a FastAPI backend hosted
              on Render paired with this React frontend on Vercel.
            </p>

            <div className="card-grid-3">
              <div className="feature-card feature-fire">
                <h3>Fire Detection</h3>
                <p>
                  Detects visible flame regions, highlights them with bounding boxes, and
                  measures approximate visual fire coverage percentage.
                </p>
              </div>
              <div className="feature-card feature-smoke">
                <h3>Smoke Detection</h3>
                <p>
                  Detects smoke plumes independently or alongside fire to support early visual
                  fire and smoke hazard identification.
                </p>
              </div>
              <div className="feature-card feature-software">
                <h3>Software-Only Detection</h3>
                <p>
                  Requires zero hardware sensors, Arduino, IoT modules, or thermal cameras —
                  works directly on uploaded JPG, PNG, and MP4 files.
                </p>
              </div>
            </div>

            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setActivePage("live-audit")}
            >
              Start Detection &rarr;
            </button>
          </section>
        ) : null}

        {activePage === "live-audit" ? (
          <section className="page-header">
            <h1>Live Audit</h1>
            <p className="page-sub">Upload an image or video to detect fire and smoke.</p>

            <div
              className={`dropzone ${isDragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                const f = e.dataTransfer?.files?.[0];
                if (f) handleFileChange(f);
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.mp4"
                style={{ display: "none" }}
                onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
              />
              <div className="dropzone-title">Drag and drop image or video, or click to browse</div>
              <div className="dropzone-sub">Supported formats: JPG, JPEG, PNG, MP4</div>

              {selectedFile ? (
                <div
                  className="selected-file-chip"
                  onClick={(e) => e.stopPropagation()}
                >
                  <span>
                    {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </span>
                  <button
                    type="button"
                    className="clear-file-btn"
                    title="Clear file"
                    onClick={() => {
                      if (fileInputRef.current) fileInputRef.current.value = "";
                      handleFileChange(null);
                    }}
                  >
                    &times;
                  </button>
                </div>
              ) : null}
            </div>

            <div className="action-row">
              <button
                type="button"
                className="btn btn-primary"
                disabled={!selectedFile || !isModelReady || analyzing}
                onClick={handleAnalyze}
              >
                {analyzing ? "Analyzing with YOLO11n..." : "Analyze Fire & Smoke"}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!analyzing && !isVideoSelected}
                onClick={handleStop}
              >
                Stop
              </button>
            </div>

            {auditError ? (
              <div className="error-box">
                <strong>Analysis Failed:</strong> {auditError}
              </div>
            ) : null}

            {auditResult && auditResult.status === "failed" ? (
              <div className="result-banner sev-red">
                <div style={{ fontSize: "1.06rem", fontWeight: 700, marginBottom: "0.2rem" }}>
                  Final result: Analysis Failed
                </div>
                <div style={{ fontSize: "0.9rem" }}>
                  Visual severity: <strong>Analysis Failed</strong> &nbsp;|&nbsp; Detection
                  confidence: <strong>N/A</strong>
                </div>
              </div>
            ) : null}

            {!auditResult && previewUrl ? (
              <div className="panel-card" style={{ marginBottom: "1rem" }}>
                <div className="media-box">
                  {isVideoSelected ? (
                    <video src={previewUrl} controls playsInline />
                  ) : (
                    <img src={previewUrl} alt="Selected preview" />
                  )}
                  <div className="media-caption">
                    Ready to analyze: <strong>{selectedFile?.name}</strong>
                  </div>
                </div>
              </div>
            ) : null}

            {auditResult && auditResult.status === "completed" ? (
              <div>
                <div className="card-grid-2">
                  <div className="media-box">
                    {auditResult.media_type === "image" ? (
                      <img
                        src={resolveMediaUrl(auditResult.original_media_url)}
                        alt="Original uploaded media"
                      />
                    ) : (
                      <video
                        src={resolveMediaUrl(auditResult.original_media_url)}
                        controls
                        playsInline
                      />
                    )}
                    <div className="media-caption">
                      Original {auditResult.media_type}
                    </div>
                  </div>

                  <div className="media-box">
                    {auditResult.media_type === "image" ? (
                      <img
                        src={resolveMediaUrl(auditResult.analyzed_media_url)}
                        alt="AI analyzed image with bounding boxes"
                      />
                    ) : (
                      <video
                        src={resolveMediaUrl(auditResult.analyzed_media_url)}
                        controls
                        playsInline
                      />
                    )}
                    <div className="media-caption">
                      AI analyzed {auditResult.media_type} (Red: Fire | Orange: Smoke)
                    </div>
                  </div>
                </div>

                {auditResult.media_type === "video" && auditResult.snapshot_url ? (
                  <div className="media-box" style={{ marginBottom: "1rem" }}>
                    <img
                      src={resolveMediaUrl(auditResult.snapshot_url)}
                      alt="Peak annotated video frame"
                    />
                    <div className="media-caption">
                      Peak annotated video frame snapshot (frames processed:{" "}
                      {auditResult.frames_processed})
                    </div>
                  </div>
                ) : null}

                <div className="card-grid-2">
                  <CoverageCard
                    title={
                      auditResult.media_type === "video"
                        ? "Fire affected region (Max)"
                        : "Fire affected region"
                    }
                    percentage={auditResult.fire_pct}
                    color="#ef4444"
                    subtitle={
                      auditResult.media_type === "video"
                        ? `Average across frames: ${Number(auditResult.avg_fire_pct || 0).toFixed(2)}% | Combined max: ${Number(auditResult.combined_pct || 0).toFixed(2)}%`
                        : `Combined visible region: ${Number(auditResult.combined_pct || 0).toFixed(2)}%`
                    }
                  />
                  <CoverageCard
                    title={
                      auditResult.media_type === "video"
                        ? "Smoke affected region (Max)"
                        : "Smoke affected region"
                    }
                    percentage={auditResult.smoke_pct}
                    color="#f97316"
                    subtitle={
                      auditResult.media_type === "video"
                        ? `Average across frames: ${Number(auditResult.avg_smoke_pct || 0).toFixed(2)}%`
                        : "Approximate 2D visible area"
                    }
                  />
                </div>

                <div className="card-grid-4">
                  <div className="metric-card">
                    <div className="metric-label">Fire detected</div>
                    <div className="metric-value">{auditResult.has_fire ? "Yes" : "No"}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Smoke detected</div>
                    <div className="metric-value">{auditResult.has_smoke ? "Yes" : "No"}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Detection confidence</div>
                    <div className="metric-value">
                      {auditResult.confidence_display || "N/A"}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Visual severity</div>
                    <div className="metric-value">{auditResult.visual_severity}</div>
                  </div>
                </div>

                <div
                  className={`result-banner ${severityCssClass(
                    auditResult.visual_severity,
                    auditResult.final_result
                  )}`}
                >
                  <div style={{ fontSize: "1.06rem", fontWeight: 700, marginBottom: "0.2rem" }}>
                    Final result: {auditResult.final_result}
                  </div>
                  <div style={{ fontSize: "0.9rem" }}>
                    Visual severity: <strong>{auditResult.visual_severity}</strong> &nbsp;|&nbsp;
                    Detection confidence:{" "}
                    <strong>{auditResult.confidence_display || "N/A"}</strong>
                    {auditResult.fire_confidence > 0 ? (
                      <>
                        {" "}
                        &nbsp;|&nbsp; Fire:{" "}
                        <strong>{(auditResult.fire_confidence * 100).toFixed(1)}%</strong>
                      </>
                    ) : null}
                    {auditResult.smoke_confidence > 0 ? (
                      <>
                        {" "}
                        &nbsp;|&nbsp; Smoke:{" "}
                        <strong>{(auditResult.smoke_confidence * 100).toFixed(1)}%</strong>
                      </>
                    ) : null}
                  </div>
                </div>

                <p className="disclaimer-note">
                  Note: Fire and smoke affected region percentages are approximate 2D bounding-box
                  visual measurements relative to the frame area, not actual physical damage, heat,
                  or safety risk.
                </p>
              </div>
            ) : null}
          </section>
        ) : null}

        {activePage === "history" ? (
          <section className="page-header">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem" }}>
              <div>
                <h1>Detection History</h1>
                <p className="page-sub">
                  Automatically saved results from completed image and video analyses.
                </p>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={loadHistory}
                disabled={historyLoading}
              >
                {historyLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>

            {historyMsg ? <div className="smoke-note">{historyMsg}</div> : null}

            {historyRows.length === 0 ? (
              <div className="panel-card">
                No detections recorded yet. Go to <strong>Live Audit</strong> to analyze an image
                or video.
              </div>
            ) : (
              <>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Date and time</th>
                        <th>Uploaded file name</th>
                        <th>Type</th>
                        <th>Fire coverage (%)</th>
                        <th>Smoke coverage (%)</th>
                        <th>Visual severity</th>
                        <th>Detection result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {historyRows.map((row) => (
                        <tr key={row.id}>
                          <td>{row.created_at}</td>
                          <td>{row.file_name}</td>
                          <td>{row.file_type}</td>
                          <td>{Number(row.fire_coverage || 0).toFixed(2)}%</td>
                          <td>{Number(row.smoke_coverage || 0).toFixed(2)}%</td>
                          <td>{row.visual_severity}</td>
                          <td>{row.detection_type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "1rem",
                    marginTop: "1rem",
                    flexWrap: "wrap",
                  }}
                >
                  <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.9rem" }}>
                    <input
                      type="checkbox"
                      checked={confirmClear}
                      onChange={(e) => setConfirmClear(e.target.checked)}
                    />
                    Confirm clearing all saved detection records
                  </label>

                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={!confirmClear}
                    onClick={handleClearHistory}
                  >
                    Clear History
                  </button>
                </div>
              </>
            )}
          </section>
        ) : null}

        {activePage === "about" ? (
          <section className="page-header">
            <h1>About Project</h1>
            <p className="page-sub">
              Architecture, AI model specification, and evaluation details.
            </p>

            <div className="card-grid-2">
              <div className="panel-card">
                <h3 style={{ marginTop: 0 }}>Cloud Split Architecture</h3>
                <ul style={{ color: "#cbd5e1", lineHeight: 1.75, paddingLeft: "1.2rem" }}>
                  <li>
                    <strong>Frontend (Vercel):</strong> React + Vite single-page dashboard (
                    <code>frontend/</code>)
                  </li>
                  <li>
                    <strong>Backend (Render):</strong> FastAPI + Uvicorn inference API (
                    <code>backend/main.py</code>)
                  </li>
                  <li>
                    <strong>Object Detector:</strong> Ultralytics YOLO11n trained on D-Fire (
                    <code>0: smoke</code>, <code>1: fire</code>)
                  </li>
                  <li>
                    <strong>Confidence Threshold:</strong>{" "}
                    {aboutData?.confidence_threshold ?? 0.25} |{" "}
                    <strong>Input Size:</strong> {aboutData?.image_size ?? 640}px
                  </li>
                </ul>
              </div>

              <div className="panel-card">
                <h3 style={{ marginTop: 0 }}>Model Evaluation Summary</h3>
                {aboutData?.evaluation_summary?.map50 != null ? (
                  <ul style={{ color: "#cbd5e1", lineHeight: 1.75, paddingLeft: "1.2rem" }}>
                    <li>
                      <strong>Model Tier:</strong>{" "}
                      {aboutData?.model_metadata?.model_tier || "Trained"}
                    </li>
                    <li>
                      <strong>Test Images:</strong>{" "}
                      {aboutData.evaluation_summary.test_images_evaluated}
                    </li>
                    <li>
                      <strong>mAP@50:</strong> {aboutData.evaluation_summary.map50} |{" "}
                      <strong>F1:</strong> {aboutData.evaluation_summary.f1_score}
                    </li>
                    <li>
                      <strong>Precision:</strong> {aboutData.evaluation_summary.precision} |{" "}
                      <strong>Recall:</strong> {aboutData.evaluation_summary.recall}
                    </li>
                  </ul>
                ) : (
                  <p style={{ color: "#94a3b8" }}>
                    Connect to the Render backend API to view live model metadata and evaluation
                    metrics.
                  </p>
                )}
              </div>
            </div>

            <div className="panel-card">
              <h3 style={{ marginTop: 0 }}>Academic Prototype Notice</h3>
              <p style={{ color: "#cbd5e1", lineHeight: 1.6, margin: 0 }}>
                {aboutData?.disclaimer ||
                  "FireGuard AI is an educational research prototype. It is not a certified fire safety device or life-safety alarm system."}
              </p>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
