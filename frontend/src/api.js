const RAW_BASE = (import.meta.env.VITE_API_URL || "").trim();
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, "");

export function apiUrl(path) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${clean}`;
}

export function resolveMediaUrl(url) {
  if (!url) return "";
  if (url.startsWith("data:") || url.startsWith("http://") || url.startsWith("https://") || url.startsWith("blob:")) {
    return url;
  }
  return apiUrl(url);
}

async function parseJsonResponse(resp) {
  const text = await resp.text();
  try {
    return JSON.parse(text);
  } catch {
    return { error_message: text || `HTTP ${resp.status}` };
  }
}

async function safeFetch(url, options) {
  try {
    return await fetch(url, options);
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new Error(
      "Could not reach the Render backend (it may be waking up from sleep or restarting — wait ~15–30 seconds and click Retry)."
    );
  }
}

export async function fetchHealth(refresh = false) {
  const resp = await safeFetch(apiUrl(`/api/health${refresh ? "?refresh=true" : ""}`));
  const data = await parseJsonResponse(resp);
  if (!resp.ok) {
    throw new Error(data.error_message || data.detail || `Health check failed (${resp.status})`);
  }
  return data;
}

export async function analyzeMedia(file, signal) {
  const form = new FormData();
  form.append("file", file);
  const resp = await safeFetch(apiUrl("/api/analyze"), {
    method: "POST",
    body: form,
    signal,
  });
  const data = await parseJsonResponse(resp);
  if (!resp.ok || data.status === "failed") {
    throw new Error(data.error_message || data.detail || `Analysis failed (${resp.status})`);
  }
  return data;
}

export async function fetchHistory() {
  const resp = await safeFetch(apiUrl("/api/history"));
  const data = await parseJsonResponse(resp);
  if (!resp.ok) {
    throw new Error(data.error_message || data.detail || `Failed to load history (${resp.status})`);
  }
  return data;
}

export async function clearHistory() {
  const resp = await safeFetch(apiUrl("/api/history"), { method: "DELETE" });
  const data = await parseJsonResponse(resp);
  if (!resp.ok) {
    throw new Error(data.error_message || data.detail || `Failed to clear history (${resp.status})`);
  }
  return data;
}

export async function fetchAbout() {
  const resp = await safeFetch(apiUrl("/api/about"));
  const data = await parseJsonResponse(resp);
  if (!resp.ok) {
    throw new Error(data.error_message || data.detail || `Failed to load project metadata (${resp.status})`);
  }
  return data;
}
