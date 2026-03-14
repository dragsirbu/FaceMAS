// src/lib/workerClient.ts
export type B64Image = { filename: string; mime: string; base64: string };

export type QAItem = {
  filename: string;
  face_found: boolean;
  multiple_faces: boolean;
  bbox: number[] | null;
  landmarks_5: number[] | null;
  aligned_256: B64Image | null;
  aligned_112: B64Image | null;
  ofiq_uqs: number | null;
};

export type QualityAssessResponse = { ok: true; results: QAItem[] };

export type PipelineResponse = { ok: true; results: QAItem[]; aged: B64Image[] };

const BASE = process.env.NEXT_PUBLIC_WORKER_URL;

function assertBase() {
  if (!BASE) throw new Error("NEXT_PUBLIC_WORKER_URL is not set");
}

// Build FormData with reference images first, test image last
function toFormData(refFiles: File[], testFile?: File) {
  const fd = new FormData();
  refFiles.forEach((f) => fd.append("images", f));
  if (testFile) fd.append("images", testFile);
  return fd;
}

export async function startPipeline(refFiles: File[], testFile: File, targetAge: number = 50, gender: string = "", attributes: string = ""): Promise<{ job_id: string }> {
  assertBase();
  const fd = toFormData(refFiles, testFile);
  fd.append("target_age", String(targetAge));
  if (gender) fd.append("gender", gender);
  if (attributes) fd.append("attributes", attributes);

  const res = await fetch(`${BASE}/v1/pipeline`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function reEdit(jobId: string, targetAge: number, gender: string = "", attributes: string = ""): Promise<{ job_id: string }> {
  assertBase();
  const fd = new FormData();
  fd.append("target_age", String(targetAge));
  if (gender) fd.append("gender", gender);
  if (attributes) fd.append("attributes", attributes);

  const res = await fetch(`${BASE}/v1/jobs/${jobId}/re-edit`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJob(jobId: string): Promise<any> {
  assertBase();
  const res = await fetch(`${BASE}/v1/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function qualityAssess(refFiles: File[], testFile?: File): Promise<QualityAssessResponse> {
  assertBase();
  const res = await fetch(`${BASE}/v1/quality-assess`, {
    method: "POST",
    body: toFormData(refFiles, testFile),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runPipeline(refFiles: File[], testFile: File): Promise<PipelineResponse> {
  assertBase();
  const res = await fetch(`${BASE}/v1/pipeline`, {
    method: "POST",
    body: toFormData(refFiles, testFile),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// helper for rendering in <img/>
export function b64ToDataUrl(img: B64Image): string {
  return `data:${img.mime};base64,${img.base64}`;
}