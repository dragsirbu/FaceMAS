"use client";

import { useMemo, useRef, useState } from "react";
import { qualityAssess, startPipeline, reEdit, getJob, b64ToDataUrl } from "../lib/client";

type UploadItem = { file: File; previewUrl: string };

type QAItem = {
  filename: string;
  face_found: boolean;
  multiple_faces: boolean;
  bbox: number[] | null;
  landmarks_5: number[] | null;
  aligned_256: { mime: string; base64: string; filename?: string } | null;
  aligned_112: { mime: string; base64: string; filename?: string } | null;
  ofiq_uqs: number | null;
};

type AgedItem = { filename: string; mime: string; base64: string };

type MatchItem = {
  filename: string;
  name: string;
  age: number | null;
  similarity: number;
  mime: string;
  base64: string;
};

type EditRun = {
  targetAge: number;
  gender: string;
  attributes: string;
  aged: AgedItem[];
  testMatches: MatchItem[];
  agedMatches: Record<string, MatchItem[]>;
  searchErrors: string[];
};

export default function HomePage() {
  const refInputRef = useRef<HTMLInputElement | null>(null);
  const testInputRef = useRef<HTMLInputElement | null>(null);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [testPhoto, setTestPhoto] = useState<UploadItem | null>(null);

  const [qaResults, setQaResults] = useState<QAItem[] | null>(null);
  const [editHistory, setEditHistory] = useState<EditRun[]>([]);
  const [selectedEditIdx, setSelectedEditIdx] = useState<number>(0);
  const [isAssessing, setIsAssessing] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobMsg, setJobMsg] = useState<string | null>(null);
  const [targetAge, setTargetAge] = useState<number>(50);
  const [gender, setGender] = useState<string>("");
  const [attributes, setAttributes] = useState<string>("");

  const selectedEdit: EditRun | null = editHistory[selectedEditIdx] ?? null;


  const maxFiles = 5;
  const minFiles = 3;

  const canRun = items.length >= minFiles && testPhoto !== null;
  const remaining = Math.max(0, maxFiles - items.length);

  function addFiles(files: FileList | null) {
    if (!files) return;

    const next: UploadItem[] = [];
    for (const f of Array.from(files)) {
      if (next.length + items.length >= maxFiles) break;
      if (!f.type.startsWith("image/")) continue;
      next.push({ file: f, previewUrl: URL.createObjectURL(f) });
    }
    if (next.length) {
      setItems((prev) => [...prev, ...next]);
      setQaResults(null);
      setEditHistory([]);
      setSelectedEditIdx(0);
      setError(null);
    }
  }

  function addTestPhoto(files: FileList | null) {
    if (!files || files.length === 0) return;
    const f = files[0];
    if (!f.type.startsWith("image/")) return;
    if (testPhoto) URL.revokeObjectURL(testPhoto.previewUrl);
    setTestPhoto({ file: f, previewUrl: URL.createObjectURL(f) });
    setQaResults(null);
    setEditHistory([]);
    setSelectedEditIdx(0);
    setError(null);
  }

  function onDropRef(e: React.DragEvent) {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  }

  function onDropTest(e: React.DragEvent) {
    e.preventDefault();
    addTestPhoto(e.dataTransfer.files);
  }

  function onBrowseRef() {
    refInputRef.current?.click();
  }

  function onBrowseTest() {
    testInputRef.current?.click();
  }

  function clearAll() {
    items.forEach((x) => URL.revokeObjectURL(x.previewUrl));
    if (testPhoto) URL.revokeObjectURL(testPhoto.previewUrl);
    setItems([]);
    setTestPhoto(null);
    setQaResults(null);
    setEditHistory([]);
    setSelectedEditIdx(0);
    setError(null);
  }

  async function onQualityAssess() {
    if (items.length === 0) return;
    setError(null);
    setIsAssessing(true);
    try {
      const refFiles = items.map((x) => x.file);
      const resp = await qualityAssess(refFiles, testPhoto?.file);
      setQaResults(resp.results as QAItem[]);
    } catch (e: any) {
      setError(e?.message ?? "Quality assess failed");
    } finally {
      setIsAssessing(false);
    }
  }

  async function onRunPipeline() {
    if (!canRun) return;
    setError(null);
    setIsRunning(true);
    setEditHistory([]);
    setSelectedEditIdx(0);
    setQaResults(null);

    try {
      const refFiles = items.map((x) => x.file);

      // start -> job_id (test photo is always appended last)
      const editAge = targetAge;
      const editGender = gender;
      const editAttrs = attributes;
      const { job_id } = await startPipeline(refFiles, testPhoto!.file, editAge, editGender, editAttrs);
      setJobId(job_id);

      // poll
      await pollJob(job_id, editAge, editGender, editAttrs);
    } catch (e: any) {
      setError(e?.message ?? "Pipeline failed");
    } finally {
      setIsRunning(false);
    }
  }

  async function onReEdit() {
    if (!jobId) return;
    setError(null);
    setIsRunning(true);

    try {
      const editAge = targetAge;
      const editGender = gender;
      const editAttrs = attributes;
      await reEdit(jobId, editAge, editGender, editAttrs);
      await pollJob(jobId, editAge, editGender, editAttrs);
    } catch (e: any) {
      setError(e?.message ?? "Re-edit failed");
    } finally {
      setIsRunning(false);
    }
  }

  async function pollJob(job_id: string, editAge: number, editGender: string, editAttrs: string) {
    let failures = 0;
    const MAX_FAILURES = 60; // give up after ~90s of consecutive failures
    while (true) {
      let j: any;
      try {
        j = await getJob(job_id);
        failures = 0; // reset on success
      } catch {
        failures++;
        if (failures >= MAX_FAILURES) {
          throw new Error("Lost connection to server");
        }
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }
      const status = j.status ?? j?.job?.status;
      setJobStatus(status ?? "unknown");
      setJobMsg(j.message ?? "");

      if (j.results) setQaResults(j.results);

      if (status === "done") {
        const run: EditRun = {
          targetAge: editAge,
          gender: editGender,
          attributes: editAttrs,
          aged: j.aged ?? [],
          testMatches: j.test_matches ?? [],
          agedMatches: j.aged_matches ?? {},
          searchErrors: j.search_errors ?? [],
        };
        setEditHistory((prev) => {
          const next = [...prev, run];
          setSelectedEditIdx(next.length - 1);
          return next;
        });
        break;
      }

      if (status === "error") {
        throw new Error(j.message || "Job failed");
      }

      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  const qualityCounts = useMemo(() => {
    const rs = qaResults ?? [];
    let ok = 0,
      low = 0,
      reject = 0;

    for (const r of rs) {
      // reject if no face or no score
      if (!r.face_found || r.ofiq_uqs == null) {
        reject++;
        continue;
      }
      // simple thresholds - tweak later
      if (r.ofiq_uqs >= 60) ok++;
      else if (r.ofiq_uqs >= 40) low++;
      else reject++;
    }
    return { ok, low, reject };
  }, [qaResults]);

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-semibold text-slate-900">Upload input images</h1>
            <p className="mt-2 text-slate-600">
              Add 3-5 reference photos and a test photo. We&apos;ll crop, align, and generate aged versions.
            </p>
          </div>
        </div>

        <div className="mt-6 h-px w-full bg-slate-200" />

        {/* Grid */}
        <div className="mt-8 grid gap-6 lg:grid-cols-[420px_1fr]">
          {/* Left column */}
          <section className="space-y-6">
            {/* Input Images card */}
            <Card>
              <CardHeader title="Input Images" />
              <div className="p-5">
                <div className="grid grid-cols-[1fr_auto_1fr] gap-0">
                  {/* Left - Reference Photos */}
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={onDropRef}
                    className="rounded-l-xl border-2 border-dashed border-slate-300 bg-white px-4 py-5 text-center"
                  >
                    <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Reference Photos ({items.length}/{maxFiles})
                    </div>

                    {items.length > 0 ? (
                      <div className="grid grid-cols-2 gap-2">
                        {items.map((item, idx) => (
                          <div key={idx} className="group relative aspect-square overflow-hidden rounded-lg bg-slate-100">
                            <img
                              src={item.previewUrl}
                              alt={item.file.name}
                              className="h-full w-full object-cover"
                            />
                            <button
                              type="button"
                              onClick={() => {
                                URL.revokeObjectURL(item.previewUrl);
                                setItems((prev) => prev.filter((_, i) => i !== idx));
                                setQaResults(null);
                                setEditHistory([]);
                                setSelectedEditIdx(0);
                              }}
                              className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-black/50 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black/70"
                            >
                              ×
                            </button>
                            <div className="absolute bottom-0 left-0 right-0 truncate bg-black/40 px-1 py-0.5 text-[10px] text-white">
                              {item.file.name}
                            </div>
                          </div>
                        ))}
                        {remaining > 0 && (
                          <button
                            type="button"
                            onClick={onBrowseRef}
                            className="flex aspect-square flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-200 text-slate-400 hover:border-blue-400 hover:text-blue-500 transition-colors"
                          >
                            <span className="text-2xl leading-none">+</span>
                            <span className="mt-1 text-[10px]">Add more</span>
                          </button>
                        )}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-4 text-slate-700">
                        <div className="text-sm font-medium">Drop reference photos</div>
                        <div className="mt-2 text-sm text-slate-600">
                          or{" "}
                          <button
                            type="button"
                            onClick={onBrowseRef}
                            className="font-medium text-blue-600 hover:text-blue-700"
                          >
                            Browse
                          </button>
                        </div>
                        <div className="mt-2 text-xs text-slate-500">3-5 photos</div>
                      </div>
                    )}

                    <input
                      ref={refInputRef}
                      type="file"
                      accept="image/png,image/jpeg"
                      multiple
                      className="hidden"
                      onChange={(e) => addFiles(e.target.files)}
                    />
                  </div>

                  {/* Divider */}
                  <div className="flex items-center">
                    <div className="h-3/4 w-px bg-slate-200" />
                  </div>

                  {/* Right - Test Photo */}
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={onDropTest}
                    className="rounded-r-xl border-2 border-l-0 border-dashed border-slate-300 bg-white px-4 py-5 text-center"
                  >
                    <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Test Photo
                    </div>

                    {testPhoto ? (
                      <div className="mx-auto w-28">
                        <div className="group relative aspect-square overflow-hidden rounded-lg bg-slate-100">
                          <img
                            src={testPhoto.previewUrl}
                            alt={testPhoto.file.name}
                            className="h-full w-full object-cover"
                          />
                          <button
                            type="button"
                            onClick={() => {
                              URL.revokeObjectURL(testPhoto.previewUrl);
                              setTestPhoto(null);
                              setQaResults(null);
                              setEditHistory([]);
                              setSelectedEditIdx(0);
                            }}
                            className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-black/50 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black/70"
                          >
                            ×
                          </button>
                          <div className="absolute bottom-0 left-0 right-0 truncate bg-black/40 px-1 py-0.5 text-[10px] text-white">
                            {testPhoto.file.name}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-4 text-slate-700">
                        <div className="text-sm font-medium">Drop test photo</div>
                        <div className="mt-2 text-sm text-slate-600">
                          or{" "}
                          <button
                            type="button"
                            onClick={onBrowseTest}
                            className="font-medium text-blue-600 hover:text-blue-700"
                          >
                            Browse
                          </button>
                        </div>
                        <div className="mt-2 text-xs text-slate-500">1 photo</div>
                      </div>
                    )}

                    <input
                      ref={testInputRef}
                      type="file"
                      accept="image/png,image/jpeg"
                      className="hidden"
                      onChange={(e) => addTestPhoto(e.target.files)}
                    />
                  </div>
                </div>

                <div className="mt-4 text-center">
                  <div className="text-sm text-slate-600">
                    {items.length}/{maxFiles} reference &middot; {testPhoto ? 1 : 0}/1 test
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    Min {minFiles} reference + 1 test to process
                  </div>

                  <div className="mt-3 flex items-center justify-center gap-2">
                    <label htmlFor="target-age" className="text-sm text-slate-600">Target Age:</label>
                    <input
                      id="target-age"
                      type="number"
                      min={1}
                      max={100}
                      value={targetAge}
                      onChange={(e) => setTargetAge(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
                      className="w-16 rounded-md border border-slate-300 px-2 py-1 text-center text-sm text-slate-800"
                    />
                    <label htmlFor="gender" className="ml-3 text-sm text-slate-600">Gender:</label>
                    <select
                      id="gender"
                      value={gender}
                      onChange={(e) => setGender(e.target.value)}
                      className="rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-800"
                    >
                      <option value="">Auto</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                    </select>
                  </div>

                  <div className="mt-2 flex items-center justify-center gap-2">
                    <label htmlFor="attributes" className="text-sm text-slate-600">Attributes:</label>
                    <input
                      id="attributes"
                      type="text"
                      value={attributes}
                      onChange={(e) => setAttributes(e.target.value)}
                      placeholder="e.g. smiling, wearing glasses"
                      className="w-56 rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-800"
                    />
                  </div>

                  {(items.length > 0 || testPhoto) && (
                    <div className="mt-3 flex items-center justify-center gap-3">
                      <button
                        type="button"
                        onClick={clearAll}
                        className="text-sm font-medium text-slate-600 hover:text-slate-800"
                      >
                        Clear All
                      </button>

                      <button
                        type="button"
                        onClick={onQualityAssess}
                        disabled={isAssessing || isRunning}
                        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50 disabled:opacity-50"
                      >
                        {isAssessing ? "Assessing..." : "Quality Assess"}
                      </button>
                    </div>
                  )}

                  {error && <div className="mt-3 text-sm text-red-600">{error}</div>}
                </div>
              </div>
            </Card>


          </section>

          {/* Right column */}
          <section>
            <Card className="min-h-[520px]">
              <div className="flex items-center justify-between px-5 py-4">
                <div className="text-sm font-semibold text-slate-900">
                  Quality Check ({qualityCounts.ok} OK, {qualityCounts.low} Low, {qualityCounts.reject} Reject)
                </div>
              </div>
              <div className="border-t border-slate-200" />

              {/* Content */}
              {qaResults && qaResults.length > 0 ? (
                <div className="p-5">
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                    {qaResults.map((r, idx) => {
                      const thumb =
                        r.aligned_256 ? `data:${r.aligned_256.mime};base64,${r.aligned_256.base64}` : null;

                      const badge =
                        !r.face_found || r.ofiq_uqs == null
                          ? { label: "Reject", cls: "bg-red-100 text-red-700" }
                          : r.ofiq_uqs >= 60
                          ? { label: "OK", cls: "bg-green-100 text-green-700" }
                          : r.ofiq_uqs >= 40
                          ? { label: "Low", cls: "bg-amber-100 text-amber-800" }
                          : { label: "Reject", cls: "bg-red-100 text-red-700" };

                      return (
                        <div key={`${r.filename}-${idx}`} className="rounded-xl border border-slate-200 bg-white">
                          <div className="aspect-square overflow-hidden rounded-t-xl bg-slate-100">
                            {thumb ? (
                              <img src={thumb} alt={r.filename} className="h-full w-full object-cover" />
                            ) : (
                              <div className="flex h-full w-full items-center justify-center text-sm text-slate-500">
                                No face
                              </div>
                            )}
                          </div>
                          <div className="flex items-center justify-between px-3 py-2">
                            <div className="truncate text-xs text-slate-600">{r.filename}</div>
                            <span className={`rounded-full px-2 py-1 text-xs font-semibold ${badge.cls}`}>
                              {badge.label}
                            </span>
                          </div>
                          <div className="px-3 pb-3 text-xs text-slate-500">
                            OFIQ: {r.ofiq_uqs == null ? "-" : r.ofiq_uqs.toFixed(1)}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Edit history tabs + Re-Edit button */}
                  {(editHistory.length > 0 || isRunning) && (
                    <>
                      <div className="mt-6 flex items-center justify-between">
                        <div className="text-sm font-semibold text-slate-900">Aged Outputs</div>
                        <button
                          type="button"
                          onClick={onReEdit}
                          disabled={isRunning || !jobId}
                          className="rounded-lg border border-blue-300 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:opacity-50"
                        >
                          {isRunning ? "Re-editing..." : "Re-Edit"}
                        </button>
                      </div>

                      {/* Edit run tabs */}
                      {editHistory.length > 1 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {editHistory.map((run, idx) => {
                            const label = `Age ${run.targetAge}${run.gender ? ` / ${run.gender}` : ""}${run.attributes ? ` / ${run.attributes}` : ""}`;
                            const isSelected = idx === selectedEditIdx;
                            return (
                              <button
                                key={`edit-tab-${idx}`}
                                type="button"
                                onClick={() => setSelectedEditIdx(idx)}
                                className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                                  isSelected
                                    ? "bg-blue-600 text-white"
                                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                                }`}
                              >
                                #{idx + 1}: {label}
                              </button>
                            );
                          })}
                        </div>
                      )}

                      {/* Selected edit params summary */}
                      {selectedEdit && (
                        <div className="mt-2 text-xs text-slate-500">
                          Age {selectedEdit.targetAge}
                          {selectedEdit.gender ? ` \u00b7 ${selectedEdit.gender}` : ""}
                          {selectedEdit.attributes ? ` \u00b7 ${selectedEdit.attributes}` : ""}
                        </div>
                      )}

                      {/* Aged images for selected run */}
                      {selectedEdit && selectedEdit.aged.length > 0 && (
                        <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
                          {selectedEdit.aged.map((img, idx) => (
                            <div key={`${img.filename}-${idx}`} className="overflow-hidden rounded-xl border border-slate-200">
                              <img
                                src={b64ToDataUrl(img as any)}
                                alt={img.filename}
                                className="h-full w-full object-cover"
                              />
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Search errors for selected run */}
                      {selectedEdit && selectedEdit.searchErrors.length > 0 && (
                        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
                          <div className="text-sm font-semibold text-red-800">AgeDB search failed</div>
                          {selectedEdit.searchErrors.map((err, idx) => (
                            <div key={`search-err-${idx}`} className="mt-1 text-xs text-red-600">{err}</div>
                          ))}
                        </div>
                      )}

                      {/* AgeDB Matches - Original for selected run */}
                      {selectedEdit && selectedEdit.testMatches.length > 0 && (
                        <>
                          <div className="mt-6 text-sm font-semibold text-slate-900">AgeDB Matches - Original</div>
                          <div className="mt-3 grid grid-cols-5 gap-3">
                            {selectedEdit.testMatches.map((m, idx) => (
                              <MatchCard key={`test-match-${idx}`} match={m} />
                            ))}
                          </div>
                        </>
                      )}

                      {/* AgeDB Matches - Aged for selected run */}
                      {selectedEdit && Object.keys(selectedEdit.agedMatches).length > 0 && (
                        <>
                          {Object.entries(selectedEdit.agedMatches).map(([fname, matches]) => (
                            matches && matches.length > 0 && (
                              <div key={`aged-match-group-${fname}`}>
                                <div className="mt-6 text-sm font-semibold text-slate-900">AgeDB Matches - {fname}</div>
                                <div className="mt-3 grid grid-cols-5 gap-3">
                                  {matches.map((m, idx) => (
                                    <MatchCard key={`aged-match-${fname}-${idx}`} match={m} />
                                  ))}
                                </div>
                              </div>
                            )
                          ))}
                        </>
                      )}
                    </>
                  )}
                </div>
              ) : (
                <div className="flex h-[380px] flex-col items-center justify-center px-6 text-center">
                  <div className="h-28 w-44 rounded-2xl bg-slate-100" />
                  <p className="mt-6 text-base font-medium text-slate-700">
                    No photos yet - add 3 to 5 to continue.
                  </p>
                </div>
              )}
            </Card>
          </section>
        </div>

        {/* Bottom bar */}
        <div className="mt-8 flex items-center justify-between">
          <div className="text-sm text-slate-600">
            {canRun ? "Ready to run." : `Add at least ${minFiles} reference photos and 1 test photo to enable processing`}
          </div>

          {jobId && (
            <div className="text-xs text-slate-600">
              Job: {jobId.slice(0, 8)} - {jobStatus} {jobMsg ? `(${jobMsg})` : ""}
            </div>
          )}


          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onBrowseRef}
              disabled={remaining === 0 || isAssessing || isRunning}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50 disabled:opacity-50"
            >
              Add More
            </button>

            <button
              type="button"
              onClick={onRunPipeline}
              disabled={!canRun || isRunning || isAssessing}
              className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {isRunning ? "Running..." : "Run Pipeline"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

/* Small UI helpers */
function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>{children}</div>;
}

function CardHeader({ title }: { title: string }) {
  return (
    <>
      <div className="flex items-center justify-between px-5 py-4">
        <div className="text-sm font-semibold text-slate-900">{title}</div>
      </div>
      <div className="border-t border-slate-200" />
    </>
  );
}

function MatchCard({ match }: { match: { name: string; age: number | null; similarity: number; mime: string; base64: string } }) {
  const simPct = (match.similarity * 100).toFixed(1);
  const simCls =
    match.similarity >= 0.6
      ? "text-green-700"
      : match.similarity >= 0.4
      ? "text-amber-700"
      : "text-red-600";

  return (
    <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
      <div className="aspect-square bg-slate-100">
        {match.base64 ? (
          <img
            src={`data:${match.mime};base64,${match.base64}`}
            alt={match.name}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-xs text-slate-400">N/A</div>
        )}
      </div>
      <div className="px-2 py-1.5">
        <div className="truncate text-xs font-medium text-slate-800">{match.name}</div>
        <div className="flex items-center justify-between text-[10px] text-slate-500">
          <span>{match.age != null ? `Age ${match.age}` : ""}</span>
          <span className={`font-semibold ${simCls}`}>{simPct}%</span>
        </div>
      </div>
    </div>
  );
}