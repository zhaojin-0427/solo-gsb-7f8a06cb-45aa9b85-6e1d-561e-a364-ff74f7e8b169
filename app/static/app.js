/* 录音片段定位 —— 原生前端逻辑 */
"use strict";

const $ = (id) => document.getElementById(id);

const refInput = $("refInput");
const clipInput = $("clipInput");
const submitBtn = $("submitBtn");
const statusText = $("statusText");
const errorBanner = $("errorBanner");
const resultPanel = $("resultPanel");
const resultSummary = $("resultSummary");
const waveCanvas = $("waveCanvas");
const waveError = $("waveError");
const audioEl = $("audioEl");
const playHitBtn = $("playHitBtn");
const playFullBtn = $("playFullBtn");
const metaBody = $("metaBody");
const exportBlock = $("exportBlock");
const preMsInput = $("preMs");
const postMsInput = $("postMs");
const exportBtn = $("exportBtn");
const exportMsg = $("exportMsg");

let refFile = null;
let clipFile = null;
let refObjectUrl = null;
let lastResult = null;
let waveState = null; // { peaks, duration, candidates, primaryIdx }

/* ---------- 工具 ---------- */

function fmtTime(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, "0")}:${rest.toFixed(3).padStart(6, "0")}`;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

function setBusy(busy) {
  submitBtn.disabled = busy || !refFile || !clipFile;
  statusText.textContent = busy ? "正在上传并分析…" : "";
}

/* ---------- 文件选择与提交 ---------- */

function bindFileInput(input, nameEl, setter) {
  input.addEventListener("change", () => {
    const file = input.files[0] || null;
    setter(file);
    nameEl.textContent = file
      ? `${file.name}（${(file.size / 1024).toFixed(1)} KB）`
      : "未选择文件";
    submitBtn.disabled = !(refFile && clipFile);
  });
}

bindFileInput(refInput, $("refName"), (f) => (refFile = f));
bindFileInput(clipInput, $("clipName"), (f) => (clipFile = f));

submitBtn.addEventListener("click", async () => {
  clearError();
  resultPanel.classList.add("hidden");
  setBusy(true);
  try {
    const form = new FormData();
    form.append("reference", refFile);
    form.append("clip", clipFile);

    const resp = await fetch("/api/locate", { method: "POST", body: form });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `请求失败（HTTP ${resp.status}）`);
    }
    lastResult = data;
    renderResult(data);
    await prepareWaveform();
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setBusy(false);
  }
});

/* ---------- 结果渲染 ---------- */

function renderResult(data) {
  resultPanel.classList.remove("hidden");
  playHitBtn.disabled = true;

  // 仅定位唯一时开放「导出命中片段」；不唯一 / 失败时隐藏，原提示保留
  exportBlock.classList.toggle("hidden", data.status !== "located");
  exportMsg.classList.add("hidden");
  exportMsg.textContent = "";
  exportMsg.classList.remove("error");

  const hit = data.candidates && data.candidates.length ? data.candidates[0] : null;
  let html = "";

  if (data.status === "located") {
    html = `
      <div class="result-status"><span class="tag ok">定位成功</span></div>
      <p class="result-detail">
        片段位于参考录音的
        <span class="num">${fmtTime(hit.start_s)}</span> ～
        <span class="num">${fmtTime(hit.end_s)}</span>
        （起点 ${hit.start_s.toFixed(3)} s，终点 ${hit.end_s.toFixed(3)} s），
        相关得分 <span class="num">${hit.score.toFixed(4)}</span>
      </p>`;
    playHitBtn.disabled = false;
  } else if (data.status === "ambiguous") {
    const items = data.candidates
      .map(
        (c, i) => `
        <li>
          <span>候选 ${i + 1}：<span class="num">${fmtTime(c.start_s)}</span> ～
          <span class="num">${fmtTime(c.end_s)}</span>，
          得分 <span class="num">${c.score.toFixed(4)}</span></span>
          <button data-cand="${i}">从此处试听</button>
        </li>`
      )
      .join("");
    html = `
      <div class="result-status"><span class="tag warn">定位不唯一</span></div>
      <p class="result-detail">${data.message}（最高得分
        <span class="num">${data.best_score.toFixed(4)}</span>）：</p>
      <ul class="candidate-list">${items}</ul>`;
    playHitBtn.disabled = false;
  } else {
    html = `
      <div class="result-status"><span class="tag err">无法定位</span></div>
      <p class="result-detail">${data.message}</p>`;
  }

  resultSummary.innerHTML = html;

  resultSummary.querySelectorAll("button[data-cand]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.cand);
      playFrom(data.candidates[idx].start_s);
      waveState && (waveState.primaryIdx = idx);
      drawWaveform();
    });
  });

  metaBody.innerHTML = `
    <tr>
      <td>参考录音</td>
      <td>${data.reference.duration_s.toFixed(3)} s</td>
      <td>${data.reference.sample_rate} Hz</td>
      <td>${data.reference.channels === 2 ? "双声道" : "单声道"}</td>
    </tr>
    <tr>
      <td>待找片段</td>
      <td>${data.clip.duration_s.toFixed(3)} s</td>
      <td>${data.clip.sample_rate} Hz</td>
      <td>${data.clip.channels === 2 ? "双声道" : "单声道"}</td>
    </tr>
    <tr>
      <td>分析用采样率</td>
      <td colspan="3">${data.sample_rate} Hz（统一重采样后）</td>
    </tr>`;
}

/* ---------- 波形解码与绘制 ---------- */

async function prepareWaveform() {
  waveError.classList.add("hidden");
  waveState = null;

  if (refObjectUrl) URL.revokeObjectURL(refObjectUrl);
  refObjectUrl = URL.createObjectURL(refFile);
  audioEl.src = refObjectUrl;

  try {
    const buf = await refFile.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await ctx.decodeAudioData(buf);
    ctx.close();
    waveState = {
      samples: audioBuf.getChannelData(0),
      duration: audioBuf.duration,
      candidates: (lastResult && lastResult.candidates) || [],
      primaryIdx: 0,
      peaks: null,
      peaksWidth: 0,
    };
    // 主命中默认取得分最高的候选（列表本身按时间排序）
    let best = -Infinity;
    waveState.candidates.forEach((c, i) => {
      if (c.score > best) {
        best = c.score;
        waveState.primaryIdx = i;
      }
    });
    drawWaveform();
  } catch (err) {
    waveError.textContent = "浏览器无法解码参考录音，波形与试听不可用。";
    waveError.classList.remove("hidden");
  }
}

function computePeaks(samples, width) {
  const peaks = new Float32Array(width);
  const step = samples.length / width;
  for (let x = 0; x < width; x++) {
    const start = Math.floor(x * step);
    const end = Math.min(samples.length, Math.max(start + 1, Math.floor((x + 1) * step)));
    let max = 0;
    for (let i = start; i < end; i++) {
      const v = Math.abs(samples[i]);
      if (v > max) max = v;
    }
    peaks[x] = max;
  }
  return peaks;
}

function drawWaveform() {
  if (!waveState) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = waveCanvas.clientWidth || waveCanvas.parentElement.clientWidth;
  const cssHeight = 180;
  waveCanvas.width = Math.round(cssWidth * dpr);
  waveCanvas.height = Math.round(cssHeight * dpr);

  const ctx = waveCanvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssWidth;
  const H = cssHeight;
  const mid = H / 2;

  if (!waveState.peaks || waveState.peaksWidth !== W) {
    waveState.peaks = computePeaks(waveState.samples, Math.max(1, Math.round(W)));
    waveState.peaksWidth = W;
  }
  const { peaks, duration, candidates, primaryIdx } = waveState;
  const toX = (t) => (t / duration) * W;

  ctx.clearRect(0, 0, W, H);

  // 命中 / 候选区间高亮
  candidates.forEach((c, i) => {
    const x0 = toX(c.start_s);
    const x1 = toX(c.end_s);
    const primary = i === primaryIdx;
    ctx.fillStyle = primary
      ? "rgba(37, 99, 235, 0.18)"
      : "rgba(180, 83, 9, 0.15)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), H);
  });

  // 波形
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < peaks.length; x++) {
    const y = peaks[x] * (mid - 4);
    ctx.moveTo(x + 0.5, mid - y);
    ctx.lineTo(x + 0.5, mid + y);
  }
  ctx.stroke();

  // 区间边界线
  candidates.forEach((c, i) => {
    const primary = i === primaryIdx;
    ctx.strokeStyle = primary
      ? "rgba(37, 99, 235, 0.9)"
      : "rgba(180, 83, 9, 0.8)";
    ctx.lineWidth = primary ? 2 : 1.5;
    [toX(c.start_s), toX(c.end_s)].forEach((x) => {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    });
  });

  // 中线
  ctx.strokeStyle = "rgba(0, 0, 0, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(W, mid);
  ctx.stroke();

  // 播放头
  if (!audioEl.paused && audioEl.duration) {
    const px = (audioEl.currentTime / audioEl.duration) * W;
    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
  }
}

window.addEventListener("resize", drawWaveform);
audioEl.addEventListener("timeupdate", drawWaveform);
audioEl.addEventListener("ended", drawWaveform);

waveCanvas.addEventListener("click", (e) => {
  if (!waveState || !audioEl.duration) return;
  const rect = waveCanvas.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / rect.width;
  audioEl.currentTime = ratio * audioEl.duration;
  drawWaveform();
});

/* ---------- 播放 ---------- */

function playFrom(t) {
  if (!audioEl.src) return;
  audioEl.currentTime = t;
  audioEl.play().catch(() => {});
}

playHitBtn.addEventListener("click", () => {
  if (!lastResult || !lastResult.candidates || !lastResult.candidates.length) return;
  const idx = (waveState && waveState.primaryIdx) || 0;
  playFrom(lastResult.candidates[idx].start_s);
});

playFullBtn.addEventListener("click", () => playFrom(0));

/* ---------- 导出命中片段 ---------- */

const CONTEXT_MAX_MS = 2000;
const CONTEXT_STEP_MS = 100;

function validContext(v) {
  return (
    Number.isInteger(v) &&
    v >= 0 &&
    v <= CONTEXT_MAX_MS &&
    v % CONTEXT_STEP_MS === 0
  );
}

function showExportMsg(msg, isError) {
  exportMsg.textContent = msg;
  exportMsg.classList.toggle("error", !!isError);
  exportMsg.classList.remove("hidden");
}

exportBtn.addEventListener("click", async () => {
  if (!lastResult || lastResult.status !== "located" || !refFile || !clipFile) return;

  const pre = Number(preMsInput.value);
  const post = Number(postMsInput.value);
  if (!validContext(pre) || !validContext(post)) {
    // 范围无效：不发请求、不生成文件，定位结果区的原提示保持不变
    showExportMsg(
      `上下文需为 0～${CONTEXT_MAX_MS} 毫秒且为 ${CONTEXT_STEP_MS} 的整数倍。`,
      true
    );
    return;
  }

  exportBtn.disabled = true;
  try {
    const form = new FormData();
    form.append("reference", refFile);
    form.append("clip", clipFile);
    form.append("pre_ms", String(pre));
    form.append("post_ms", String(post));

    const resp = await fetch("/api/export", { method: "POST", body: form });
    if (!resp.ok) {
      // 定位不唯一 / 失败 / 范围无效：服务端不生成文件，仅在导出区给出原因
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `导出失败（HTTP ${resp.status}）`);
    }

    const startMs = Number(resp.headers.get("X-Export-Start-Ms"));
    const endMs = Number(resp.headers.get("X-Export-End-Ms"));
    const blob = await resp.blob();

    const filename = `hit_${startMs}-${endMs}ms.wav`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    // 与期望范围对比，判断是否因超出边界而收紧
    const hit = lastResult.candidates[0];
    const wantStartMs = hit.start_s * 1000 - pre;
    const wantEndMs = hit.end_s * 1000 + post;
    const clamped =
      Math.abs(wantStartMs - startMs) > 1 || Math.abs(wantEndMs - endMs) > 1;
    showExportMsg(
      `已导出 ${filename}：实际范围 ${startMs} ms ～ ${endMs} ms` +
        (clamped ? "（超出参考录音边界，已自动收紧）" : "")
    );
  } catch (err) {
    showExportMsg(err.message || String(err), true);
  } finally {
    exportBtn.disabled = false;
  }
});

/* ============================================================
   片段顺序核验
   ============================================================ */

const SEQ_MIN = 2;
const SEQ_MAX = 5;

const seqClipsEl = $("seqClips");
const seqAddBtn = $("seqAddBtn");
const seqSubmitBtn = $("seqSubmitBtn");
const seqStatusText = $("seqStatusText");
const seqErrorBanner = $("seqErrorBanner");
const seqResultPanel = $("seqResultPanel");
const seqResultSummary = $("seqResultSummary");
const seqWaveCanvas = $("seqWaveCanvas");
const seqWaveError = $("seqWaveError");
const seqAudioEl = $("seqAudioEl");
const seqPlayBtn = $("seqPlayBtn");
const seqMetaBody = $("seqMetaBody");

let seqFiles = [null, null]; // 按预期顺序排列的片段文件
let seqObjectUrl = null;
let seqLastResult = null;
let seqWaveState = null; // { samples, duration, marks }

/* ---------- 片段行（2～5 段，顺序即输入顺序） ---------- */

function seqReady() {
  return !!refFile && seqFiles.length >= SEQ_MIN && seqFiles.every(Boolean);
}

function renderSeqRows() {
  seqClipsEl.innerHTML = "";
  seqFiles.forEach((file, i) => {
    const row = document.createElement("label");
    row.className = "seq-clip-row seq-order";

    const idx = document.createElement("span");
    idx.className = "seq-idx";
    idx.textContent = String(i + 1);

    const body = document.createElement("span");
    body.className = "seq-clip-body";
    const label = document.createElement("span");
    label.className = "seq-clip-label";
    label.textContent = `片段 ${i + 1}（0.2～5 s，按顺序）`;
    const name = document.createElement("span");
    name.className = "seq-clip-name";
    name.textContent = file
      ? `${file.name}（${(file.size / 1024).toFixed(1)} KB）`
      : "未选择文件";
    body.append(label, name);

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/wav,audio/wave,audio/x-wav";
    input.addEventListener("change", () => {
      seqFiles[i] = input.files[0] || null;
      renderSeqRows();
      refreshSeqButton();
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "seq-remove";
    remove.textContent = "移除";
    remove.disabled = seqFiles.length <= SEQ_MIN;
    remove.addEventListener("click", (e) => {
      // 按钮位于 <label> 内：阻止冒泡到 label 触发文件选择
      e.preventDefault();
      e.stopPropagation();
      seqFiles.splice(i, 1);
      renderSeqRows();
      refreshSeqButton();
    });

    row.append(idx, body, input, remove);
    seqClipsEl.appendChild(row);
  });

  seqAddBtn.disabled = seqFiles.length >= SEQ_MAX;
}

function refreshSeqButton() {
  seqSubmitBtn.disabled = !seqReady();
}

seqAddBtn.addEventListener("click", () => {
  if (seqFiles.length >= SEQ_MAX) return;
  seqFiles.push(null);
  renderSeqRows();
  refreshSeqButton();
});

function showSeqError(msg) {
  seqErrorBanner.textContent = msg;
  seqErrorBanner.classList.remove("hidden");
}

function clearSeqError() {
  seqErrorBanner.classList.add("hidden");
  seqErrorBanner.textContent = "";
}

function setSeqBusy(busy) {
  seqSubmitBtn.disabled = busy || !seqReady();
  seqStatusText.textContent = busy ? "正在上传并核验…" : "";
}

// 参考录音变更时同步刷新顺序核验按钮（复用定位页的参考录音）
refInput.addEventListener("change", refreshSeqButton);

renderSeqRows();

/* ---------- 提交 ---------- */

seqSubmitBtn.addEventListener("click", async () => {
  if (!seqReady()) return;
  clearSeqError();
  seqResultPanel.classList.add("hidden");
  setSeqBusy(true);
  try {
    const form = new FormData();
    form.append("reference", refFile);
    seqFiles.forEach((f) => form.append("clips", f));

    const resp = await fetch("/api/sequence", { method: "POST", body: form });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `请求失败（HTTP ${resp.status}）`);
    }
    seqLastResult = data;
    renderSeqResult(data);
    await prepareSeqWaveform();
  } catch (err) {
    showSeqError(err.message || String(err));
  } finally {
    setSeqBusy(false);
  }
});

/* ---------- 结果渲染 ---------- */

function fmtMs(v) {
  return `${Number(v).toFixed(0)} ms`;
}

function chainTableHtml(chain, title, rowClass) {
  const rows = chain.segments
    .map((s) => {
      const gap = s.gap_before_ms == null ? "—" : fmtMs(s.gap_before_ms);
      return `
        <tr class="${rowClass}" data-start="${s.start_ms / 1000}">
          <td>${s.segment}</td>
          <td class="num">${fmtMs(s.start_ms)}</td>
          <td class="num">${fmtMs(s.end_ms)}</td>
          <td class="num">${gap}</td>
          <td class="num">${s.score.toFixed(4)}</td>
        </tr>`;
    })
    .join("");
  return `
    <h3 class="chain-title">${title}（总分 <span class="num">${chain.total_score.toFixed(4)}</span>）</h3>
    <table class="chain-table">
      <thead>
        <tr><th>段</th><th>起点</th><th>终点</th><th>与上段空档</th><th>得分</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function bindChainPlayback(rootEl) {
  rootEl.querySelectorAll("tr[data-start]").forEach((tr) => {
    tr.addEventListener("click", () => {
      playSeqFrom(Number(tr.dataset.start));
    });
  });
}

function renderSeqResult(data) {
  seqResultPanel.classList.remove("hidden");
  seqPlayBtn.disabled = false;
  let html = "";

  if (data.status === "verified") {
    html = `
      <div class="result-status"><span class="tag ok">顺序成立（唯一链）</span></div>
      <p class="result-detail">${data.message}</p>
      ${chainTableHtml(data.chain, "核验通过的完整链", "seq-row-playable")}`;
  } else if (data.status === "ambiguous") {
    html = `
      <div class="result-status"><span class="tag warn">多解提示</span></div>
      <p class="result-detail">${data.message}</p>
      ${chainTableHtml(data.best_chain, "得分最高的链", "seq-row-playable")}
      ${chainTableHtml(data.alternative_chain, "另一条不同链", "seq-row-playable")}`;
  } else {
    // no_chain
    html = `
      <div class="result-status"><span class="tag err">顺序不成立</span></div>
      <p class="result-detail">${data.message}</p>`;
    if (data.reason === "segment_not_found") {
      const items = data.failed_segments
        .map((f) => {
          const reason =
            f.reason === "silence"
              ? "近乎静音"
              : `最高得分 ${f.best_score.toFixed(4)} 低于 0.8`;
          return `<li>片段 ${f.segment}：${reason}</li>`;
        })
        .join("");
      html += `<ul class="candidate-list">${items}</ul>`;
    }
  }

  seqResultSummary.innerHTML = html;
  bindChainPlayback(seqResultSummary);

  seqMetaBody.innerHTML = data.segments_info
    .map(
      (s, i) => `
    <tr>
      <td>片段 ${i + 1}</td>
      <td>${s.duration_s.toFixed(3)} s</td>
      <td>${s.sample_rate} Hz</td>
      <td>${s.channels === 2 ? "双声道" : "单声道"}</td>
    </tr>`
    )
    .join("");
}

/* ---------- 参考波形与链标注 ---------- */

function seqMarksFor(result) {
  // marks: [{ start_s, end_s, variant: 'best'|'alt' }]
  if (!result) return [];
  const fromChain = (chain, variant) =>
    chain.segments.map((s) => ({
      start_s: s.start_s,
      end_s: s.end_s,
      variant,
    }));
  if (result.status === "verified") return fromChain(result.chain, "best");
  if (result.status === "ambiguous") {
    return [
      ...fromChain(result.alternative_chain, "alt"),
      ...fromChain(result.best_chain, "best"),
    ];
  }
  return [];
}

async function prepareSeqWaveform() {
  seqWaveError.classList.add("hidden");
  seqWaveState = null;

  if (seqObjectUrl) URL.revokeObjectURL(seqObjectUrl);
  seqObjectUrl = URL.createObjectURL(refFile);
  seqAudioEl.src = seqObjectUrl;

  try {
    const buf = await refFile.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await ctx.decodeAudioData(buf);
    ctx.close();
    seqWaveState = {
      samples: audioBuf.getChannelData(0),
      duration: audioBuf.duration,
      marks: seqMarksFor(seqLastResult),
      peaks: null,
      peaksWidth: 0,
    };
    drawSeqWaveform();
  } catch (err) {
    seqWaveError.textContent = "浏览器无法解码参考录音，波形与试听不可用。";
    seqWaveError.classList.remove("hidden");
  }
}

function drawSeqWaveform() {
  if (!seqWaveState) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = seqWaveCanvas.clientWidth || seqWaveCanvas.parentElement.clientWidth;
  const cssHeight = 180;
  seqWaveCanvas.width = Math.round(cssWidth * dpr);
  seqWaveCanvas.height = Math.round(cssHeight * dpr);

  const ctx = seqWaveCanvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssWidth;
  const H = cssHeight;
  const mid = H / 2;

  if (!seqWaveState.peaks || seqWaveState.peaksWidth !== W) {
    seqWaveState.peaks = computePeaks(seqWaveState.samples, Math.max(1, Math.round(W)));
    seqWaveState.peaksWidth = W;
  }
  const { peaks, duration, marks } = seqWaveState;
  const toX = (t) => (t / duration) * W;

  ctx.clearRect(0, 0, W, H);

  // 链区间高亮：唯一链蓝色；多解时最高链蓝色、另一条琥珀色（先画 alt 再画 best）
  marks.forEach((m) => {
    const x0 = toX(m.start_s);
    const x1 = toX(m.end_s);
    ctx.fillStyle =
      m.variant === "best" ? "rgba(37, 99, 235, 0.18)" : "rgba(180, 83, 9, 0.13)";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), H);
  });

  // 波形
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < peaks.length; x++) {
    const y = peaks[x] * (mid - 4);
    ctx.moveTo(x + 0.5, mid - y);
    ctx.lineTo(x + 0.5, mid + y);
  }
  ctx.stroke();

  // 区间边界（alt 用虚线）
  marks.forEach((m) => {
    ctx.strokeStyle =
      m.variant === "best" ? "rgba(37, 99, 235, 0.9)" : "rgba(180, 83, 9, 0.85)";
    ctx.lineWidth = m.variant === "best" ? 2 : 1.5;
    ctx.setLineDash(m.variant === "best" ? [] : [5, 3]);
    [toX(m.start_s), toX(m.end_s)].forEach((x) => {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    });
    ctx.setLineDash([]);
  });

  // 段序号
  ctx.font = "600 11px sans-serif";
  ctx.textBaseline = "top";
  marks.forEach((m, i) => {
    if (m.variant !== "best") return;
    const x = (toX(m.start_s) + toX(m.end_s)) / 2;
    ctx.fillStyle = "rgba(37, 99, 235, 0.95)";
    ctx.textAlign = "center";
    ctx.fillText(String(i + 1), x, 4);
  });
  ctx.textAlign = "start";

  // 中线
  ctx.strokeStyle = "rgba(0, 0, 0, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(W, mid);
  ctx.stroke();

  // 播放头
  if (!seqAudioEl.paused && seqAudioEl.duration) {
    const px = (seqAudioEl.currentTime / seqAudioEl.duration) * W;
    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
  }
}

window.addEventListener("resize", drawSeqWaveform);
seqAudioEl.addEventListener("timeupdate", drawSeqWaveform);
seqAudioEl.addEventListener("ended", drawSeqWaveform);

seqWaveCanvas.addEventListener("click", (e) => {
  if (!seqWaveState || !seqAudioEl.duration) return;
  const rect = seqWaveCanvas.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / rect.width;
  seqAudioEl.currentTime = ratio * seqAudioEl.duration;
  drawSeqWaveform();
});

function playSeqFrom(t) {
  if (!seqAudioEl.src) return;
  seqAudioEl.currentTime = t;
  seqAudioEl.play().catch(() => {});
}

seqPlayBtn.addEventListener("click", () => playSeqFrom(0));
