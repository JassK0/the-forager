// Small canvas charts for the data drawer: lines with optional bands, stacked
// allele frequencies and a phase plane. Colours come from CSS tokens.
const TAU = Math.PI * 2;
export const C = {};

export function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["surface", "ink", "ink-2", "muted", "grid", "axis", "series-1", "series-2", "series-3", "series-4"]) {
    C[k] = cs.getPropertyValue("--" + k).trim();
  }
}

export function alpha(color, a) {
  const n = parseInt(color.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

export function compact(v) {
  if (v == null || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/, "") + "K";
  if (a >= 10 || v === 0) return String(Math.round(v));
  return v.toFixed(a >= 1 ? 2 : 3).replace(/0+$/, "").replace(/\.$/, "");
}

function niceStep(range, n) {
  const raw = range / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

export function niceTicks(lo, hi, n) {
  if (!(hi > lo)) hi = lo + 1;
  const step = niceStep(hi - lo, n);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function niceMax(max, n) {
  const step = niceStep(Math.max(max, 1e-9), n);
  return Math.ceil(Math.max(max, 1e-9) / step) * step;
}

function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px Nunito, system-ui, sans-serif";
  return { ctx, w, h };
}

function empty(ctx, text = "Collecting data…") {
  ctx.fillStyle = C.muted;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(text, 8, 8);
}

function axes(ctx, { pad, w, h, X, Y, x0, x1, yTicks, yFormat = compact }) {
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of yTicks) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.fillText(yFormat(v), pad.l - 6, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const pw = w - pad.l - pad.r;
  for (const t of niceTicks(x0, x1, Math.max(2, Math.floor(pw / 90)))) {
    if (t < x0 || t > x1) continue;
    ctx.fillText(compact(t), X(t), h - pad.b + 5);
  }
  ctx.strokeStyle = C.axis;
  ctx.beginPath();
  ctx.moveTo(pad.l, Math.round(h - pad.b) + 0.5);
  ctx.lineTo(w - pad.r, Math.round(h - pad.b) + 0.5);
  ctx.stroke();
}

function marker(ctx, x, y, color) {
  ctx.fillStyle = C.surface;
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, TAU);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, TAU);
  ctx.fill();
}

// Average runs of samples so there is at most about one point per pixel.
function bins(n, pw) {
  const step = Math.max(1, Math.floor(n / Math.max(1, pw)));
  const out = [];
  for (let i = 0; i < n; i += step) out.push([i, Math.min(n, i + step)]);
  return out;
}

function avg(arr, a, b) {
  let s = 0, k = 0;
  for (let j = a; j < b; j++) if (arr[j] != null) { s += arr[j]; k++; }
  return k ? s / k : null;
}

// series: [{ys, color, lo?, hi?}]; xs = ticks. Returns hover geometry.
export function lineChart(canvas, { xs, series, zero = true, hover = null, yFormat, emptyText }) {
  const { ctx, w, h } = prep(canvas);
  canvas._geo = null;
  const n = xs.length;
  const pad = { l: 42, r: 12, t: 10, b: 20 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  if (n < 2 || pw < 20) return empty(ctx);
  let lo = Infinity, hi = -Infinity;
  for (const s of series) {
    for (let i = 0; i < n; i++) {
      const a = s.lo ? s.lo[i] : s.ys[i], b = s.hi ? s.hi[i] : s.ys[i];
      if (a != null && a < lo) lo = a;
      if (b != null && b > hi) hi = b;
    }
  }
  if (!Number.isFinite(lo)) return empty(ctx, emptyText || "No animals of this species right now");
  if (zero) {
    lo = 0;
    hi = niceMax(hi, 4);
  } else {
    const p = (hi - lo) * 0.25 || Math.abs(hi) * 0.1 || 1;
    lo -= p;
    hi += p;
  }
  const x0 = xs[0], x1 = xs[n - 1];
  const X = (t) => pad.l + ((t - x0) / (x1 - x0 || 1)) * pw;
  const Yf = (v) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * ph;
  axes(ctx, { pad, w, h, X, Y: Yf, x0, x1, yTicks: niceTicks(lo, hi, 4).filter((v) => v >= lo && v <= hi), yFormat });
  const groups = bins(n, pw);
  for (const s of series) {
    const pts = groups.map(([a, b]) => {
      const y = avg(s.ys, a, b);
      return y == null ? null : [X(avg(xs, a, b)), Yf(y), s.lo ? Yf(avg(s.lo, a, b)) : 0, s.hi ? Yf(avg(s.hi, a, b)) : 0];
    });
    if (s.lo) {
      ctx.fillStyle = alpha(s.color, 0.13);
      let run = [];
      const flush = () => {
        if (run.length > 1) {
          ctx.beginPath();
          run.forEach((p, i) => (i ? ctx.lineTo(p[0], p[3]) : ctx.moveTo(p[0], p[3])));
          for (let i = run.length - 1; i >= 0; i--) ctx.lineTo(run[i][0], run[i][2]);
          ctx.closePath();
          ctx.fill();
        }
        run = [];
      };
      for (const p of pts) (p ? run.push(p) : flush());
      flush();
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    let pen = false;
    for (const p of pts) {
      if (!p) { pen = false; continue; }
      pen ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      pen = true;
    }
    ctx.stroke();
  }
  if (hover != null && hover < n) {
    const x = Math.round(X(xs[hover])) + 0.5;
    ctx.strokeStyle = C.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
    for (const s of series) if (s.ys[hover] != null) marker(ctx, X(xs[hover]), Yf(s.ys[hover]), s.color);
  }
  canvas._geo = { pad, pw, x0, x1 };
}

// Stacked shares that sum to 1 at each sample (allele frequencies).
export function stackedChart(canvas, { xs, layers, hover = null }) {
  const { ctx, w, h } = prep(canvas);
  canvas._geo = null;
  const n = xs.length;
  const pad = { l: 42, r: 12, t: 10, b: 20 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  if (n < 2 || pw < 20) return empty(ctx);
  const x0 = xs[0], x1 = xs[n - 1];
  const X = (t) => pad.l + ((t - x0) / (x1 - x0 || 1)) * pw;
  const Yf = (v) => pad.t + (1 - v) * ph;
  const groups = bins(n, pw);
  const cum = groups.map(() => 0);
  layers.forEach((layer) => {
    const top = groups.map(([a, b], k) => {
      const v = avg(layer.ys, a, b);
      return v == null ? null : cum[k] + v;
    });
    ctx.fillStyle = layer.color;
    ctx.beginPath();
    let started = false;
    groups.forEach(([a, b], k) => {
      if (top[k] == null) return;
      const x = X(avg(xs, a, b));
      started ? ctx.lineTo(x, Yf(top[k])) : ctx.moveTo(x, Yf(top[k]));
      started = true;
    });
    for (let k = groups.length - 1; k >= 0; k--) {
      if (top[k] == null) continue;
      ctx.lineTo(X(avg(xs, groups[k][0], groups[k][1])), Yf(cum[k]));
    }
    ctx.closePath();
    ctx.fill();
    // 2px surface gap between stacked bands.
    ctx.strokeStyle = C.surface;
    ctx.lineWidth = 2;
    ctx.beginPath();
    started = false;
    groups.forEach(([a, b], k) => {
      if (top[k] == null) return;
      const x = X(avg(xs, a, b));
      started ? ctx.lineTo(x, Yf(top[k])) : ctx.moveTo(x, Yf(top[k]));
      started = true;
    });
    ctx.stroke();
    top.forEach((v, k) => { if (v != null) cum[k] = v; });
  });
  axes(ctx, { pad, w, h, X, Y: Yf, x0, x1, yTicks: [0, 0.25, 0.5, 0.75, 1], yFormat: (v) => `${Math.round(v * 100)}%` });
  if (hover != null && hover < n) {
    const x = Math.round(X(xs[hover])) + 0.5;
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ph);
    ctx.stroke();
  }
  canvas._geo = { pad, pw, x0, x1 };
}

export function phaseChart(canvas, { xs, ys, xLabel, yLabel, color, hover = null }) {
  const { ctx, w, h } = prep(canvas);
  canvas._phase = null;
  const n = xs.length;
  if (n < 2) return empty(ctx);
  const pad = { l: 42, r: 12, t: 10, b: 34 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const xmax = niceMax(Math.max(...xs), 4), ymax = niceMax(Math.max(...ys), 4);
  const X = (v) => pad.l + (v / xmax) * pw, Yf = (v) => pad.t + (1 - v / ymax) * ph;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of niceTicks(0, ymax, 4)) {
    const y = Math.round(Yf(v)) + 0.5;
    ctx.strokeStyle = C.grid;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.fillText(compact(v), pad.l - 6, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of niceTicks(0, xmax, Math.max(2, Math.floor(pw / 70)))) ctx.fillText(compact(v), X(v), pad.t + ph + 5);
  ctx.fillText(xLabel, pad.l + pw / 2, h - 14);
  ctx.save();
  ctx.translate(11, pad.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, -6);
  ctx.restore();
  const chunks = 24;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  for (let k = 0; k < chunks; k++) {
    const a = Math.floor((k * (n - 1)) / chunks), b = Math.floor(((k + 1) * (n - 1)) / chunks);
    if (b <= a) continue;
    ctx.strokeStyle = alpha(color, 0.12 + 0.88 * ((k + 1) / chunks) ** 2);
    ctx.beginPath();
    const stride = Math.max(1, Math.floor((b - a) / 200));
    for (let i = a; i <= b; i += stride) (i === a ? ctx.moveTo(X(xs[i]), Yf(ys[i])) : ctx.lineTo(X(xs[i]), Yf(ys[i])));
    ctx.lineTo(X(xs[b]), Yf(ys[b]));
    ctx.stroke();
  }
  marker(ctx, X(xs[n - 1]), Yf(ys[n - 1]), color);
  if (hover != null && hover < n) marker(ctx, X(xs[hover]), Yf(ys[hover]), C.ink);
  canvas._phase = { X, Y: Yf, xs, ys };
}

export function hoverIndex(ticks, t) {
  let lo = 0, hi = ticks.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ticks[mid] < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(ticks[lo - 1] - t) < Math.abs(ticks[lo] - t)) lo--;
  return lo;
}
