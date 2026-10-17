/* Cross4D-JEPA project page: point-cloud viewers (WebGL2), the correspondence
   illustration, and the result charts (SVG). No dependencies.
   Data: data/demo.json + per-clip base64 JSON arrays, written by scripts/make_website_data.py. */
"use strict";

/* ------------------------------------------------------------------ results
   Values as printed in the paper (mean, std over seeds). */
const R = {
  main: [
    { name: "MSR-Action3D", short: "MSR-Action3D", teacher: "DINOv2", chance: 0.05, random: [0.42, 0.02], global: [0.537, 0.056], dense: [0.829, 0.025] },
    { name: "DeformingThings4D", short: "DeformingThings4D", teacher: "V-JEPA 2", chance: 0.03, random: [0.24, 0.02], global: [0.334, 0.013], dense: [0.516, 0.027] },
    { name: "NTU RGB+D 60", short: "NTU RGB+D 60", teacher: "V-JEPA 2", chance: 0.02, random: [0.11, 0.02], global: [0.206, 0.038], dense: [0.470, 0.010] },
  ],
  modality: [
    { name: "DINOv2", kind: "image", global: [0.537, 0.056], dense: [0.829, 0.025] },
    { name: "DINOv3", kind: "image", global: [0.633, 0.056], dense: [0.851, 0.022] },
    { name: "V-JEPA 2", kind: "video", global: [0.572, 0.039], dense: [0.852, 0.025] },
    { name: "VideoMAE V2", kind: "video", global: [0.620, 0.046], dense: [0.826, 0.025] },
  ],
  teacher: [
    { name: "Random init (no distillation)", v: [0.42, 0.02], miou: "4.6", none: true },
    { name: "Raw rendered pixels", v: [0.685, 0.049], miou: "13.95 ± 0.42" },
    { name: "ViT-B/16, untrained", v: [0.776, 0.011], miou: "15.81 ± 0.24" },
    { name: "ViT-B/16, ImageNet-supervised", v: [0.787, 0.048], miou: "14.24 ± 1.28" },
    { name: "DINOv2-L", v: [0.846, 0.004], miou: "16.61 ± 0.23" },
  ],
  gran: {
    x: [64, 16, 4, 2, 1],
    dense: [[0.846, 0.004], [0.828, 0.029], [0.838, 0.016], [0.736, 0.030], [0.519, 0.014]],
    global: [[0.537, 0.056], [0.578, 0.040], [0.560, 0.083], [0.505, 0.019], [0.478, 0.028]],
  },
  backbone: [
    { name: "Point4D (3.3M)", global: [0.537, 0.056], dense: [0.829, 0.025] },
    { name: "P4Transformer (42M)", global: [0.376, 0.027], dense: [0.617, 0.030] },
  ],
  radius: [[0.7, 0.617], [0.4, 0.612], [0.2, 0.676], [0.1, 0.577]],
  sota: [
    { name: "Random init", v: 42.0, kind: "none" },
    { name: "MaST-Pre (1,200 ep)", v: 44.7, kind: "pretext" },
    { name: "i-JEPA (300 ep)", v: 53.8, kind: "pretext" },
    { name: "Uni4D (1,200 ep)", v: 65.8, kind: "pretext" },
    { name: "Cross4D-JEPA (dense)", v: 82.9, kind: "dense" },
  ],
  published: [
    ["MaST-Pre", "P4Transformer", "LP", "63.8"], ["Uni4D", "P4Transformer", "LP", "84.6"], ["PointCMP", "PSTNet", "LP", "92.9"],
    ["PointCMP", "PSTNet", "FT", "93.3"], ["Uni4D", "P4Transformer", "FT", "93.4"], ["C2P", "PPTr", "FT", "94.8"],
    ["Cross4D-JEPA (ours)", "Point4D", "FT", "92.0"],
  ],
  labeleff: {
    x: [10, 25, 50, 100],
    scratch: [[0.29, 0.11], [0.63, 0.02], [0.76, 0.01], [0.82, 0.01]],
    dense: [[0.46, 0.04], [0.73, 0.02], [0.88, 0.02], [0.92, 0.01]],
  },
  hoi4d: [
    { group: null, name: "Random init", v: ["4.6", "7.3", "48.0"] },
    { group: "Zero-shot (pretrained on MSR-Action3D)" },
    { name: "Global", v: ["4.4", "7.2", "43.7"] },
    { name: "Dense (ours)", v: ["16.2", "21.4", "59.4"], best: true },
    { group: "In-domain (pretrained on HOI4D)" },
    { name: "Global", v: ["7.2", "10.8", "48.5"] },
    { name: "Dense (ours)", v: ["23.6", "32.5", "65.3"], best: true },
  ],
  ckpt: [
    ["cross4d_jepa_main.tar", "Table 1", "MSR dense (5 seeds) / global (8); DT4D and NTU dense / global (5 each)", "433 MB"],
    ["cross4d_jepa_modality.tar", "Table 2", "MSR dense / global with DINOv3, V-JEPA 2, VideoMAE V2 (5 seeds each)", "394 MB"],
    ["cross4d_jepa_teacher_capability.tar", "Table 3", "MSR dense with ViT-B/16 ImageNet, untrained ViT-B/16, raw pixels", "118 MB"],
    ["cross4d_jepa_locality_point4d.tar", "Table 4", "Point4D at 16 / 4 / 2 / 1 tokens per frame, dense / global", "315 MB"],
    ["cross4d_jepa_locality_p4t_r070.tar", "Table 4", "Official P4Transformer, radius 0.7, dense / global (5 seeds)", "1.68 GB"],
    ["cross4d_jepa_locality_p4t_radius.tar", "Table 4", "Official P4Transformer, radius 0.4 / 0.2 / 0.1, dense", "1.51 GB"],
    ["cross4d_jepa_hoi4d_semseg.tar", "Table 5", "Zero-shot (MSR) and in-domain (HOI4D) dense / global encoders", "53 MB"],
    ["cross4d_jepa_ssl_baselines.tar", "Table 6", "Uni4D, MaST-Pre, i-JEPA on the same Point4D student (3 seeds each)", "118 MB"],
  ],
};
const RELEASE = "https://github.com/tuananh1007/Cross4D-JEPA/releases/download/v1.0/";

/* HOI4D class colors: matplotlib tab20 + tab20b + tab20c, label % 60 (as in the paper). */
const SEG = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2", "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb", "#636363", "#969696", "#bdbdbd", "#d9d9d9"];

/* ------------------------------------------------------------------ helpers */
const $ = (s, r = document) => r.querySelector(s);
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 3) => v.toFixed(d).replace(/^0\./, "0.");
function hexRGB(h) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else e.setAttribute(k, v); }
  for (const k of kids) e.append(k);
  return e;
}

/* ------------------------------------------------------------------ math */
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
function lookAt(e, c, u) {
  let z = [e[0] - c[0], e[1] - c[1], e[2] - c[2]]; let l = Math.hypot(...z); z = z.map((a) => a / l);
  let x = [u[1] * z[2] - u[2] * z[1], u[2] * z[0] - u[0] * z[2], u[0] * z[1] - u[1] * z[0]]; l = Math.hypot(...x); x = x.map((a) => a / l);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -(x[0] * e[0] + x[1] * e[1] + x[2] * e[2]), -(y[0] * e[0] + y[1] * e[1] + y[2] * e[2]), -(z[0] * e[0] + z[1] * e[1] + z[2] * e[2]), 1];
}
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) o[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
  return o;
}

/* ------------------------------------------------------------------ orbit camera shared by linked views */
class Orbit {
  constructor(dist = 3.2) { this.home = { yaw: 0.35, pitch: 0.12, dist }; this.reset(); this.views = new Set(); }
  reset() { Object.assign(this, { ...this.home }); }
  redraw() { for (const v of this.views) v.draw(); }
  attach(canvas) {
    const ptrs = new Map(); let pinch = 0;
    canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); });
    canvas.addEventListener("pointermove", (e) => {
      if (!ptrs.has(e.pointerId)) return;
      const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      if (ptrs.size === 1) {
        this.yaw -= (e.clientX - prev[0]) * 0.01;
        this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - prev[1]) * 0.01));
      } else if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch) this.dist = Math.max(1.4, Math.min(8, this.dist * pinch / d));
        pinch = d;
      }
      this.redraw();
    });
    const up = (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = 0; };
    canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); this.dist = Math.max(1.4, Math.min(8, this.dist * Math.exp(e.deltaY * 0.001))); this.redraw(); }, { passive: false });
    canvas.addEventListener("dblclick", () => { this.reset(); this.redraw(); });
  }
}

/* ------------------------------------------------------------------ WebGL2 point view */
const VS = `#version 300 es
in vec3 aPos; in vec3 aCol; in float aTag;
uniform mat4 uMVP; uniform float uSize; uniform float uHi;
out vec3 vCol; out float vDim;
void main() {
  vec4 p = uMVP * vec4(-aPos.x, -aPos.y, aPos.z, 1.0);
  gl_Position = p;
  bool off = uHi >= 0.0 && abs(aTag - uHi) > 0.5;
  vDim = off ? 1.0 : 0.0;
  gl_PointSize = uSize / p.w * (off ? 0.8 : 1.0);
  vCol = aCol;
}`;
const FS = `#version 300 es
precision mediump float;
in vec3 vCol; in float vDim; out vec4 o;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0; float r = dot(c, c);
  if (r > 1.0) discard;
  vec3 col = vCol * (0.82 + 0.18 * (1.0 - r));
  if (vDim > 0.5) col = mix(col, vec3(0.32, 0.35, 0.40), 0.78);
  o = vec4(col, 1.0);
}`;
class PointView {
  constructor(canvas, orbit, opts = {}) {
    this.canvas = canvas; this.orbit = orbit; this.frame = 0; this.N = 0; this.T = 0; this.hi = -1;
    this.size = opts.size || 2.8;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
    this.gl = gl;
    if (!gl) return;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    this.p = p; this.u = { mvp: gl.getUniformLocation(p, "uMVP"), size: gl.getUniformLocation(p, "uSize"), hi: gl.getUniformLocation(p, "uHi") };
    this.vao = gl.createVertexArray(); this.buf = { pos: gl.createBuffer(), col: gl.createBuffer(), tag: gl.createBuffer() };
    gl.bindVertexArray(this.vao);
    const attr = (name, buf, size, type, norm) => {
      const loc = gl.getAttribLocation(p, name); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, norm, 0, 0); return loc;
    };
    attr("aPos", this.buf.pos, 3, gl.SHORT, true); attr("aCol", this.buf.col, 3, gl.UNSIGNED_BYTE, true);
    this.tagLoc = gl.getAttribLocation(p, "aTag"); gl.disableVertexAttribArray(this.tagLoc); gl.vertexAttrib1f(this.tagLoc, 0);
    orbit.views.add(this); orbit.attach(canvas);
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  setPositions(xyz, T, N) { if (!this.gl) return; const gl = this.gl; this.T = T; this.N = N; gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.pos); gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW); }
  setColors(rgb) { if (!this.gl) return; const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.col); gl.bufferData(gl.ARRAY_BUFFER, rgb, gl.STATIC_DRAW); this.draw(); }
  setTags(tags) {
    if (!this.gl) return; const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.tag);
    gl.bufferData(gl.ARRAY_BUFFER, tags, gl.STATIC_DRAW); gl.enableVertexAttribArray(this.tagLoc); gl.vertexAttribPointer(this.tagLoc, 1, gl.UNSIGNED_BYTE, false, 0, 0);
  }
  draw() {
    const gl = this.gl; if (!gl || !this.N) return;
    const c = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(c.clientWidth * dpr), hgt = Math.round(c.clientHeight * dpr);
    if (!w || !hgt) return;
    if (c.width !== w || c.height !== hgt) { c.width = w; c.height = hgt; }
    gl.viewport(0, 0, w, hgt);
    const bg = hexRGB(css("--stage-2") || "#161c25");
    gl.clearColor(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    const o = this.orbit, cp = Math.cos(o.pitch);
    const eye = [o.dist * cp * Math.sin(o.yaw), o.dist * Math.sin(o.pitch), -o.dist * cp * Math.cos(o.yaw)];
    const mvp = mul(perspective(0.62, w / hgt, 0.05, 30), lookAt(eye, [0, 0, 0], [0, 1, 0]));
    gl.useProgram(this.p); gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.u.mvp, false, mvp); gl.uniform1f(this.u.size, this.size * dpr * (hgt / dpr / 420) * 3.2); gl.uniform1f(this.u.hi, this.hi);
    gl.drawArrays(gl.POINTS, this.frame * this.N, this.N);
  }
}
function noWebGL(statusEl) { if (statusEl) { statusEl.textContent = "This viewer needs WebGL2, which your browser has turned off."; statusEl.hidden = false; } }

/* ------------------------------------------------------------------ playback */
class Player {
  constructor(views, slider, out, button, T = 24) {
    Object.assign(this, { views, slider, out, button, T, f: 0, playing: !REDUCED, last: 0 });
    slider.addEventListener("input", () => { this.set(+slider.value); this.pause(); });
    button.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    this.sync(); requestAnimationFrame((t) => this.tick(t));
  }
  set(f) { this.f = f; for (const v of this.views) { v.frame = f; v.draw(); } this.slider.value = f; this.out.value = `frame ${f + 1} / ${this.T}`; }
  play() { this.playing = true; this.sync(); }
  pause() { this.playing = false; this.sync(); }
  sync() { this.button.textContent = this.playing ? "Pause" : "Play"; this.button.setAttribute("aria-pressed", String(!this.playing)); }
  tick(t) {
    if (this.playing && t - this.last > 100 && this.views.some((v) => v.N)) { this.last = t; this.set((this.f + 1) % this.T); }
    requestAnimationFrame((tt) => this.tick(tt));
  }
}

/* ------------------------------------------------------------------ data */
let DEMO = null;
async function demo() { if (!DEMO) DEMO = fetch("data/demo.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }); return DEMO; }
async function bin(file) {
  const r = await fetch("data/" + file); if (!r.ok) throw new Error(`${file}: ${r.status}`);
  const s = atob((await r.json()).data), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u.buffer;
}

/* depth shading for the "input" mode: closer to the sensor = brighter, like the teacher's render */
function depthColors(xyz, T, N) {
  const out = new Uint8Array(T * N * 3);
  for (let f = 0; f < T; f++) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < N; i++) { const z = xyz[(f * N + i) * 3 + 2]; if (z < lo) lo = z; if (z > hi) hi = z; }
    for (let i = 0; i < N; i++) {
      const k = f * N + i, s = 1 - (xyz[k * 3 + 2] - lo) / (hi - lo + 1), g = Math.round(70 + 185 * s);
      out[k * 3] = g; out[k * 3 + 1] = g; out[k * 3 + 2] = g;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ hero explorer */
async function initExplorer() {
  const orbit = new Orbit(3.2);
  const vd = new PointView($("#v-dense"), orbit), vo = new PointView($("#v-other"), orbit);
  if (!vd.gl) { noWebGL($("#v-dense-status")); noWebGL($("#v-other-status")); return; }
  const player = new Player([vd, vo], $("#frame"), $("#frame-out"), $("#play"));
  const names = { global: ["Global distillation", "var(--global)", "seed 0"], random: ["Random init", "var(--none)", "untrained"], input: ["Input depth", "var(--stage-muted)", "geometry only"] };
  let mode = "global", clip = null;
  const setMode = (m) => {
    mode = m;
    document.querySelectorAll(".explorer [data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    $("#v-other-name").textContent = names[m][0]; $("#v-other-key").style.background = names[m][1]; $("#v-other-note").textContent = names[m][2];
    if (clip) vo.setColors(clip[m]);
  };
  document.querySelectorAll(".explorer [data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  let meta;
  try { meta = await demo(); } catch (e) { $("#v-dense-status").textContent = $("#v-other-status").textContent = "The demo data could not be loaded."; return; }
  const cache = new Map();
  const load = async (c) => {
    if (!cache.has(c.file)) {
      const buf = await bin(c.file), T = c.frames, N = c.points, n = T * N * 3;
      const xyz = new Int16Array(buf, 0, n), off = n * 2;
      cache.set(c.file, { T, N, xyz, dense: new Uint8Array(buf, off, n), global: new Uint8Array(buf, off + n, n), random: new Uint8Array(buf, off + 2 * n, n), input: depthColors(xyz, T, N) });
    }
    return cache.get(c.file);
  };
  const chips = $("#clip-chips");
  const select = async (c, btn) => {
    chips.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    try { clip = await load(c); } catch (e) { $("#v-dense-status").textContent = "The demo data could not be loaded."; return; }
    for (const v of [vd, vo]) v.setPositions(clip.xyz, clip.T, clip.N);
    $("#v-dense-status").hidden = $("#v-other-status").hidden = true;
    player.T = clip.T; $("#frame").max = clip.T - 1;
    vd.setColors(clip.dense); vo.setColors(clip[mode]); player.set(Math.min(player.f, clip.T - 1));
  };
  meta.msr.forEach((c, i) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": "false" }, c.action);
    b.addEventListener("click", () => select(c, b)); chips.append(b);
    if (i === 0) select(c, b);
  });
}

/* ------------------------------------------------------------------ correspondence illustration */
async function initCorrespondence() {
  const rc = $("#c-render"), pc = $("#c-patch"), orbit = new Orbit(3.0), pv = new PointView($("#c-points"), orbit, { size: 3.2 });
  let meta; try { meta = (await demo()).pipeline; } catch (e) { return; }
  if (!meta) return;
  const buf = await bin(meta.file), N = meta.points;
  const xyz = new Int16Array(buf, 0, N * 3), rgb = new Uint8Array(buf, N * 6, N * 3).slice(), tag = new Uint8Array(buf, N * 9, N);
  for (let i = 0; i < N; i++) if (tag[i] === 255) { rgb[i * 3] = 78; rgb[i * 3 + 1] = 84; rgb[i * 3 + 2] = 94; }
  if (pv.gl) { pv.setPositions(xyz, 1, N); pv.setTags(tag); pv.setColors(rgb); }
  $("#c-render-sub").textContent = `${meta.image_size} × ${meta.image_size} px, frame ${meta.frame + 1}`;
  $("#c-points-sub").textContent = `${meta.visible_points.toLocaleString()} of ${N.toLocaleString()} points get a target`;
  const S = meta.image_size, G = meta.grid, img = meta.render, P = meta.patch_rgb;
  let hi = -1;
  const paint = () => {
    for (const [cv, withRender] of [[rc, true], [pc, false]]) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2), w = Math.round(cv.clientWidth * dpr);
      if (!w) continue; cv.width = cv.height = w;
      const ctx = cv.getContext("2d"), cell = w / G;
      ctx.fillStyle = css("--stage") || "#0d1117"; ctx.fillRect(0, 0, w, w);
      if (withRender) {
        const id = ctx.createImageData(S, S);
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { const v = img[y][x], k = (y * S + x) * 4; id.data[k] = id.data[k + 1] = id.data[k + 2] = v; id.data[k + 3] = 255; }
        const tmp = document.createElement("canvas"); tmp.width = tmp.height = S; tmp.getContext("2d").putImageData(id, 0, 0);
        ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, w, w);
        ctx.strokeStyle = "rgba(255,255,255,0.22)"; ctx.lineWidth = 1;
        for (let i = 1; i < G; i++) { ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, w); ctx.moveTo(0, i * cell); ctx.lineTo(w, i * cell); ctx.stroke(); }
      } else {
        for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { const c = P[y][x]; ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5); }
      }
      if (hi >= 0) { ctx.strokeStyle = "#ffffff"; ctx.lineWidth = Math.max(2, dpr * 2); ctx.strokeRect((hi % G) * cell + 1, Math.floor(hi / G) * cell + 1, cell - 2, cell - 2); }
    }
  };
  const setHi = (k) => { hi = k; pv.hi = k; paint(); pv.draw(); };
  for (const cv of [rc, pc]) {
    cv.addEventListener("pointermove", (e) => { const r = cv.getBoundingClientRect(); const x = Math.floor((e.clientX - r.left) / r.width * G), y = Math.floor((e.clientY - r.top) / r.height * G); if (x >= 0 && y >= 0 && x < G && y < G) setHi(y * G + x); });
    cv.addEventListener("pointerleave", () => setHi(-1));
  }
  rc.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -G, ArrowDown: G }[e.key];
    if (e.key === "Escape") { setHi(-1); return; }
    if (d === undefined) return; e.preventDefault();
    const cur = hi < 0 ? Math.floor(G * G / 2) : hi, x = cur % G + (Math.abs(d) === 1 ? d : 0), y = Math.floor(cur / G) + (Math.abs(d) === G ? Math.sign(d) : 0);
    setHi(Math.max(0, Math.min(G - 1, y)) * G + Math.max(0, Math.min(G - 1, x)));
  });
  new ResizeObserver(paint).observe(rc); paint();
}

/* ------------------------------------------------------------------ HOI4D segmentation */
async function initSeg() {
  const orbit = new Orbit(3.0);
  const views = { gt: new PointView($("#s-gt"), orbit), dense: new PointView($("#s-dense"), orbit), global: new PointView($("#s-global"), orbit) };
  if (!views.gt.gl) { for (const k of ["gt", "dense", "global"]) noWebGL($(`#s-${k}-status`)); return; }
  const player = new Player(Object.values(views), $("#seg-frame"), $("#seg-frame-out"), $("#seg-play"));
  let meta; try { meta = (await demo()).hoi4d; } catch (e) { return; }
  if (!meta || !meta.length) return;
  const lut = SEG.map(hexRGB); lut[0] = [219, 219, 219];
  const colorize = (lab) => { const o = new Uint8Array(lab.length * 3); for (let i = 0; i < lab.length; i++) { const c = lut[lab[i] % 60]; o[i * 3] = c[0]; o[i * 3 + 1] = c[1]; o[i * 3 + 2] = c[2]; } return o; };
  const chips = $("#seg-chips");
  const select = async (c, btn) => {
    chips.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    const buf = await bin(c.file), T = c.frames, N = c.points, n = T * N;
    const xyz = new Int16Array(buf, 0, n * 3), off = n * 6;
    const labs = { gt: new Uint8Array(buf, off, n), dense: new Uint8Array(buf, off + n, n), global: new Uint8Array(buf, off + 2 * n, n) };
    for (const [k, v] of Object.entries(views)) { v.setPositions(xyz, T, N); v.setColors(colorize(labs[k])); $(`#s-${k}-status`).hidden = true; }
    $("#s-dense-acc").textContent = `${(100 * c.point_accuracy.dense).toFixed(1)}% points correct`;
    $("#s-global-acc").textContent = `${(100 * c.point_accuracy.global).toFixed(1)}% points correct`;
    player.T = T; $("#seg-frame").max = T - 1; player.set(Math.min(player.f, T - 1));
  };
  meta.forEach((c, i) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": "false" }, `Clip ${i + 1}`);
    b.addEventListener("click", () => select(c, b)); chips.append(b);
    if (i === 0) select(c, b);
  });
}

/* ------------------------------------------------------------------ charts */
const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, parent) { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.append(e); return e; }
const tip = () => $("#tip");
function showTip(evt, title, rows) {
  const t = tip(); t.replaceChildren(h("div", { class: "t" }, title));
  for (const [label, value, color] of rows) {
    const key = h("i"); key.style.background = color;
    t.append(h("div", { class: "r" }, key, h("b", {}, value), h("span", {}, label)));
  }
  t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8), y = Math.min(evt.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = x + "px"; t.style.top = y + "px";
}
function hideTip() { tip().hidden = true; }
function hover(el, title, rows) {
  el.setAttribute("tabindex", "0");
  el.addEventListener("pointermove", (e) => showTip(e, title, rows()));
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, title, rows()); });
  el.addEventListener("blur", hideTip);
}
function legend(items) {
  const l = h("div", { class: "legend" });
  for (const [label, color, shape] of items) { const i = h("i", { class: shape || "" }); i.style.background = color; l.append(h("span", {}, i, label)); }
  return l;
}
function tableView(head, rows) {
  const t = h("table"); const tr = h("tr"); head.forEach((c) => tr.append(h("th", { scope: "col" }, c)));
  t.append(h("thead", {}, tr)); const tb = h("tbody");
  for (const r of rows) { const row = h("tr"); r.forEach((c) => row.append(h("td", {}, c))); tb.append(row); }
  t.append(tb);
  return h("details", { class: "table" }, h("summary", {}, "Show the numbers"), h("div", { class: "tscroll" }, t));
}
const pm = (a, d = 3) => `${fmt(a[0], d)} ± ${a[1].toFixed(d).replace(/^0/, "")}`;
/* a column with a 4px rounded data end, square at the baseline */
function colPath(x, y0, y1, w, r = 4) { const rr = Math.min(r, w / 2, Math.abs(y0 - y1)); return `M${x},${y0}V${y1 + rr}Q${x},${y1} ${x + rr},${y1}H${x + w - rr}Q${x + w},${y1} ${x + w},${y1 + rr}V${y0}Z`; }
function barPath(x0, x1, y, hgt, r = 4) { const rr = Math.min(r, hgt / 2, Math.abs(x1 - x0)); return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + hgt - rr}Q${x1},${y + hgt} ${x1 - rr},${y + hgt}H${x0}Z`; }
function yAxis(g, y, ticks, x0, x1, f = (v) => v.toFixed(1)) {
  const grid = s("g", { class: "grid" }, g), tk = s("g", { class: "tick" }, g);
  for (const v of ticks) { s("line", { x1: x0, x2: x1, y1: y(v), y2: y(v) }, grid); const t = s("text", { x: x0 - 8, y: y(v) + 4, "text-anchor": "end" }, tk); t.textContent = f(v); }
}
function errBar(g, x, y0, y1, cap = 4) { s("path", { class: "err", d: `M${x},${y0}V${y1}M${x - cap},${y0}H${x + cap}M${x - cap},${y1}H${x + cap}`, fill: "none" }, g); }
function responsive(el, render) { let w = 0; new ResizeObserver(() => { const nw = Math.round(el.clientWidth); if (nw && nw !== w) { w = nw; render(nw); } }).observe(el); }

function chartMain() {
  const box = $("#ch-main"); box.append(legend([["Random init", "var(--none)"], ["Global distillation", "var(--global)"], ["Dense (ours)", "var(--dense)"]]));
  const holder = h("div"); box.append(holder);
  box.append(tableView(["Dataset", "Teacher", "Chance", "Random init", "Global", "Dense (ours)"], R.main.map((d) => [d.name, d.teacher, d.chance.toFixed(2), pm(d.random, 2), pm(d.global), pm(d.dense)])));
  responsive(holder, (W) => {
    const H = 280, m = { l: 36, r: 6, t: 18, b: 40 }, svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Linear-probe accuracy for random init, global and dense on three datasets" });
    const y = (v) => m.t + (1 - v) * (H - m.t - m.b);
    yAxis(svg, y, [0, 0.2, 0.4, 0.6, 0.8, 1.0], m.l, W - m.r);
    const gw = (W - m.l - m.r) / R.main.length, bw = Math.min(24, gw / 5), gap = 2;
    R.main.forEach((d, i) => {
      const cx = m.l + gw * (i + 0.5), keys = [["random", "var(--none)", "Random init"], ["global", "var(--global)", "Global"], ["dense", "var(--dense)", "Dense (ours)"]];
      keys.forEach(([k, col, lab], j) => {
        const x = cx + (j - 1.5) * (bw + gap) + gap / 2, v = d[k];
        s("path", { class: "mark", d: colPath(x, y(0), y(v[0]), bw), fill: col }, svg);
        errBar(svg, x + bw / 2, y(v[0] - v[1]), y(v[0] + v[1]), 3);
        if (k === "dense") { const t = s("text", { class: "val", x: x + bw / 2, y: y(v[0] + v[1]) - 6, "text-anchor": "middle" }, svg); t.textContent = fmt(v[0]); }
        const hit = s("rect", { class: "hit", x: x - gap, y: m.t, width: bw + 2 * gap, height: y(0) - m.t }, svg);
        hover(hit, `${d.name} · ${lab}`, () => [[lab, pm(v, k === "random" ? 2 : 3), col], ["Chance", d.chance.toFixed(2), "var(--muted)"]]);
      });
      const t = s("text", { x: cx, y: H - m.b + 18, "text-anchor": "middle" }, svg); t.textContent = d.short;
      const t2 = s("text", { x: cx, y: H - m.b + 33, "text-anchor": "middle", class: "tick" }, svg); t2.textContent = `${d.teacher} teacher`; t2.style.fill = "var(--muted)";
    });
    s("line", { class: "base", x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, svg);
    holder.replaceChildren(svg);
  });
}

function dumbbell(boxSel, rows, opts) {
  const box = $(boxSel); box.append(legend([["Global distillation", "var(--global)", "dot"], ["Dense (ours)", "var(--dense)", "dot"]]));
  const holder = h("div"); box.append(holder);
  box.append(tableView(opts.head, rows.map(opts.row)));
  responsive(holder, (W) => {
    const rh = 44, m = { l: Math.min(150, W * 0.36), r: 54, t: 6, b: 28 }, H = m.t + m.b + rh * rows.length;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label });
    const x = (v) => m.l + (v - opts.min) / (opts.max - opts.min) * (W - m.l - m.r);
    const grid = s("g", { class: "grid" }, svg), tk = s("g", { class: "tick" }, svg);
    for (const v of opts.ticks) { s("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, grid); const t = s("text", { x: x(v), y: H - m.b + 18, "text-anchor": "middle" }, tk); t.textContent = v.toFixed(1); }
    rows.forEach((d, i) => {
      const cy = m.t + rh * (i + 0.5);
      const lab = s("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end" }, svg); lab.textContent = d.name;
      if (d.kind) { const k = s("text", { x: m.l - 12, y: cy + 18, "text-anchor": "end", class: "tick" }, svg); k.textContent = d.kind; k.style.fill = "var(--muted)"; lab.setAttribute("y", cy - 2); }
      s("line", { x1: x(d.global[0]), x2: x(d.dense[0]), y1: cy, y2: cy, stroke: "var(--line)", "stroke-width": 3 }, svg);
      for (const [k, col] of [["global", "var(--global)"], ["dense", "var(--dense)"]]) {
        s("circle", { class: "mark", cx: x(d[k][0]), cy, r: 6, fill: col, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      }
      const t = s("text", { class: "val", x: x(d.dense[0]) + 12, y: cy + 4 }, svg); t.textContent = `+${(d.dense[0] - d.global[0]).toFixed(2)}`;
      const hit = s("rect", { class: "hit", x: m.l, y: cy - rh / 2, width: W - m.l - m.r + 40, height: rh }, svg);
      hover(hit, d.name, () => [["Dense (ours)", pm(d.dense), "var(--dense)"], ["Global", pm(d.global), "var(--global)"], ["Gap", `+${(d.dense[0] - d.global[0]).toFixed(3)}`, "transparent"]]);
    });
    holder.replaceChildren(svg);
  });
}

function hbars(boxSel, rows, opts) {
  const box = $(boxSel); if (opts.legend) box.append(legend(opts.legend));
  const holder = h("div"); box.append(holder);
  box.append(tableView(opts.head, rows.map(opts.row)));
  responsive(holder, (W) => {
    const rh = 38, bh = 20, m = { l: Math.min(190, W * 0.42), r: 92, t: 4, b: 26 }, H = m.t + m.b + rh * rows.length;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label });
    const x = (v) => m.l + v / opts.max * (W - m.l - m.r);
    const grid = s("g", { class: "grid" }, svg), tk = s("g", { class: "tick" }, svg);
    for (const v of opts.ticks) { s("line", { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, grid); const t = s("text", { x: x(v), y: H - m.b + 18, "text-anchor": "middle" }, tk); t.textContent = opts.tickFmt(v); }
    rows.forEach((d, i) => {
      const cy = m.t + rh * (i + 0.5), v = opts.value(d), col = opts.color(d);
      const lab = s("text", { x: m.l - 10, y: cy + 4, "text-anchor": "end" }, svg); lab.textContent = d.name;
      s("path", { class: "mark", d: barPath(x(0), x(v[0]), cy - bh / 2, bh), fill: col }, svg);
      let end = x(v[0]);
      if (v[1]) { s("path", { class: "err", d: `M${x(v[0] - v[1])},${cy}H${x(v[0] + v[1])}M${x(v[0] - v[1])},${cy - 4}V${cy + 4}M${x(v[0] + v[1])},${cy - 4}V${cy + 4}`, fill: "none" }, svg); end = x(v[0] + v[1]); }
      const t = s("text", { class: "val", x: end + 8, y: cy + 4 }, svg); t.textContent = opts.valFmt(d);
      const hit = s("rect", { class: "hit", x: 0, y: cy - rh / 2, width: W, height: rh }, svg);
      hover(hit, d.name, () => opts.tip(d));
    });
    s("line", { class: "base", x1: x(0), x2: x(0), y1: m.t, y2: H - m.b }, svg);
    holder.replaceChildren(svg);
  });
}

function lines(boxSel, opts) {
  const box = $(boxSel); box.append(legend(opts.series.map((sr) => [sr.label, sr.color, "ln"])));
  const holder = h("div"); box.append(holder);
  box.append(tableView(opts.head, opts.x.map((xv, i) => [opts.xFmt(xv), ...opts.series.map((sr) => pm(sr.data[i], opts.digits))])));
  responsive(holder, (W) => {
    const H = 270, m = { l: 36, r: 64, t: 14, b: 44 }, n = opts.x.length;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label });
    const x = (i) => m.l + 18 + i / (n - 1) * (W - m.l - m.r - 36), y = (v) => m.t + (opts.max - v) / (opts.max - opts.min) * (H - m.t - m.b);
    yAxis(svg, y, opts.ticks, m.l, W - m.r);
    opts.x.forEach((xv, i) => { const t = s("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, svg); t.textContent = opts.xFmt(xv); t.style.fill = "var(--muted)"; });
    const xl = s("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, svg); xl.textContent = opts.xLabel;
    if (opts.band) {
      const [a, b] = opts.band, top = opts.x.map((_, i) => `${x(i)},${y(b.data[i][0])}`), bot = opts.x.map((_, i) => `${x(i)},${y(a.data[i][0])}`).reverse();
      s("polygon", { points: [...top, ...bot].join(" "), fill: b.color, opacity: 0.1 }, svg);
    }
    for (const sr of opts.series) {
      s("polyline", { class: "mark", points: opts.x.map((_, i) => `${x(i)},${y(sr.data[i][0])}`).join(" "), fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      sr.data.forEach((d, i) => { errBar(svg, x(i), y(d[0] - d[1]), y(d[0] + d[1]), 3); s("circle", { cx: x(i), cy: y(d[0]), r: 4.5, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2 }, svg); });
      const last = sr.data[n - 1], t = s("text", { class: "val", x: x(n - 1) + 10, y: y(last[0]) + 4 }, svg); t.textContent = fmt(last[0], opts.digits);
    }
    const cross = s("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" }, svg);
    opts.x.forEach((xv, i) => {
      const w = (W - m.l - m.r - 36) / (n - 1), hit = s("rect", { class: "hit", x: x(i) - w / 2, y: m.t, width: w, height: H - m.t - m.b }, svg);
      hover(hit, opts.tipTitle(xv), () => opts.series.map((sr) => [sr.label, pm(sr.data[i], opts.digits), sr.color]));
      hit.addEventListener("pointerenter", () => { cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible"); });
      hit.addEventListener("pointerleave", () => cross.setAttribute("visibility", "hidden"));
    });
    holder.replaceChildren(svg);
  });
}

function initCharts() {
  chartMain();
  dumbbell("#ch-modality", R.modality, {
    label: "Dense versus global linear-probe accuracy for four teachers", min: 0.4, max: 0.9, ticks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    head: ["Teacher", "Modality", "Global", "Dense (ours)", "Gap"], row: (d) => [d.name, d.kind, pm(d.global), pm(d.dense), `+${(d.dense[0] - d.global[0]).toFixed(3)}`],
  });
  const floor = R.teacher[0].v[0], top = R.teacher[R.teacher.length - 1].v[0];
  hbars("#ch-teacher", R.teacher, {
    label: "Linear-probe accuracy by teacher type", max: 1.0, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1.0], tickFmt: (v) => v.toFixed(1),
    value: (d) => d.v, color: (d) => (d.none ? "var(--none)" : "var(--dense)"),
    valFmt: (d) => (d.none ? fmt(d.v[0], 2) : `${fmt(d.v[0])} (${Math.round(100 * (d.v[0] - floor) / (top - floor))}%)`),
    tip: (d) => [["Linear probe", pm(d.v, d.none ? 2 : 3), d.none ? "var(--none)" : "var(--dense)"], ["HOI4D zero-shot mIoU", d.miou, "transparent"], ...(d.none ? [] : [["Share of DINOv2's gain", `${Math.round(100 * (d.v[0] - floor) / (top - floor))}%`, "transparent"]])],
    head: ["Teacher", "Linear probe", "Share of gain", "HOI4D zero-shot mIoU"], row: (d) => [d.name, pm(d.v, d.none ? 2 : 3), d.none ? "0%" : `${Math.round(100 * (d.v[0] - floor) / (top - floor))}%`, d.miou],
  });
  lines("#ch-gran", {
    label: "Dense and global accuracy as token granularity drops", x: R.gran.x, xFmt: (v) => String(v), xLabel: "tokens per frame (n₂), Point4D at 3.28M parameters",
    tipTitle: (v) => `${v} tokens per frame`, min: 0.4, max: 0.9, ticks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], digits: 3,
    series: [{ label: "Global distillation", color: "var(--global)", data: R.gran.global }, { label: "Dense (ours)", color: "var(--dense)", data: R.gran.dense }],
    head: ["Tokens per frame", "Global", "Dense (ours)"],
  });
  const gb = $("#ch-gran"), extra = h("div", { class: "tscroll" });
  const bt = h("table"); bt.append(h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Backbone"), h("th", { scope: "col" }, "Global"), h("th", { scope: "col" }, "Dense (ours)"))));
  const tb = h("tbody"); R.backbone.forEach((d) => tb.append(h("tr", {}, h("td", {}, d.name), h("td", {}, pm(d.global)), h("td", { class: "best" }, pm(d.dense)))));
  R.radius.forEach(([r, v]) => tb.append(h("tr", {}, h("td", {}, `P4Transformer, radius ${r}`), h("td", {}, "–"), h("td", {}, fmt(v)))));
  bt.append(tb); extra.append(bt); gb.querySelector("details.table").append(h("p", { style: "margin:12px 4px 4px;color:var(--muted);font-size:.85rem" }, "Across backbones (Table 4, top and bottom blocks)"), extra);
  const colOf = { none: "var(--none)", pretext: "var(--pretext)", dense: "var(--dense)" };
  hbars("#ch-sota", R.sota, {
    label: "Linear-probe accuracy of SSL methods on the same Point4D student", max: 100, ticks: [0, 20, 40, 60, 80, 100], tickFmt: (v) => String(v),
    legend: [["No pretraining", "var(--none)"], ["Intra-modal pretext", "var(--pretext)"], ["Dense (ours)", "var(--dense)"]],
    value: (d) => [d.v, 0], color: (d) => colOf[d.kind], valFmt: (d) => d.v.toFixed(1),
    tip: (d) => [["Linear probe, %", d.v.toFixed(1), colOf[d.kind]]],
    head: ["Method (Point4D, 3.3M)", "Linear probe, %"], row: (d) => [d.name, d.v.toFixed(1)],
  });
  const st = $("#ch-sota details.table");
  const pt = h("table"); pt.append(h("thead", {}, h("tr", {}, ...["Published method", "Backbone", "Protocol", "Accuracy, %"].map((c) => h("th", { scope: "col" }, c)))));
  const pb = h("tbody"); R.published.forEach((r) => pb.append(h("tr", {}, ...r.map((c) => h("td", {}, c))))); pt.append(pb);
  st.append(h("p", { style: "margin:12px 4px 4px;color:var(--muted);font-size:.85rem" }, "Published results on their native backbones (LP: linear probe, FT: fine-tuning)"), h("div", { class: "tscroll" }, pt));
  lines("#ch-labeleff", {
    label: "Fine-tuning accuracy versus label fraction", x: R.labeleff.x, xFmt: (v) => `${v}%`, xLabel: "labeled training clips",
    tipTitle: (v) => `${v}% of labels`, min: 0.1, max: 1.0, ticks: [0.2, 0.4, 0.6, 0.8, 1.0], digits: 2,
    series: [{ label: "From scratch", color: "var(--none)", data: R.labeleff.scratch }, { label: "Dense init (ours)", color: "var(--dense)", data: R.labeleff.dense }],
    band: [{ data: R.labeleff.scratch }, { data: R.labeleff.dense, color: "var(--dense)" }],
    head: ["Labels", "From scratch", "Dense init (ours)"],
  });
}

/* ------------------------------------------------------------------ tables + copy */
function initTables() {
  const t = $("#t-hoi4d");
  t.append(h("thead", {}, h("tr", {}, ...["Encoder", "mIoU, %", "mAcc, %", "OA, %"].map((c) => h("th", { scope: "col" }, c)))));
  const tb = h("tbody");
  for (const r of R.hoi4d) {
    if (r.group) { tb.append(h("tr", { class: "group" }, h("td", { colspan: "4" }, r.group))); continue; }
    if (r.group === undefined && !r.name) continue;
    tb.append(h("tr", {}, h("td", {}, r.name), ...r.v.map((v) => h("td", { class: r.best ? "best" : "" }, v))));
  }
  t.append(tb);
  const c = $("#t-ckpt");
  c.append(h("thead", {}, h("tr", {}, ...["Archive", "Paper table", "Contents", "Size"].map((x) => h("th", { scope: "col" }, x)))));
  const cb = h("tbody");
  for (const [f, tab, what, size] of R.ckpt) cb.append(h("tr", {}, h("td", {}, h("a", { href: RELEASE + f }, f)), h("td", {}, tab), h("td", { style: "text-align:left;white-space:normal" }, what), h("td", {}, size)));
  c.append(cb);
  $("#copy-bib").addEventListener("click", async (e) => {
    const b = e.currentTarget, txt = $("#bibtex").textContent;
    try { await navigator.clipboard.writeText(txt); b.textContent = "Copied"; }
    catch (err) { const r = document.createRange(); r.selectNodeContents($("#bibtex")); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); b.textContent = "Selected, press Ctrl+C"; }
    setTimeout(() => (b.textContent = "Copy"), 2200);
  });
}

/* ------------------------------------------------------------------ boot */
initCharts();
initTables();
initExplorer();
initCorrespondence();
initSeg();
