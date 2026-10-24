// App controller: real-time clock, interpolation, input, tools and API calls.
import * as THREE from "three";
import { World3D, decode } from "./scene.js";
import { Animals } from "./animals.js";
import { $, ACT_EMOJI, TOOLS, UI } from "./ui.js";

// At 1x the world runs 4 ticks per second, so one simulated day lasts a minute
// and animals walk at a natural-looking pace.
const BASE_TPS = 4;
const FRAME_MS = 120; // how often to fetch the latest state while running
const SKIP_FRAME_MS = 500; // ...and while skipping ahead (fewer frames leave more time for simulating)
const SPEEDS = [[0.5, "½×"], [1, "1×"], [2, "2×"], [5, "5×"], [10, "10×"], [25, "25×"], [60, "Max"]];
const EVENT_COLORS = {
  birth: "#ff8fb8", predation: "#d64545", starvation: "#a39e93", dehydration: "#d9b56b",
  "old age": "#ffffff", disease: "#8bc34a", removed: "#c77dff", thump: "#9b7b55", cache: "#6f4f30",
  fight: "#ff7a1a", fighting: "#b3261e",
};

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.error || `${path} failed (${res.status})`);
  return data;
}

const lerpAngle = (a, b, t) => {
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
};

class App {
  constructor() {
    this.canvas = $("#view");
    this.w3 = new World3D(this.canvas);
    this.speeds = SPEEDS;
    this.speed = 2;
    this.running = true;
    this.tool = "select";
    this.brush = 3;
    this.spawn = { species: null, preset: "", count: 4 };
    this.bubbles = true;
    this.follow = false;
    this.breeding = false;
    this.selected = null;
    this.detail = null;
    this.hist = null;
    this.ledger = null; // daily records (births, deaths by cause, adults, food), for the AI and Deaths tabs
    this._ledgerTick = -Infinity;
    this.busy = false;
    this.rendered = [];
    this.prev = new Map();
    this.curr = [];
    this.tPrev = this.tCurr = this.renderTick = 0;
    this.rate = 0; // ticks per second the server is actually achieving, measured from frames
    this.frameAt = 0;
    this.clock = null;
    this.plantsAt = 0;
    this.trail = [];
    this.lastInsight = 0;
    this.endOnExtinction = true;
    this.endShown = false;
    this.ui = new UI(this);
    this.keys = new Set();
    this._initSelectionVisuals();
    this._bindInput();
    new ResizeObserver(() => this.w3.resize()).observe(this.canvas);
    // Panels sit just below the top bar, whose height depends on how it wraps.
    const topbar = document.querySelector(".topbar");
    new ResizeObserver(() => {
      document.documentElement.style.setProperty("--top", `${topbar.offsetTop + topbar.offsetHeight + 10}px`);
    }).observe(topbar);
  }

  get tps() {
    return BASE_TPS * this.speed;
  }

  get skipping() {
    return !!this.clock?.skip;
  }

  async start() {
    try {
      await this.load(await api("/api/world"));
    } catch (err) {
      this.ui.toast(`Could not reach the simulation server: ${err.message}`, "error");
      return;
    }
    this.last = performance.now();
    requestAnimationFrame((t) => this.loop(t));
    setInterval(() => { if (this.running || !this.hist) this.pullHistory(); }, 1000);
    setInterval(() => {
      if (!$("#drawer").hidden && (this.ui.tab === "ai" || this.ui.tab === "deaths")) this.pullLedger();
    }, 2000);
  }

  async load(info) {
    this.info = info;
    this.spawn.species = info.species[0].name;
    this.w3.setTerrain(info, info.terrain);
    this.animals?.dispose();
    this.animals = new Animals(this.w3, info, () => this.w3.toon(0xffffff));
    this.ledger = null;
    this._ledgerTick = -Infinity;
    this.ui.build(info);
    this.hist = null;
    this.selected = null;
    this.detail = null;
    this.prev = new Map();
    this.curr = [];
    const f = await api("/api/frame", { plants: true });
    this.renderTick = this.tPrev = f.tick;
    this.applyFrame(f);
    await this.sendClock({ running: this.running, speed: this.speedValue() });
    this.prev = new Map(this.curr.map((a) => [a.id, a]));
    await this.pullHistory();
    const log = await api("/api/insights");
    this.ui.setInsights(log);
    this.lastInsight = log.length ? log[log.length - 1].id : 0;
    this._applyUrlOptions();
  }

  // ?inspect=fox (or an animal's id) selects and follows it (&tab=brain
  // opens an inspector tab, &memory=1 shows its memory map); ?data=genes opens
  // the data drawer on a tab. Handy for sharing a view.
  _applyUrlOptions() {
    const params = new URLSearchParams(location.search);
    const species = params.get("inspect");
    if (species) {
      // ?inspect=fox picks a grown fox; ?inspect=1234 picks that animal.
      const si = this.info.species.findIndex((s) => s.name === species);
      const a = /^\d+$/.test(species) ? this.curr.find((c) => c.id === Number(species))
        : this.curr.find((c) => c.si === si && !(c.flags & 4));
      if (a) {
        this.select(a.id).then(() => {
          this.follow = true;
          this.ui.inspTab = params.get("tab") || this.ui.inspTab;
          if (params.get("memory")) this.showMemory = true;
          this.ui.renderInspector();
        });
      }
    }
    // ?why=explain opens the explanations with a fresh report; ?why=recent
    // replays the latest explanations as pop-ups.
    const why = params.get("why");
    if (why === "explain") this.explain();
    if (why === "recent") this.ui.pushInsights(this.ui.insightLog.splice(-3).filter((i) => i.level !== "report"));
    const tab = params.get("data");
    if (tab) {
      this.ui.tab = tab;
      $("#drawer").hidden = false;
      document.body.classList.add("drawer-open");
      this.ui.buildDrawer();
    }
  }

  // ------------------------------------------------------------------ simulation clock

  // The server runs the simulation on its own clock; the browser sets the
  // speed and fetches the latest state a few times a second.
  speedValue() {
    return this.speed >= 60 ? "max" : this.speed;
  }

  async sendClock(body) {
    try {
      this.clock = await api("/api/clock", body);
      this.ui.updateSimClock(this.clock);
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  setSpeed(v) {
    this.speed = v;
    this.running = true;
    this.ui.buildSpeeds();
    this.sendClock({ speed: this.speedValue(), running: true });
  }

  togglePlay() {
    if (this.skipping) return this.cancelSkip();
    this.running = !this.running;
    this.ui.buildSpeeds();
    this.sendClock({ running: this.running, speed: this.speedValue() });
  }

  async skip(days) {
    $("#skip-menu").hidden = true;
    await this.sendClock({ skip_days: days, speed: this.speedValue() });
    this.ui.toast(`⏩ Skipping ahead ${days >= 12 ? `${days / 12} year${days > 12 ? "s" : ""}` : `${days} day${days > 1 ? "s" : ""}`}. Every tick is still simulated in full.`);
  }

  cancelSkip() {
    return this.sendClock({ cancel_skip: true });
  }

  // Kept for the sandbox tools: refresh the view now (the clock does the stepping).
  requestFrame() {
    return this.fetchFrame();
  }

  async fetchFrame() {
    if (this.busy) return;
    this.busy = true;
    this.frameAt = performance.now();
    const plants = performance.now() - this.plantsAt > (this.skipping ? 1500 : 450);
    try {
      const f = await api("/api/frame", { selected: this.selected, plants, insights_since: this.lastInsight });
      this.applyFrame(f);
    } catch (err) {
      this.ui.toast(err.message, "error");
    } finally {
      this.busy = false;
    }
  }

  applyFrame(f) {
    // How fast the simulation is really going, from the ticks between frames.
    const now = performance.now();
    if (this._lastFrame && f.tick >= this._lastFrame.tick) {
      const dt = (now - this._lastFrame.at) / 1000;
      if (dt > 0.02) this.rate += 0.35 * ((f.tick - this._lastFrame.tick) / dt - this.rate);
    }
    this._lastFrame = { tick: f.tick, at: now };
    if (f.clock) {
      if (this.clock?.skip && !f.clock.skip) {
        const tpd = this.info.clock.ticks_per_day;
        this.ui.toast(`⏩ Skipped ahead to day ${Math.floor((f.tick + tpd / 4) / tpd) + 1}.`);
        this.pullHistory();
      }
      this.clock = f.clock;
      if (this.running && !f.clock.running) {
        this.running = false; // the run ended (or a skip finished while paused)
        this.ui.buildSpeeds();
      } else if (!this.running && f.clock.running && !f.clock.skip) {
        this.running = true;
        this.ui.buildSpeeds();
      }
      this.ui.updateSimClock(f.clock);
    }
    // Freeze what is on screen now as the start of the next interpolation.
    this.prev = new Map(this.rendered.map((a) => [a.id, a]));
    this.tPrev = Math.min(this.renderTick, this.tCurr || this.renderTick);
    this.curr = f.animals.map(([id, si, x, y, h, act, flags, coat, size, vision]) => ({ id, si, x, y, h, act, flags, coat, size, vision }));
    this.tCurr = f.tick;
    if (this.renderTick > this.tCurr) this.renderTick = this.tCurr;
    this.frame = f;
    this.env = f.env;
    if (this.animals) this.animals.moult = f.env.moult || 0;
    if (f.plants) {
      this.plantsAt = performance.now();
      this.w3.updateGround(decode(f.plants), f.env);
      if (f.fruit) this.w3.updateFruit(decode(f.fruit));
    }
    if (f.terrain_version !== this.w3.version) this._terrainDirty = true;
    for (const [kind, x, y, , id] of f.events) {
      if (kind === "pounce") {
        this.animals.pounce(id);
        continue;
      }
      const color = EVENT_COLORS[kind];
      if (!color) continue;
      if (kind === "thump" || kind === "cache") this.w3.burst(x, y, color, 6, 1.2);
      else this.w3.burst(x, y, color, kind === "birth" ? 6 : 12, kind === "birth" ? 3.2 : 2.4);
    }
    this.w3.setCarrion?.(f.carrion);
    this.w3.setCaches?.(f.caches);
    this.endOnExtinction = f.end_on_extinction;
    if (f.finished && !this.endShown) {
      this.endShown = true;
      this.running = false;
      this.ui.buildSpeeds();
      this.showSummary();
    } else if (!f.finished) {
      this.endShown = false;
    }
    if ("selected" in f) {
      this.detail = f.selected;
      if (!f.selected) this.follow = false;
    }
    this.ui.updatePops(f);
    if (f.insights?.length) {
      this.lastInsight = f.insights[f.insights.length - 1].id;
      this.ui.pushInsights(f.insights);
    }
    if (!this._inspT || performance.now() - this._inspT > 400) {
      this._inspT = performance.now();
      this.ui.renderInspector();
    }
  }

  async reloadTerrain() {
    if (this._terrainBusy) return;
    this._terrainBusy = true;
    this._terrainDirty = false;
    this._terrainAt = performance.now();
    try {
      const t = await api("/api/terrain");
      this.w3.setTerrain(this.info, t, true);
      this.plantsAt = 0;
    } finally {
      this._terrainBusy = false;
    }
  }

  loop(now) {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    const time = now / 1000;
    if (this.info) {
      // Rebuilding the landscape is costly, so while a stroke is being painted
      // it happens at most every 0.6 s (painted spots show meanwhile).
      if (this._terrainDirty && !this._terrainBusy && (!this.painting || now - (this._terrainAt || 0) > 600)) this.reloadTerrain();
      if (this.running || this.skipping) {
        // Glide towards the latest state at the speed the simulation is really
        // going, staying about one frame behind so motion never stalls.
        const rate = Math.max(this.rate, 0.5);
        this.renderTick = Math.min(this.tCurr, this.renderTick + dt * rate);
        if (this.tCurr - this.renderTick > rate * 0.6) this.renderTick = this.tCurr - rate * 0.3;
        // At top speed fewer frames leave more of the computer's time for simulating.
        const every = this.skipping ? SKIP_FRAME_MS : this.speed >= 25 ? 2 * FRAME_MS : FRAME_MS;
        if (!this.busy && now - this.frameAt > every) this.fetchFrame();
      } else {
        this.renderTick = this.tCurr;
        if (!this.busy && now - this.frameAt > 1000) this.fetchFrame(); // keep the inspector fresh
      }
      this._interpolate();
      this.animals.update(this.rendered, this.info.acts, dt, time);
      this._moveCamera(dt);
      const tpd = this.info.clock.ticks_per_day;
      const hours = (((this.renderTick + tpd / 4) % tpd) / tpd) * 24;
      if (this.env) {
        this.w3.updateSky(this.env, hours, time);
        this.ui.updateClock(this.env, hours);
      }
      this.w3.updateParticles(dt);
      this._updateSelection();
      this._updateBubbles();
    }
    this.w3.render();
    requestAnimationFrame((t) => this.loop(t));
  }

  _interpolate() {
    const span = this.tCurr - this.tPrev;
    const t = span > 0 ? Math.min(1, Math.max(0, (this.renderTick - this.tPrev) / span)) : 1;
    this.rendered = this.curr.map((c) => {
      const p = this.prev.get(c.id);
      if (!p || Math.abs(p.x - c.x) + Math.abs(p.y - c.y) > 8) return c;
      return { ...c, x: p.x + (c.x - p.x) * t, y: p.y + (c.y - p.y) * t, h: lerpAngle(p.h, c.h, Math.min(1, t * 1.5)) };
    });
  }

  // ------------------------------------------------------------------ daily records

  // The ledger only gains a row once a simulated day, so it is fetched again only
  // when a day has passed (or when asked to).
  async pullLedger(force = false) {
    if (this._ledgerBusy || !this.info) return;
    const tick = this.frame?.tick ?? 0;
    if (!force && this.ledger && tick - this._ledgerTick < this.info.clock.ticks_per_day) return;
    this._ledgerBusy = true;
    try {
      this.ledger = await api("/api/ledger");
      this._ledgerTick = tick;
      this.ui.ledgerUpdated();
    } catch (err) {
      this.ui.toast(err.message, "error");
    } finally {
      this._ledgerBusy = false;
    }
  }

  // ------------------------------------------------------------------ history

  async pullHistory() {
    if (this._histBusy || !this.info) return;
    this._histBusy = true;
    try {
      const since = this.hist?.tick.length ? this.hist.tick[this.hist.tick.length - 1] : -1;
      const h = await api(`/api/history?since=${since}`);
      if (!h.tick.length) return;
      if (!this.hist) this.hist = h;
      else {
        const append = (dst, src) => {
          for (const k of Object.keys(src)) {
            if (Array.isArray(src[k])) for (const v of src[k]) dst[k].push(v);
            else append(dst[k], src[k]);
          }
        };
        append(this.hist, h);
      }
      this.ui.drawCharts();
    } catch (err) {
      this.ui.toast(err.message, "error");
    } finally {
      this._histBusy = false;
    }
  }

  downloadCsv() {
    const h = this.hist;
    if (!h || !h.tick.length) return;
    const header = ["tick", "plant_biomass", "voles"], cols = [h.tick, h.plants, h.voles || []];
    for (const sp of this.info.species) {
      const row = h.species[sp.name];
      header.push(`${sp.name}_count`, `${sp.name}_juveniles`, `${sp.name}_vigor`, `${sp.name}_load`, `${sp.name}_marker_he`, `${sp.name}_marker_ho`, `${sp.name}_instinct_agreement`);
      cols.push(row.count, row.juveniles, row.vigor, row.load, row.diversity.he, row.diversity.ho, row.instinct || []);
      for (const [t, s] of Object.entries(row.traits)) {
        header.push(`${sp.name}_${t}_mean`, `${sp.name}_${t}_sd`, `${sp.name}_${t}_breeding_value`);
        cols.push(s.mean, s.sd, s.genetic);
      }
      for (const g of sp.genes) {
        g.alleles.forEach((a, k) => {
          header.push(`${sp.name}_${g.symbol}_${a}_freq`);
          cols.push(row.genes[g.symbol].freq.map((f) => (f ? f[k] : "")));
        });
      }
    }
    const lines = [header.join(",")];
    for (let i = 0; i < h.tick.length; i++) lines.push(cols.map((c) => c[i] ?? "").join(","));
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `forager-seed${this.info.seed}-t${h.tick.at(-1)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ------------------------------------------------------------------ selection

  _initSelectionVisuals() {
    const flat = (inner, outer, color, opacity) => {
      const m = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 64),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2;
      m.renderOrder = 5;
      m.visible = false;
      this.w3.scene.add(m);
      return m;
    };
    this.selRing = flat(0.55, 0.72, 0xffd23f, 0.95);
    this.visionRing = flat(0.985, 1.0, 0xffffff, 0.6);
    this.homeRing = flat(0.96, 1.0, 0xffe7a0, 0.45);
    const lineMat = (color, opacity) => new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false });
    this.trailLine = new THREE.Line(new THREE.BufferGeometry(), lineMat(0xffffff, 0.8));
    this.targetLine = new THREE.Line(new THREE.BufferGeometry(), lineMat(0xffd23f, 0.95));
    this.trailLine.renderOrder = this.targetLine.renderOrder = 6;
    // The selected animal's memory map: remembered food as glowing green tiles,
    // remembered danger as red rings on top (so the two never blend together).
    this.showMemory = false;
    const memoryLayer = (geometry, material, order) => {
      const mesh = new THREE.InstancedMesh(geometry.rotateX(-Math.PI / 2), material, 64);
      mesh.renderOrder = order;
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.visible = false;
      this.w3.scene.add(mesh);
      return mesh;
    };
    this.memoryFood = memoryLayer(new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending }), 4);
    this.memoryDanger = memoryLayer(new THREE.RingGeometry(0.26, 0.4, 40),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }), 5);
    // What it does that leaves a mark: buried food, its den, scent marks, its territory and its mate.
    const marker = (geometry, color, count) => {
      const mesh = new THREE.InstancedMesh(geometry, new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }), count);
      mesh.renderOrder = 7;
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.visible = false;
      this.w3.scene.add(mesh);
      return mesh;
    };
    this.memoryCaches = marker(new THREE.OctahedronGeometry(0.5), 0xffb347, 64);
    this.memoryLarder = marker(new THREE.OctahedronGeometry(0.6), 0xfff1a8, 64);
    this.memoryMarks = marker(new THREE.ConeGeometry(0.16, 0.5, 6), 0xffe14d, 32);
    this.memoryDen = new THREE.Mesh(new THREE.RingGeometry(0.7, 1.0, 32).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xff8c42, transparent: true, opacity: 0.95, depthTest: false, side: THREE.DoubleSide }));
    this.memoryTerritory = new THREE.LineLoop(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xff9f43, transparent: true, opacity: 0.8, depthTest: false }));
    this.memoryMate = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xff7eb6, transparent: true, opacity: 0.9, depthTest: false }));
    for (const o of [this.memoryDen, this.memoryTerritory, this.memoryMate]) {
      o.renderOrder = 7;
      o.frustumCulled = false;
      o.visible = false;
      this.w3.scene.add(o);
    }
    this.trailLine.frustumCulled = this.targetLine.frustumCulled = false;
    this.w3.scene.add(this.trailLine, this.targetLine);
  }

  async select(id) {
    this.breeding = false;
    if (id !== this.selected) {
      this.trail = [];
      this.follow = false;
    }
    this.selected = id;
    this.detail = null;
    if (id != null) {
      try {
        this.detail = await api(`/api/animal/${id}`);
      } catch (err) {
        this.ui.toast(err.message, "error");
      }
    }
    this.ui.renderInspector();
  }

  _updateSelection() {
    const a = this.selected != null ? this.rendered.find((r) => r.id === this.selected) : null;
    const d = this.detail;
    const on = !!(a && d && !d.dead);
    for (const o of [this.selRing, this.visionRing, this.homeRing, this.trailLine, this.targetLine]) o.visible = on;
    this._updateMemory(on ? d : null);
    if (!on) {
      this.ui.tag(null);
      return;
    }
    const W = this.w3.W, H = this.w3.H;
    const gy = this.w3.groundY(a.x, a.y);
    const hidden = this.info.acts[a.act] === "hide";
    this.selRing.position.set(a.x - W / 2, gy + 0.06, a.y - H / 2);
    const s = Math.max(0.8, a.size * 1.2);
    this.selRing.scale.set(s, s, s);
    this.visionRing.position.set(a.x - W / 2, gy + 0.08, a.y - H / 2);
    this.visionRing.scale.setScalar(a.vision);
    this.homeRing.position.set(d.home[0] - W / 2, this.w3.groundY(d.home[0], d.home[1]) + 0.07, d.home[1] - H / 2);
    this.homeRing.scale.setScalar(d.home_radius);

    const last = this.trail[this.trail.length - 1];
    if (!last || Math.hypot(last[0] - a.x, last[1] - a.y) > 0.25) {
      this.trail.push([a.x, a.y]);
      if (this.trail.length > 400) this.trail.shift();
    }
    const pts = this.trail.map(([x, y]) => new THREE.Vector3(x - W / 2, this.w3.groundY(x, y) + 0.12, y - H / 2));
    this.trailLine.geometry.setFromPoints(pts);
    if (d.target || d.route) {
      // Show the planned route round obstacles when there is one, else a straight line.
      const stops = d.route && d.route.length ? d.route : [d.target];
      this.targetLine.geometry.setFromPoints([
        new THREE.Vector3(a.x - W / 2, gy + 0.4, a.y - H / 2),
        ...stops.map(([x, y]) => new THREE.Vector3(x - W / 2, this.w3.groundY(x, y) + 0.4, y - H / 2)),
      ]);
    } else this.targetLine.visible = false;

    const head = new THREE.Vector3(a.x - W / 2, gy + (hidden ? 0.4 : 1.2 * a.size + 0.4), a.y - H / 2).project(this.w3.camera);
    if (head.z > 1) return this.ui.tag(null);
    const r = this.canvas.getBoundingClientRect();
    this.ui.tag({ x: r.left + ((head.x + 1) / 2) * r.width, y: r.top + ((1 - head.y) / 2) * r.height }, d);
  }

  _updateMemory(d) {
    const m = d?.memory;
    const on = !!(this.showMemory && m);
    this.memoryFood.visible = this.memoryDanger.visible = on;
    this._updateDoings(on ? d : null);
    if (!on || m === this._memShown) return;
    this._memShown = m;
    const W = this.w3.W, H = this.w3.H, size = m.region;
    const mat = new THREE.Matrix4(), color = new THREE.Color();
    const fill = (mesh, list, lift, colorOf, scaleOf) => {
      let n = 0;
      for (const [x, y, v] of list) {
        if (v < 0.05 || n >= mesh.instanceMatrix.count) continue;
        const cx = x + size / 2, cy = y + size / 2, s = size * scaleOf(v);
        mat.makeScale(s, 1, s).setPosition(cx - W / 2, this.w3.groundY(cx, cy) + lift, cy - H / 2);
        mesh.setMatrixAt(n, mat);
        mesh.setColorAt(n, colorOf(Math.min(1, v)));
        n++;
      }
      mesh.count = n;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    };
    // Food: additive green, brighter for better-remembered patches.
    const green = new THREE.Color(0x3fd16b), black = new THREE.Color(0x000000);
    fill(this.memoryFood, m.food, 0.15, (v) => color.copy(black).lerp(green, 0.25 + 0.75 * v), () => 0.94);
    // Danger: red rings, bigger for fresher scares.
    const red = new THREE.Color(0xff3b3b);
    fill(this.memoryDanger, m.danger, 0.2, () => color.copy(red), (v) => 0.5 + 0.5 * v);
  }

  // Caches, den, scent marks, territory and mate of the selected animal.
  _updateDoings(d) {
    const m = d?.memory;
    const layers = [this.memoryCaches, this.memoryLarder, this.memoryMarks, this.memoryDen, this.memoryTerritory, this.memoryMate];
    if (!m) {
      for (const o of layers) o.visible = false;
      return;
    }
    const W = this.w3.W, H = this.w3.H;
    const ground = (x, y, lift) => new THREE.Vector3(x - W / 2, this.w3.groundY(x, y) + lift, y - H / 2);
    const mat = new THREE.Matrix4();
    const bob = 0.15 * Math.sin(performance.now() / 300);
    const place = (mesh, list, lift) => {
      let n = 0;
      for (const [x, y] of list) {
        if (n >= mesh.instanceMatrix.count) break;
        mat.makeTranslation(x - W / 2, this.w3.groundY(x, y) + lift + bob, y - H / 2);
        mesh.setMatrixAt(n++, mat);
      }
      mesh.count = n;
      mesh.visible = n > 0;
      mesh.instanceMatrix.needsUpdate = true;
    };
    const caches = m.caches || [];
    place(this.memoryCaches, caches.filter((c) => !c[3]), 0.9);
    place(this.memoryLarder, caches.filter((c) => c[3]), 1.1);
    place(this.memoryMarks, m.marks || [], 0.35);
    this.memoryDen.visible = !!m.den;
    if (m.den) this.memoryDen.position.copy(ground(m.den[0], m.den[1], 0.2));
    this.memoryTerritory.visible = !!m.territory;
    if (m.territory) {
      const [hx, hy, r] = m.territory;
      const pts = [];
      for (let k = 0; k < 64; k++) {
        const a = (k / 64) * Math.PI * 2;
        const x = Math.min(W - 0.5, Math.max(0.5, hx + Math.cos(a) * r)), y = Math.min(H - 0.5, Math.max(0.5, hy + Math.sin(a) * r));
        pts.push(ground(x, y, 0.3));
      }
      this.memoryTerritory.geometry.setFromPoints(pts);
    }
    const self = this.rendered.find((r) => r.id === this.selected);
    this.memoryMate.visible = !!(m.mate && self);
    if (m.mate && self) this.memoryMate.geometry.setFromPoints([ground(self.x, self.y, 0.8), ground(m.mate[0], m.mate[1], 0.8)]);
  }

  _updateBubbles() {
    if (!this.bubbles) return this.ui.bubbles([]);
    const cam = this.w3.camera;
    const r = this.canvas.getBoundingClientRect();
    const W = this.w3.W, H = this.w3.H;
    const target = this.w3.controls.target;
    const acts = this.info.acts;
    const dist = cam.position.distanceTo(target);
    if (dist > 95) return this.ui.bubbles([]);
    const near = [];
    for (const a of this.rendered) {
      const act = acts[a.act];
      if (act === "wander" || act === "hide" || a.id === this.selected) continue;
      const dx = a.x - W / 2 - target.x, dz = a.y - H / 2 - target.z;
      const d2 = dx * dx + dz * dz;
      if (d2 < (dist * 0.7) ** 2) near.push([d2, a]);
    }
    near.sort((p, q) => p[0] - q[0]);
    const v = new THREE.Vector3();
    const items = [];
    for (const [, a] of near.slice(0, 60)) {
      v.set(a.x - W / 2, this.w3.groundY(a.x, a.y) + 1.05 * a.size + 0.35, a.y - H / 2).project(cam);
      if (v.z > 1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05) continue;
      items.push({
        text: ACT_EMOJI[acts[a.act]] || "", x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height,
        scale: Math.max(0.55, Math.min(1.1, 30 / dist)),
      });
    }
    this.ui.bubbles(items);
  }

  toggleBubbles() {
    this.bubbles = !this.bubbles;
    this.ui.buildToolbar();
  }

  toggleFollow() {
    this.follow = !this.follow;
    this.ui.renderInspector();
  }

  // ------------------------------------------------------------------ camera

  _moveCamera(dt) {
    const c = this.w3.controls;
    const move = new THREE.Vector3();
    const fwd = new THREE.Vector3().subVectors(c.target, this.w3.camera.position).setY(0).normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0));
    if (this.keys.has("w") || this.keys.has("arrowup")) move.add(fwd);
    if (this.keys.has("s") || this.keys.has("arrowdown")) move.sub(fwd);
    if (this.keys.has("d") || this.keys.has("arrowright")) move.add(right);
    if (this.keys.has("a") || this.keys.has("arrowleft")) move.sub(right);
    if (move.lengthSq() > 0) {
      this.follow = false;
      move.normalize().multiplyScalar(dt * Math.max(10, this.w3.camera.position.distanceTo(c.target) * 0.9));
      c.target.add(move);
      this.w3.camera.position.add(move);
    }
    if (this.follow && this.selected != null) {
      const a = this.rendered.find((r) => r.id === this.selected);
      if (a) {
        const goal = new THREE.Vector3(a.x - this.w3.W / 2, this.w3.groundY(a.x, a.y), a.y - this.w3.H / 2);
        const delta = goal.sub(c.target).multiplyScalar(Math.min(1, dt * 4));
        c.target.add(delta);
        this.w3.camera.position.add(delta);
      }
    }
  }

  // ------------------------------------------------------------------ tools and input

  setTool(id) {
    this.tool = id;
    this.breeding = false;
    const brush = TOOLS.find((t) => t.id === id)?.brush || id === "spawn";
    // With a painting tool the left button paints; orbit moves to the right button.
    this.w3.controls.mouseButtons = brush
      ? { LEFT: -1, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE }
      : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    if (!brush) this.w3.showCursor(null);
    this.ui.buildToolbar();
    this.ui.buildToolOptions();
  }

  _bindInput() {
    let down = null;
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    this.canvas.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY, button: e.button };
      if (e.button === 0 && this.tool !== "select" && this.tool !== "spawn") {
        this.painting = true;
        this._paintAt(e, true);
      }
    });
    this.canvas.addEventListener("pointermove", (e) => {
      const tool = TOOLS.find((t) => t.id === this.tool);
      if (tool?.brush || this.tool === "spawn") {
        const p = this.w3.pickGround(e);
        this.w3.showCursor(p, this.tool === "spawn" ? 1.2 : this.brush, tool?.color || "#ffd23f");
        if (down && down.button === 0 && tool?.brush) this._paintAt(e, false);
      }
    });
    window.addEventListener("pointerup", (e) => {
      if (down && e.target === this.canvas && down.button === 0 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6) this._click(e);
      down = null;
      this.painting = false; // the stroke is over: the terrain is rebuilt on the next frame
    });
    this.canvas.addEventListener("pointerleave", () => this.w3.showCursor(null));

    window.addEventListener("keydown", (e) => {
      if (e.target.closest?.("input, select, textarea")) return;
      const k = e.key.toLowerCase();
      if (k === " ") { e.preventDefault(); this.togglePlay(); return; }
      if (k === "escape") { this.select(null); this.setTool("select"); return; }
      if (k === "b") { this.toggleBubbles(); return; }
      if (k === "f" && this.selected != null) { this.toggleFollow(); return; }
      const tool = TOOLS.find((t) => t.key === k);
      if (tool) { this.setTool(tool.id); return; }
      this.keys.add(k);
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.key.toLowerCase()));
    window.addEventListener("blur", () => this.keys.clear());

    $("#play").addEventListener("click", () => this.togglePlay());
    $("#btn-data").addEventListener("click", () => {
      $("#drawer").hidden = !$("#drawer").hidden;
      document.body.classList.toggle("drawer-open", !$("#drawer").hidden);
      this.ui.buildDrawer();
    });
    $("#drawer-close").addEventListener("click", () => {
      $("#drawer").hidden = true;
      document.body.classList.remove("drawer-open");
    });
    $("#csv").addEventListener("click", () => this.downloadCsv());
    $("#btn-insights").addEventListener("click", () => this.ui.toggleInsights());
    $("#btn-skip").addEventListener("click", (e) => {
      e.stopPropagation();
      this.ui.buildSkipMenu();
      $("#skip-menu").hidden = !$("#skip-menu").hidden;
    });
    $("#btn-events").addEventListener("click", (e) => {
      e.stopPropagation();
      $("#events-menu").hidden = !$("#events-menu").hidden;
    });
    document.addEventListener("click", (e) => {
      if (!e.target.closest("#events-menu")) $("#events-menu").hidden = true;
      if (!e.target.closest("#skip-menu")) $("#skip-menu").hidden = true;
    });
    $("#btn-world").addEventListener("click", () => { $("#world-modal").hidden = false; });
    $("#btn-repop").addEventListener("click", () => this.repopulate());
  }

  async _click(e) {
    if (this.tool === "spawn") {
      const p = this.w3.pickGround(e);
      if (!p) return;
      try {
        const r = await api("/api/tool", { tool: "spawn", x: p.x, y: p.y, species: this.spawn.species, preset: this.spawn.preset, count: this.spawn.count });
        if (!r.ids.length) this.ui.toast("There's no room for animals there.", "error");
        await this.requestFrame(0);
      } catch (err) {
        this.ui.toast(err.message, "error");
      }
      return;
    }
    if (this.tool !== "select") return;
    this.w3.raycaster.setFromCamera(this.w3.ndc(e), this.w3.camera);
    const id = this.animals.pick(this.w3.raycaster);
    if (this.breeding && id != null && id !== this.selected) {
      this.breeding = false;
      try {
        await api("/api/lab", { action: "breed", id: this.selected, partner: id });
        this.ui.toast("💞 They mated. Watch for a litter after the pregnancy.");
      } catch (err) {
        this.ui.toast(err.message, "error");
      }
      this.ui.renderInspector();
      return;
    }
    this.select(id);
  }

  async _paintAt(e, first) {
    const now = performance.now();
    if (!first && now - (this._paintT || 0) < 90) return;
    this._paintT = now;
    const p = this.w3.pickGround(e);
    if (!p) return;
    const tool = this.tool;
    const body = tool === "fertilise" ? { tool: "fertilise" }
      : tool === "remove" ? { tool: "remove" }
      : { tool: "paint", terrain: tool };
    if (body.tool === "paint") this.w3.paintMark(p.x, p.y, this.brush, TOOLS.find((t) => t.id === tool)?.color || "#ffffff");
    try {
      const r = await api("/api/tool", { ...body, x: p.x, y: p.y, radius: this.brush });
      if (r.version != null && r.version !== this.w3.version) this._terrainDirty = true;
      if (tool === "fertilise") this.plantsAt = 0;
      if (!this.running) this.requestFrame(0);
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async showSummary() {
    try {
      const [sum] = await Promise.all([api("/api/summary"), this.ui.refreshAIStatus()]);
      this.ui.showSummary(sum);
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async setEnding(on) {
    try {
      const r = await api("/api/run", { end_on_extinction: on });
      this.endOnExtinction = r.end_on_extinction;
      this.ui.toast(on ? "🏁 The run will stop when a species dies out." : "♾️ The run will keep going through extinctions.");
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async resume() {
    try {
      await api("/api/run", { resume: true });
      this.endOnExtinction = false;
      this.endShown = false;
      if (!this.running) this.togglePlay();
      this.ui.toast("▶ Carrying on. The run won't stop for extinctions now.");
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async explain() {
    try {
      this.ui.showReport(await api("/api/explain"));
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async trigger(kind, species) {
    $("#events-menu").hidden = true;
    try {
      await api("/api/tool", { tool: "event", event: kind, species });
      this.plantsAt = 0;
      this.ui.toast({ drought: "☀️ A drought begins.", bloom: "🌸 Everything bursts into growth!", cold: "❄️ A cold snap sweeps in.", plague: "🦠 Disease is spreading." }[kind]);
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async lab(action, extra = {}) {
    if (this.selected == null) return;
    try {
      const r = await api("/api/lab", { action, id: this.selected, ...extra });
      if (action === "clone") {
        this.ui.toast("🧬 Clone created. Same genome, new individual.");
        await this.requestFrame(0);
      } else if (action === "edit") {
        this.detail = r;
        this.ui.toast("🧪 Gene edited.");
        this.ui.renderInspector();
      }
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  startBreeding() {
    this.breeding = !this.breeding;
    if (this.breeding) {
      this.setTool("select");
      this.breeding = true;
      this.ui.toast("💞 Now click a partner of the opposite sex.");
    }
    this.ui.renderInspector();
  }

  async removeSelected() {
    const a = this.rendered.find((r) => r.id === this.selected);
    if (!a) return;
    await api("/api/tool", { tool: "remove", x: a.x, y: a.y, radius: 0.5 });
    await this.requestFrame(0);
  }

  async repopulate() {
    try {
      await api("/api/tool", { tool: "repopulate" });
      this.hist = null;
      this.ledger = null;
      this.ui.ai.answers = [];
      this.select(null);
      await this.requestFrame(0);
      this.prev = new Map(this.curr.map((a) => [a.id, a]));
      this.pullHistory();
      this.ui.toast("🐣 A fresh founding population has arrived.");
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }

  async newWorld(seed, settings) {
    try {
      const info = await api("/api/reset", { seed, settings });
      await this.load(info);
    } catch (err) {
      this.ui.toast(err.message, "error");
    }
  }
}

new App().start();
