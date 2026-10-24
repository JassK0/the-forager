// HUD, toolbar, inspector, data drawer and dialogs. All text goes in through
// textContent (via el()), never innerHTML.
import { C, alpha, compact, hoverIndex, lineChart, phaseChart, readColors, stackedChart } from "./charts.js";
import { getJSON, renderMarkdown, streamAI } from "./ai.js";

export const $ = (s) => document.querySelector(s);

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export const ACT_EMOJI = {
  wander: "🚶", rest: "💤", forage: "🌿", eat: "😋", "seek water": "💦", drink: "💧",
  "seek mate": "💕", flee: "😱", hunt: "🎯", hide: "🕳️", nest: "🪺", home: "🏠",
  freeze: "🫥", alert: "❗", groom: "🧼", play: "🎈", cache: "🦴", mark: "🚩", chase: "💢", visit: "🍼",
};
export const ACT_LABEL = {
  freeze: "frozen still", alert: "on alert", nest: "in the nest", visit: "nursing", cache: "burying food",
  mark: "scent-marking", chase: "chasing off a rival", groom: "grooming", play: "playing", home: "heading home",
};
// Decision options as the brain sees them ("eat" means hunting for predators).
const OPTION_INFO = {
  eat: ["😋", "eat"], drink: ["💧", "drink"], mate: ["💕", "find a mate"], rest: ["💤", "rest"],
  explore: ["🧭", "explore"], flee: ["😱", "flee"], freeze: ["🫥", "freeze"], hide: ["🕳️", "take cover"],
};
const INPUT_RANGE = { hunger: 1.5, threat: 3, "distance from home": 2, "mating drive": 2.5, "freezing tendency": 3 };
const EXCITE = "#e34948", INHIBIT = "#2a78d6";
const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) node.setAttribute(k, v);
  for (const c of children.flat()) {
    if (c == null) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
const SEASON_EMOJI = { spring: "🌱", summer: "☀️", autumn: "🍂", winter: "❄️" };
const CAUSE_EMOJI = { predation: "🦊", starvation: "🦴", dehydration: "🏜️", "old age": "👴", disease: "🦠", fighting: "⚔️", emigrated: "🧳", removed: "✖️" };
const title = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const plural = (s) => title(s) + (/(s|x|z|ch|sh)$/.test(s) ? "es" : "s");

export const TOOLS = [
  { id: "select", icon: "👆", label: "Select & inspect", key: "1" },
  { id: "spawn", icon: "🐣", label: "Add animals", key: "2" },
  { id: "fertilise", icon: "🌱", label: "Grow plants", key: "3", brush: true, color: "#7be07b" },
  { id: "water", icon: "💧", label: "Paint water", key: "4", brush: true, color: "#5ab0ff" },
  { id: "grass", icon: "🟩", label: "Paint grassland", key: "5", brush: true, color: "#9be36b" },
  { id: "tree", icon: "🌳", label: "Plant trees", key: "6", brush: true, color: "#3fa34d" },
  { id: "rock", icon: "🪨", label: "Raise rock", key: "7", brush: true, color: "#b0b0b8" },
  { id: "remove", icon: "🧹", label: "Remove animals", key: "8", brush: true, color: "#ff6b6b" },
];

export class UI {
  constructor(app) {
    this.app = app;
    this.tab = "ai";
    this.hover = null;
    this.ai = { status: null, answers: [], seq: 0 }; // the AI analyst's answers this session
    this.chartSel = { species: null, trait: "mass", gene: null };
    this.bubblePool = [];
    this.insightLog = [];
    this.report = null;
    this.unread = 0;
    this.popupQueue = [];
    this.prefs = { popups: true, tips: true, pause: false };
    readColors();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      readColors();
      this.drawCharts();
    });
  }

  // ------------------------------------------------------------------ top bar

  build(info) {
    this.info = info;
    this.ai.answers = [];
    this.chartSel.species = info.species[0].name;
    this.chartSel.gene = info.species[0].genes[0].symbol;
    this.buildSpeeds();
    this.showMode();
    this.buildToolbar();
    this.buildToolOptions();
    this.buildEventsMenu();
    this.buildWorldForm();
    this.buildDrawer();
    this.renderInspector();
  }

  buildSkipMenu() {
    const year = this.info.clock.days_per_year;
    const options = [[1, "1 day"], [year / 4, "1 season"], [year, "1 year"], [5 * year, "5 years"], [10 * year, "10 years"], [25 * year, "25 years"]];
    $("#skip-menu").replaceChildren(
      el("div", { class: "menu-head" }, "⏩ Skip ahead"),
      el("p", { class: "muted small menu-note" }, "Simulates every tick in full, as fast as this computer can, then carries on as before. The results are exactly the same as watching it play. A run that is set to end on an extinction still stops there."),
      ...options.map(([days, label]) => el("button", { class: "menu-item", onclick: () => this.app.skip(days) },
        el("strong", {}, label), el("span", {}, `${days} simulated day${days > 1 ? "s" : ""}`))));
  }

  updateSimClock(c) {
    const chip = $("#actual-speed");
    const fmt = (x) => (x >= 10 ? Math.round(x) : x.toFixed(1));
    if (!c || !c.running || !c.achieved) {
      chip.hidden = true;
    } else {
      chip.hidden = false;
      const slow = !c.skip && c.speed != null && c.achieved < 0.8 * c.speed;
      chip.textContent = `${c.skip ? "⏩ " : ""}${fmt(c.achieved)}× now`;
      chip.classList.toggle("slow", slow);
      chip.title = slow
        ? `You asked for ${fmt(c.speed)}×, but with this many animals the computer can simulate about ${fmt(c.achieved)}× real time. Nothing is skipped: every tick is still computed in full.`
        : "How fast the simulation is actually running (1× = 4 ticks a second)";
    }
    const panel = $("#skip-panel");
    if (!c?.skip) {
      panel.hidden = true;
      return;
    }
    const tpd = this.info.clock.ticks_per_day, year = tpd * this.info.clock.days_per_year;
    const { from, to } = c.skip;
    const done = Math.max(0, Math.min(1, (c.tick - from) / Math.max(1, to - from)));
    const ticksPerSecond = c.achieved * 4;
    const left = ticksPerSecond > 0 ? (to - c.tick) / ticksPerSecond : null;
    const eta = left == null ? "estimating…" : left < 60 ? `about ${Math.max(1, Math.round(left))} s left` : `about ${Math.round(left / 60)} min left`;
    const day = (t) => Math.floor((t + tpd / 4) / tpd) + 1;
    const counts = this.app.frame?.counts || [];
    panel.hidden = false;
    panel.replaceChildren(
      el("div", { class: "skip-top" }, el("strong", {}, "⏩ Skipping ahead"),
        el("span", { class: "muted small" }, `day ${day(c.tick)} of ${day(to)} · year ${Math.floor(c.tick / year) + 1}`),
        el("button", { class: "pill small", onclick: () => this.app.cancelSkip() }, "Stop")),
      el("div", { class: "skip-bar" }, el("div", { style: `width:${(done * 100).toFixed(1)}%` })),
      el("div", { class: "skip-bottom muted small" },
        el("span", {}, `${Math.round(done * 100)}% · ${fmt(c.achieved)}× real time · ${eta}`),
        el("span", {}, this.info.species.map((s, i) => `${s.icon} ${counts[i] ?? "–"}`).join("  "))));
  }

  showMode() {
    const brand = document.querySelector(".brand");
    brand.querySelectorAll(".mode").forEach((m) => m.remove());
    const s = this.info.settings;
    const cl = this.info.climate;
    if (cl && cl.name !== "temperate") brand.append(el("span", { class: "mode climate", title: `Climate: ${cl.text}` }, `${cl.icon} ${cl.label}`));
    if (s.super_animals) {
      brand.append(el("span", { class: "mode", title: `Super animals mode: evolution runs ${s.evolution_speed}× faster (not realistic)` },
        `🦸 SUPER ${s.evolution_speed}×`));
    }
  }

  buildSpeeds() {
    const box = $("#speeds");
    box.replaceChildren(...this.app.speeds.map(([v, label]) =>
      el("button", { class: `speed${v === this.app.speed ? " on" : ""}`, onclick: () => this.app.setSpeed(v), title: `${label} speed` }, label)));
    $("#play").textContent = this.app.running ? "⏸" : "▶";
  }

  updateClock(env, hours) {
    const h = Math.floor(hours), m = Math.floor((hours - h) * 60);
    $("#clock-time").textContent = `${String(h).padStart(2, "0")}:${String(m - (m % 5)).padStart(2, "0")}`;
    const night = hours < env.sunrise || hours > env.sunset;
    $("#clock-icon").textContent = night ? "🌙" : hours < env.sunrise + 1.2 || hours > env.sunset - 1.2 ? "🌅" : "☀️";
    $("#clock-date").textContent = `Day ${env.day} · ${SEASON_EMOJI[env.season]} ${title(env.season)} · Year ${env.year}`;
  }

  updatePops(frame) {
    const info = this.info;
    const chips = info.species.map((sp, i) =>
      el("div", { class: "chip", title: `${frame.counts[i]} ${plural(sp.name).toLowerCase()} · ${frame.births[i].toLocaleString()} born` },
        el("span", { class: "chip-icon" }, sp.icon), el("strong", {}, frame.counts[i].toLocaleString())));
    const cover = Math.round((100 * frame.plant_total) / Math.max(1, info.plant_capacity_total));
    chips.push(el("div", { class: "chip", title: "Plant biomass as a share of carrying capacity" },
      el("span", { class: "chip-icon" }, "🌿"), el("strong", {}, `${cover}%`)));
    $("#pops").replaceChildren(...chips);

    const fx = frame.effects.map((e) => {
      const sp = e.species != null ? ` (${plural(info.species[e.species].name).toLowerCase()})` : "";
      const label = { drought: "☀️ Drought", cold: "❄️ Cold snap", plague: "🦠 Plague" }[e.kind] || e.kind;
      return el("div", { class: "effect" }, `${label}${sp} · ${Math.ceil(e.left / info.clock.ticks_per_day * 24)}h left`);
    });
    $("#effects").replaceChildren(...fx);
  }

  // ------------------------------------------------------------------ tools

  buildToolbar() {
    const bar = $("#toolbar");
    bar.replaceChildren(...TOOLS.map((t) =>
      el("button", {
        class: `tool${this.app.tool === t.id ? " on" : ""}`, title: `${t.label} (${t.key})`,
        "aria-label": t.label, onclick: () => this.app.setTool(t.id),
      }, el("span", {}, t.icon))));
    bar.append(el("div", { class: "tool-sep" }),
      el("button", { class: `tool${this.app.bubbles ? " on" : ""}`, title: "Thought bubbles (B)", onclick: () => this.app.toggleBubbles() }, el("span", {}, "💭")));
  }

  buildToolOptions() {
    const box = $("#tool-options");
    const tool = TOOLS.find((t) => t.id === this.app.tool);
    box.replaceChildren();
    if (!tool || tool.id === "select") {
      box.hidden = true;
      this.setHint("Click an animal to inspect it · drag to orbit · right-drag to pan · scroll to zoom · WASD to move");
      return;
    }
    box.hidden = false;
    box.append(el("div", { class: "opt-title" }, `${tool.icon} ${tool.label}`));
    if (tool.id === "spawn") {
      const s = this.app.spawn;
      const species = el("div", { class: "seg" }, this.info.species.map((sp) =>
        el("button", { class: s.species === sp.name ? "on" : "", onclick: () => { s.species = sp.name; s.preset = ""; this.buildToolOptions(); } }, `${sp.icon} ${title(sp.name)}`)));
      const sp = this.info.species.find((x) => x.name === s.species);
      const preset = el("select", { onchange: (e) => { s.preset = e.target.value; } },
        el("option", { value: "", selected: !s.preset }, "Natural mix (random genes)"),
        sp.presets.map((p) => el("option", { value: p, selected: s.preset === p }, p)));
      const count = el("input", { type: "range", min: 1, max: 12, value: s.count, oninput: (e) => { s.count = +e.target.value; countLabel.textContent = s.count; } });
      const countLabel = el("strong", {}, s.count);
      box.append(species, el("label", { class: "opt" }, "Coat genotype", preset), el("label", { class: "opt" }, "How many ", countLabel, count));
      this.setHint("Click the ground to release animals · right-drag to orbit");
    } else {
      const size = el("input", { type: "range", min: 1, max: 12, step: 0.5, value: this.app.brush, oninput: (e) => { this.app.brush = +e.target.value; sizeLabel.textContent = this.app.brush; } });
      const sizeLabel = el("strong", {}, this.app.brush);
      box.append(el("label", { class: "opt" }, "Brush size ", sizeLabel, size));
      this.setHint("Click or drag on the ground to paint · right-drag to orbit · scroll to zoom");
    }
  }

  setHint(text) {
    $("#hint").textContent = text;
  }

  buildEventsMenu() {
    const menu = $("#events-menu");
    const items = [];
    for (const [kind, description] of Object.entries(this.info.events)) {
      const label = { drought: "☀️ Drought", bloom: "🌸 Bloom", cold: "❄️ Cold snap", plague: "🦠 Plague" }[kind];
      if (kind === "plague") {
        for (const sp of this.info.species) {
          items.push(el("button", { class: "menu-item", onclick: () => this.app.trigger(kind, sp.name) },
            el("strong", {}, `${label}: ${plural(sp.name).toLowerCase()}`), el("span", {}, description)));
        }
      } else {
        items.push(el("button", { class: "menu-item", onclick: () => this.app.trigger(kind) }, el("strong", {}, label), el("span", {}, description)));
      }
    }
    menu.replaceChildren(...items);
  }

  // ------------------------------------------------------------------ world dialog

  buildWorldForm() {
    const s = this.info.settings;
    const form = $("#world-form");
    const num = (name, label, value, min, max, step, note) => el("label", { class: "field" },
      el("span", {}, label, note ? el("small", {}, note) : null),
      el("input", { type: "number", name, value: +(+value).toFixed(5), min, max, step }));
    form.replaceChildren(
      el("h2", {}, "🗺️ New world"),
      el("p", { class: "muted" }, "Generate a new landscape and founding population. The same seed and settings always give the same world."),
      el("div", { class: "fields" },
        el("label", { class: "field" }, el("span", {}, "Seed"),
          el("span", { class: "seed" }, el("input", { type: "number", name: "seed", value: this.info.seed, min: 0, step: 1 }),
            el("button", { type: "button", class: "pill", onclick: () => { form.seed.value = Math.floor(Math.random() * 1e6); } }, "🎲"))),
        el("label", { class: "field map-pick" },
          el("span", {}, "🗺️ Map", el("small", {}, "premade sizes and landscapes; you can still change anything below")),
          el("select", { name: "map", onchange: (e) => this.applyMapPreset(form, e.target.value) },
            Object.entries(MAPS).map(([id, m]) => el("option", { value: id, selected: id === this.mapPresetOf(s) }, m.label)))),
        el("label", { class: "field map-pick" },
          el("span", {}, "🌍 Climate", el("small", {}, "temperature, snow, dryness and the landscape that goes with them")),
          el("select", { name: "climate", onchange: (e) => this.applyClimate(form, e.target.value) },
            Object.entries(this.info.climates).map(([id, c]) => el("option", { value: id, selected: id === (s.climate || "temperate") }, `${c.icon} ${c.label}`))),
          el("small", { class: "speed-note climate-note" }, this.info.climates[s.climate || "temperate"].text)),
        num("width", "Width (cells)", s.width ?? 160, 40, 240, 1),
        num("height", "Height (cells)", s.height ?? 100, 30, 160, 1),
        num("water_fraction", "Water (%)", s.water_fraction * 100, 3, 50, 1),
        num("forest_fraction", "Forest (%)", s.forest_fraction * 100, 0, 50, 1),
        num("rock_fraction", "Mountains (%)", s.rock_fraction * 100, 0, 30, 1),
        num("plant_growth", "Plant growth rate", s.plant_growth, 0.0005, 0.05, 0.0005, "logistic r"),
        num("mutation", "Mutation rate", s.mutation, 0, 20, 0.5, "× default"),
        num("immigration", "Immigration", s.immigration, 0, 0.01, 0.0001, "per species per tick"),
        ...this.info.species.map((sp) => num(`initial.${sp.name}`, `${sp.icon} Starting ${plural(sp.name).toLowerCase()}`, s.initial[sp.name], 0, 2000, 1)),
        el("label", { class: "field" }, el("span", {}, "🧠 Decision-making", el("small", {}, "how animals choose what to do")),
          el("select", { name: "brains" },
            el("option", { value: "neural", selected: s.brains !== "rules" }, "Evolving neural networks"),
            el("option", { value: "rules", selected: s.brains === "rules" }, "Fixed instincts (rules)"))),
        el("label", { class: "field check" }, el("input", { type: "checkbox", name: "learning", checked: s.learning !== false }), el("span", {}, "🎓 Animals learn from experience during their lives (with neural networks)")),
        el("label", { class: "field check" }, el("input", { type: "checkbox", name: "seasons", checked: s.seasons }), el("span", {}, "Seasons (winter, breeding seasons)")),
        el("label", { class: "field check" }, el("input", { type: "checkbox", name: "emigration", checked: s.emigration !== false }),
          el("span", {}, "🚪 Emigration: foxes that can't find a territory eventually leave the area. Off: they stay as floaters and keep trying (a closed population).")),
        el("label", { class: "field check" }, el("input", {
          type: "checkbox", name: "super_animals", checked: !!s.super_animals,
          onchange: (e) => { form.querySelector(".super-speed").hidden = !e.target.checked; },
        }), el("span", {}, "🦸 Super animals (just for fun, not realistic): much faster evolution, traits with no natural ceiling and quicker learners. The strongest get a 👑.")),
        el("div", { class: "super-speed", hidden: !s.super_animals },
          this.superSlider("evolution_speed", "🧬 Evolution speed", s.evolution_speed ?? 6, 1, 20, 0.5, (v) => this.superSpeedNote(v)),
          this.superSlider("learning_speed", "🧠 Brain learning speed", s.learning_speed ?? 2, 1, 10, 0.5,
            (v) => `${v}×: brains rewire ${v}× as fast from rewards and punishments`),
          this.superSlider("inherit_learning", "🎓 Know-how passed to young", Math.round(100 * (s.inherit_learning ?? 0.5)), 0, 80, 5,
            (v) => (v ? `${v}%: young start life with ${v}% of their parents' learned brain wiring, hunting skill and wariness, so know-how builds up over generations`
              : "0%: every animal starts from its genes alone, as in real life"))),
        el("label", { class: "field check" }, el("input", { type: "checkbox", name: "end_on_extinction", checked: s.end_on_extinction }), el("span", {}, "🏁 End the run (with a summary) when a species dies out"))),
      el("div", { class: "modal-actions" },
        el("button", { type: "button", class: "pill", onclick: () => { $("#world-modal").hidden = true; } }, "Cancel"),
        el("button", { type: "submit", class: "pill accent" }, "Generate world")));
    form.onsubmit = (e) => {
      e.preventDefault();
      const d = new FormData(form);
      const settings = {
        water_fraction: +d.get("water_fraction") / 100,
        forest_fraction: +d.get("forest_fraction") / 100,
        rock_fraction: +d.get("rock_fraction") / 100,
        plant_growth: +d.get("plant_growth"),
        mutation: +d.get("mutation"),
        immigration: +d.get("immigration"),
        seasons: d.get("seasons") === "on",
        end_on_extinction: d.get("end_on_extinction") === "on",
        brains: d.get("brains"),
        learning: d.get("learning") === "on",
        super_animals: d.get("super_animals") === "on",
        evolution_speed: +d.get("evolution_speed"),
        learning_speed: +d.get("learning_speed"),
        inherit_learning: +d.get("inherit_learning") / 100,
        width: +d.get("width"),
        height: +d.get("height"),
        emigration: d.get("emigration") === "on",
        climate: d.get("climate"),
        initial: Object.fromEntries(this.info.species.map((sp) => [sp.name, +d.get(`initial.${sp.name}`)])),
      };
      $("#world-modal").hidden = true;
      this.app.newWorld(+d.get("seed"), settings);
    };
  }

  // What an evolution speed does (mirrors super_spread in engine/config.py).
  superSpeedNote(speed) {
    const spread = Math.min(2, 1 + 0.1 * (speed - 1));
    return `${speed}×: mutations ${speed}× as often and ${spread.toFixed(2)}× the variation between parents and young`;
  }

  superSlider(name, label, value, min, max, step, note) {
    const out = el("small", { class: "speed-note" }, note(+value));
    return el("label", { class: "field super-slider" }, el("span", {}, label),
      el("input", { type: "range", name, min, max, step, value, oninput: (e) => { out.textContent = note(+e.target.value); } }), out);
  }

  // A climate brings its own typical landscape, and starting numbers scaled to
  // how much food the land can grow there (on this map's area).
  applyClimate(form, id) {
    const c = this.info.climates[id];
    if (!c) return;
    form.water_fraction.value = c.terrain.water;
    form.forest_fraction.value = c.terrain.forest;
    form.rock_fraction.value = c.terrain.rock;
    const area = (+form.width.value * +form.height.value) / (160 * 100);
    const set = (name, n) => { const input = form.querySelector(`[name="initial.${name}"]`); if (input) input.value = n; };
    set("rabbit", Math.max(8, Math.round(160 * area * c.animals)));
    set("fox", Math.max(2, Math.round(14 * area * c.animals)));
    form.querySelector(".climate-note").textContent = c.text;
  }

  // Which premade map the current settings match ("custom" if none).
  mapPresetOf(s) {
    const hit = Object.entries(MAPS).find(([id, m]) => id !== "custom" && m.width === s.width && m.height === s.height
      && Math.abs(m.water - s.water_fraction * 100) < 0.5 && Math.abs(m.forest - s.forest_fraction * 100) < 0.5
      && Math.abs(m.rock - s.rock_fraction * 100) < 0.5);
    return hit ? hit[0] : "custom";
  }

  applyMapPreset(form, id) {
    const m = MAPS[id];
    if (!m || id === "custom") return;
    form.width.value = m.width;
    form.height.value = m.height;
    form.water_fraction.value = m.water;
    form.forest_fraction.value = m.forest;
    form.rock_fraction.value = m.rock;
    form.immigration.value = m.immigration;
    if (m.keepGoing) form.end_on_extinction.checked = false;
    for (const [name, n] of Object.entries(m.initial)) {
      const input = form.querySelector(`[name="initial.${name}"]`);
      if (input) input.value = n;
    }
  }

  // ------------------------------------------------------------------ inspector

  renderInspector() {
    const box = $("#inspector");
    const d = this.app.detail;
    if (this.app.selected == null || !d) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const sp = this.info.species.find((s) => s.name === d.species);
    const close = el("button", { class: "round small", "aria-label": "Close", onclick: () => this.app.select(null) }, "✕");
    if (d.dead) {
      box.replaceChildren(el("div", { class: "insp-head" }, el("div", { class: "avatar" }, sp.icon),
        el("div", {}, el("h2", {}, d.name), el("div", { class: "muted" }, `${title(d.species)} · died`)), close),
        el("p", { class: "dead" }, `${CAUSE_EMOJI[d.cause] || "✝"} Died of ${d.cause} at ${this.days(d.age)}.`));
      return;
    }
    const coat = sp.coats.find((c) => c.name === d.coat) || sp.coats[0];
    const stage = !d.adult ? (d.dependent ? (d.in_nest ? "newborn, hidden in the nest" : "young, near the nest") : "juvenile")
      : d.role === "breeder" ? `adult · holds territory${d.helpers ? ` · ${d.helpers} helper${d.helpers > 1 ? "s" : ""}` : ""}`
        : d.role === "helper" ? "adult · helper in her mother's family"
          : d.role === "natal" ? "grown up · with its family until autumn"
            : d.role === "floater" ? "adult · no territory (floater)" : "adult";
    const head = el("div", { class: "insp-head" },
      el("div", { class: "avatar", style: `background:${coat.colors.body}` }, sp.icon),
      el("div", { class: "insp-title" },
        el("h2", {}, d.name, " ", el("span", { class: d.sex === "female" ? "sex f" : "sex m" }, d.sex === "female" ? "♀" : "♂")),
        el("div", { class: "muted" }, `${d.coat} ${d.species}`)),
      close);
    const status = el("div", { class: "status" },
      el("span", { class: "badge" }, `${ACT_EMOJI[d.act] || ""} ${ACT_LABEL[d.act] || d.act}`),
      el("span", { class: "badge soft" }, stage),
      d.pregnant ? el("span", { class: "badge pink" }, `🤰 due in ${this.hoursOf(d.gestation_left)}`) : null,
      d.litter ? el("span", { class: "badge pink" }, `🪺 ${d.litter} young · next feed in ${this.hoursOf(d.next_nurse || 0)}`) : null,
      d.fear > 0.3 ? el("span", { class: "badge soft" }, "😰 nervous after a scare") : null,
      d.alpha ? el("span", { class: "badge gold", title: "One of the three strongest of its species alive (Super animals mode)" },
        `👑 alpha #${d.alpha}${d.power ? ` · ${d.power}× founders' prowess` : ""}`) : null,
      d.injury > 0.05 ? el("span", { class: "badge pink", title: "Injured: slower, a worse hunter and a less attractive mate until it heals" },
        `🩹 injured ${Math.round(d.injury * 100)}%${d.lame ? " · permanent limp" : ""}`) : null,
      d.sight != null && d.sight < 1 && !d.adult ? el("span", { class: "badge soft" }, d.sight === 0 ? "🙈 eyes still closed" : `👀 eyes opening (${Math.round(d.sight * 100)}%)`) : null,
      d.caches ? el("span", { class: "badge soft" }, `🦴 ${d.caches} food cache${d.caches > 1 ? "s" : ""}`) : null,
      d.winter_white ? el("span", { class: "badge soft", title: "Carries the winter-white gene: moults to a white coat for winter, timed by day length" },
        "❄️ turns white in winter") : null);
    const place = d.place ? this.placeLine(d.place) : null;
    const bars = el("div", { class: "bars" },
      this.meter("Energy", d.energy, "#f5a524"), this.meter("Water", d.water, "#3f97dc"), this.meter("Stamina", d.stamina, "#7bc950"),
      d.urge > 0.02 ? this.meter("Urge to mate", d.urge, "#e85d9a") : null);
    if (place) bars.append(place);
    const facts = el("div", { class: "facts" },
      this.fact("Age", `${this.days(d.age)}`), this.fact("Life expectancy", this.days(d.lifespan)),
      this.fact("Generation", d.generation), this.fact("Mass", `${d.mass} kg`));

    const tabs = ["traits", "brain", "genes", "genome", "family", "lab"];
    this.inspTab = this.inspTab || "traits";
    const tabBar = el("div", { class: "insp-tabs" }, tabs.map((t) =>
      el("button", { class: this.inspTab === t ? "on" : "", onclick: () => { this.inspTab = t; this.renderInspector(); } }, title(t))));
    const body = el("div", { class: "insp-body" }, this[`tab_${this.inspTab}`](d, sp));
    const scroll = box.querySelector(".insp-body")?.scrollTop || 0;
    box.replaceChildren(head, status, bars, facts, tabBar, body);
    box.querySelector(".insp-body").scrollTop = scroll;
    this.drawInspectorCharts();
  }

  days(ticks) {
    const tpd = this.info.clock.ticks_per_day;
    const d = ticks / tpd;
    return d < 1 ? `${Math.round(d * 24)} h` : `${d.toFixed(d < 10 ? 1 : 0)} days`;
  }

  hoursOf(ticks) {
    return this.days(ticks);
  }

  // Where the animal is living, and how well its coat hides it there.
  placeLine(p) {
    const t = p.temperature;
    const feel = t < -1.2 ? "bitterly cold" : t < -0.5 ? "freezing" : t < 0 ? "cold" : t < 0.6 ? "mild" : t < 1.1 ? "warm" : "scorching";
    const gap = p.camouflage ?? Math.abs(p.coat - p.ground);
    const camo = gap < 0.12 ? "well hidden" : gap < 0.28 ? "partly hidden" : "stands out";
    const snow = p.snow > 0.6 ? " · deep snow" : p.snow > 0.15 ? " · some snow" : "";
    return el("div", { class: "place muted small", title: `Coat lightness ${p.coat} against ground ${p.ground} (0 black, 1 white); colour difference as a fox or rabbit sees it: ${gap} (0 = perfect match). Camouflage works while it keeps still.` },
      `📍 ${title(p.biome)} · ${feel}${snow} · coat ${camo} against the ground`);
  }

  meter(label, v, color) {
    return el("div", { class: "meter" }, el("span", {}, label),
      el("div", { class: "track" }, el("div", { class: v < 0.25 ? "fill low" : "fill", style: `width:${Math.max(0, Math.min(1, v)) * 100}%;background:${v < 0.25 ? "" : color}` })),
      el("span", { class: "num" }, `${Math.round(v * 100)}%`));
  }

  fact(label, value) {
    return el("div", { class: "fact" }, el("span", {}, label), el("strong", {}, value));
  }

  tab_traits(d, sp) {
    const rows = Object.entries(d.traits).map(([k, t]) => {
      const meta = sp.traits[k];
      const rel = t.value / meta.mean - 1;
      const g = t.genetic / meta.mean - 1;
      const pct = (x) => `${x >= 0 ? "+" : ""}${Math.round(x * 100)}%`;
      return el("div", { class: "trait" },
        el("div", { class: "trait-top" }, el("span", {}, meta.label), el("strong", {}, `${compact(t.value)} ${meta.unit}`)),
        el("div", { class: "trait-bar" },
          el("div", { class: "mid" }),
          el("div", { class: "pheno", style: `left:${50 + Math.max(-48, Math.min(48, rel * 150))}%`, title: `Phenotype ${pct(rel)} vs species mean` }),
          el("div", { class: "geno", style: `left:${50 + Math.max(-48, Math.min(48, g * 150))}%`, title: `Breeding value ${pct(g)} vs species mean` })),
        el("div", { class: "trait-sub muted" }, `phenotype ${pct(rel)} · genes ${pct(g)} · h² ${meta.heritability}`));
    });
    return [
      el("p", { class: "muted small" }, "● phenotype (what the animal is) ◆ breeding value (what its genes predict, and what it passes on). The gap is developmental noise."),
      ...rows,
      el("div", { class: "facts" },
        this.fact("Genetic vigour", `${Math.round(d.vigor * 100)}%`),
        this.fact("Camouflage", d.visibility < 1 ? "well hidden" : d.visibility < 1.25 ? "average" : "conspicuous"),
        this.fact("Upkeep", `${d.upkeep} /tick`),
        d.kills ? this.fact("Kills", d.kills) : null),
    ];
  }

  optionInfo(name, sp) {
    return name === "eat" && sp.diet !== "plants" ? ["🎯", "hunt"] : OPTION_INFO[name] || ["", name];
  }

  tab_brain(d, sp) {
    const b = d.brain, m = d.memory, meta = this.info.brain;
    this._charts = [];
    if (!b) {
      return [el("p", { class: "muted small" }, d.dependent
        ? "Still dependent on its mother: it stays in or near the nest and makes no choices of its own yet."
        : "It hasn't made a decision yet.")];
    }
    const neural = b.mode === "neural";
    const learns = neural && b.learning;
    const opt = (k) => this.optionInfo(meta.options[k], sp);
    const [ce, cl] = opt(b.choice);
    const differs = neural && b.instinct != null && b.instinct !== b.choice;
    const out = [
      el("p", { class: "muted small" }, !neural
        ? "Instinct mode: choices follow fixed hand-written rules. Brain genes are still inherited but not used (switch to evolving brains under 🗺️ New world)."
        : learns
          ? "A small neural network makes its choices: 15 senses feed 8 hidden neurons, which score 8 options. Its genes set the wiring it was born with; since then it has been rewiring itself from experience, strengthening connections that led to good outcomes (food when hungry, water when thirsty, mating) and weakening those that led to bad ones (a predator's lunge, a lost fight)."
          : "A small neural network makes its choices, wired by its genes. Lifetime learning is switched off in this world, so its wiring stays as it was born."),
      el("div", { class: "decision" },
        el("span", { class: "badge" }, `${ce} wants to ${cl}`),
        differs ? el("span", { class: "badge soft", title: "What the founders' hand-written instincts would choose in the same situation" },
          `instinct would ${opt(b.instinct)[1]}`) : neural ? el("span", { class: "muted small" }, "same as instinct") : null),
    ];

    // Why: the senses that pushed this choice up or down most.
    if (neural && b.why) {
      const names = meta.inputs;
      const top = b.why.map((v, i) => [v, i]).filter(([, i]) => names[i] !== "bias" && Math.abs(b.why[i]) > 0.005)
        .sort((a, c) => Math.abs(c[0]) - Math.abs(a[0])).slice(0, 4);
      if (top.length) {
        out.push(el("div", { class: "why" }, el("span", { class: "muted small" }, "Because of: "),
          top.map(([v, i]) => el("span", { class: `chip-why ${v >= 0 ? "up" : "down"}`, title: `${names[i]} ${v >= 0 ? "raised" : "lowered"} the score for “${cl}” by ${Math.abs(v).toFixed(2)}` },
            `${v >= 0 ? "▲" : "▼"} ${names[i]}`))));
      }
    }

    if (learns && b.change != null) {
      const born = [...b.born_w1.flat(), ...b.born_w2.flat()];
      const scale = born.reduce((s, v) => s + Math.abs(v), 0) / born.length || 1;
      const rewired = (100 * b.change) / scale;
      out.push(el("div", { class: "facts" },
        this.fact("Rewired since birth", `${rewired < 10 ? rewired.toFixed(1) : Math.round(rewired)}%`),
        this.fact("Decides differently", `in ${Math.round(b.differs * 100)}% of recent situations`),
        this.fact("Expects", b.value > 0.02 ? "things to improve" : b.value < -0.02 ? "things to get worse" : "no change"),
        this.fact("Last surprise", Math.abs(b.surprise) < 0.002 ? "as expected"
          : b.surprise > 0 ? `better than expected (+${b.surprise.toFixed(3)})` : `worse than expected (${b.surprise.toFixed(3)})`)));
    }

    // The network, now / as born / what changed.
    if (neural && b.w1) {
      const views = learns ? [["now", "Now"], ["born", "Born with"], ["learned", "What it learned"]] : [["now", "Now"]];
      if (!views.some(([v]) => v === this.brainView)) this.brainView = "now";
      if (views.length > 1) {
        out.push(el("div", { class: "seg small" }, views.map(([v, label]) => el("button", {
          class: this.brainView === v ? "on" : "", onclick: () => { this.brainView = v; this.renderInspector(); },
        }, label))));
      }
      out.push(this.networkSvg(b, meta, sp, this.brainView));
    }

    // Habits: what it was born doing versus what it does now.
    if (learns && b.habits) {
      const h = b.habits;
      const rows = [];
      const row = (label, pair, fmt, better) => {
        const [was, now] = pair || [];
        if (was == null || now == null) return;
        const moved = Math.abs(now - was) >= 0.01;
        rows.push(el("div", { class: "habit" }, el("span", {}, label),
          el("strong", {}, `${fmt(was)} → ${fmt(now)}`),
          el("span", { class: "muted small" }, moved ? better(now - was) : "unchanged")));
      };
      const pct = (v) => `${Math.round(v * 100)}%`;
      row(`${sp.diet === "plants" ? "Starts eating" : "Starts hunting"} at hunger`, h.eat, pct, (dv) => (dv < 0 ? "sooner" : "later"));
      row("Starts drinking at thirst", h.drink, pct, (dv) => (dv < 0 ? "sooner" : "later"));
      row("Flees at (× flight distance)", h.flee, (v) => `${v.toFixed(2)}×`, (dv) => (dv > 0 ? "earlier, while the predator is farther off" : "later, letting the predator come closer"));
      const likes = (b.shift || []).map((v, k) => [v, k]).filter(([v]) => v != null && Math.abs(v) >= 0.02)
        .sort((a, c) => Math.abs(c[0]) - Math.abs(a[0])).slice(0, 4);
      out.push(el("div", { class: "section-label" }, "🎓 What it has learned"),
        rows.length ? el("div", { class: "habits" }, rows) : null,
        likes.length ? el("div", { class: "why" }, el("span", { class: "muted small" }, "Has come to prefer: "),
          likes.map(([v, k]) => el("span", { class: `chip-why ${v >= 0 ? "up" : "down"}` }, `${v >= 0 ? "more" : "less"} ${opt(k)[1]}`)))
          : el("p", { class: "muted small" }, "No strong new preferences yet."),
        el("p", { class: "muted small" }, "Measured by testing its current wiring and the wiring it was born with on the same situations. Learned changes fade back towards instinct over about two weeks unless experience keeps reinforcing them, and they are not passed on to its young."));
    }

    // Its learning over its life.
    if (learns && b.log && b.log.length >= 2) {
      const xs = b.log.map((r) => r[0]);
      const born = [...b.born_w1.flat(), ...b.born_w2.flat()];
      const scale = born.reduce((s, v) => s + Math.abs(v), 0) / born.length || 1;
      const c1 = el("canvas", { class: "mini", role: "img", "aria-label": "How much it has rewired and how often it now decides differently, over its life" });
      const c2 = el("canvas", { class: "mini", role: "img", "aria-label": "Its wellbeing and surprise over its life" });
      this._charts.push([c1, { xs, zero: true, yFormat: (v) => `${Math.round(v)}%`,
        series: [{ ys: b.log.map((r) => (100 * r[3]) / scale), color: C["series-1"] }, { ys: b.log.map((r) => 100 * r[4]), color: C["series-2"] }] }]);
      this._charts.push([c2, { xs, zero: false,
        series: [{ ys: b.log.map((r) => r[1]), color: C["series-3"] }, { ys: b.log.map((r) => r[2]), color: C["series-4"] }] }]);
      out.push(el("div", { class: "section-label" }, "📈 Learning over its life"),
        el("figure", { class: "mini-chart" }, el("figcaption", { class: "legend" },
          el("span", {}, el("i", { style: `background:${C["series-1"]}` }), "rewired %"),
          el("span", {}, el("i", { style: `background:${C["series-2"]}` }), "decides differently %")), c1),
        el("figure", { class: "mini-chart" }, el("figcaption", { class: "legend" },
          el("span", {}, el("i", { style: `background:${C["series-3"]}` }), "wellbeing (reward/tick ×1000)"),
          el("span", {}, el("i", { style: `background:${C["series-4"]}` }), "surprise (dopamine ×1000)")), c2),
        el("p", { class: "muted small" }, "x-axis: its age in days. Surprise is the reward prediction error that drives learning: high while it is still finding out what pays, falling as its expectations become accurate."));
    }

    // Option scores, centred on zero.
    const scores = b.scores || [];
    const top = Math.max(0.5, ...scores.map((v, k) => (b.mask[k] ? Math.abs(v) : 0)));
    out.push(el("div", { class: "section-label" }, "Options right now"),
      el("div", { class: "scores" }, meta.options.map((name, k) => {
        const v = scores[k] ?? 0, ok = b.mask[k];
        const [e, label] = this.optionInfo(name, sp);
        const w = Math.min(50, (Math.abs(v) / top) * 50);
        return el("div", { class: `score${k === b.choice ? " chosen" : ""}${ok ? "" : " off"}`, title: ok ? "" : "Not possible right now" },
          el("span", {}, `${e} ${label}`),
          el("div", { class: "score-track" }, el("div", { class: "mid" }),
            ok ? el("div", { class: "score-fill", style: `${v >= 0 ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:${v >= 0 ? EXCITE : INHIBIT}` }) : null),
          el("span", { class: "num" }, ok ? v.toFixed(2) : "–"));
      })));

    // What it senses.
    out.push(el("div", { class: "section-label" }, "What it senses"),
      el("div", { class: "senses" }, meta.inputs.slice(0, -1).map((name, i) => {
        const v = b.inputs[i];
        const range = INPUT_RANGE[name] || 1;
        return el("div", { class: "sense" }, el("span", {}, name),
          el("div", { class: "track" }, el("div", { class: "fill", style: `width:${Math.max(0, Math.min(1, v / range)) * 100}%` })),
          el("span", { class: "num" }, compact(v)));
      })));

    // Memory of places and experience.
    if (m) {
      const hunter = m.skill != null;
      out.push(el("div", { class: "section-label" }, "🧭 Memory & experience"),
        el("div", { class: "bars" },
          hunter ? null : this.meter("Wariness", m.experience, "#8a6fdf"),
          hunter ? this.meter("Hunt skill", m.skill, "#eb6834") : null),
        el("div", { class: "facts" },
          this.fact(hunter ? "Hunting grounds" : "Feeding spots", `${m.food.filter((f) => f[2] >= 0.2).length} remembered`),
          this.fact("Danger spots", `${m.danger.filter((f) => f[2] >= 0.2).length} remembered`),
          d.traits.learning ? this.fact("Learning-rate gene", `${compact(d.traits.learning.value)}${b.plasticity ? ` (×${b.plasticity} now)` : ""}`) : null,
          neural && b.drift != null ? this.fact("Inborn wiring vs founders", `${compact(b.drift)} (evolution)`) : null),
        el("button", { class: `pill small${this.app.showMemory ? " accent" : ""}`, onclick: () => { this.app.showMemory = !this.app.showMemory; this.renderInspector(); } },
          this.app.showMemory ? "🗺️ Hide memory map" : "🗺️ Show memory on map"),
        this.app.showMemory ? el("div", { class: "legend" },
          el("span", {}, el("i", { style: "background:#3fd16b" }), hunter ? "remembered hunting ground" : "remembered good grazing"),
          el("span", {}, el("i", { style: "background:transparent;border:2px solid #ff3b3b;border-radius:50%" }), "remembered danger"),
          m.caches?.length ? el("span", {}, el("i", { style: "background:#ffb347;transform:rotate(45deg)" }), `buried food (${m.caches.length})`) : null,
          m.caches?.some((c) => c[3]) ? el("span", {}, el("i", { style: "background:#fff1a8;transform:rotate(45deg)" }), "food carried home for the young") : null,
          m.den ? el("span", {}, el("i", { style: "background:transparent;border:2px solid #ff8c42;border-radius:50%" }), "den with young") : null,
          m.territory ? el("span", {}, el("i", { style: "background:transparent;border:2px solid #ff9f43;border-radius:50%" }), "territory core") : null,
          m.marks?.length ? el("span", {}, el("i", { style: "background:#ffe14d" }), `scent marks (${m.marks.length})`) : null,
          m.mate ? el("span", {}, el("i", { style: "background:#ff7eb6;height:3px" }), "line to its mate") : null) : null,
        el("p", { class: "muted small" }, "Wariness grows with every predator it spots and every escape; it sharpens how fast it notices danger. Memories of places fade over a day or two. Young animals inherit their mother's map of good food at weaning."));
    }
    return out;
  }

  drawInspectorCharts() {
    for (const [canvas, opts] of this._charts || []) {
      if (canvas.isConnected) lineChart(canvas, opts);
    }
  }

  networkSvg(b, meta, sp, view = "now") {
    const W = 340, rowH = 14, nIn = meta.inputs.length, nHid = meta.hidden, nOut = meta.options.length;
    const H = nIn * rowH + 8;
    const xi = 104, xh = 190, xo = 264;
    const yIn = (i) => 8 + i * rowH;
    const yHid = (j) => 14 + (j * (H - 28)) / nHid;
    const yOut = (k) => 14 + (k * (H - 28)) / (nOut - 1);
    const hidden = b.hidden || [];
    const h1 = [...hidden, 1];
    const learned = view === "learned";
    const W1 = view === "born" ? b.born_w1 : b.w1;
    const W2 = view === "born" ? b.born_w2 : b.w2;
    // In "learned" view an edge shows the change since birth; otherwise the weight.
    const d1 = learned ? b.w1.map((row, i) => row.map((w, j) => w - b.born_w1[i][j])) : W1;
    const d2 = learned ? b.w2.map((row, j) => row.map((w, k) => w - b.born_w2[j][k])) : W2;
    const edges = [];
    let maxC1 = 1e-6, maxC2 = 1e-6, maxW = 1e-6;
    d1.forEach((row, i) => row.forEach((w, j) => { maxC1 = Math.max(maxC1, Math.abs(b.inputs[i] * w)); maxW = Math.max(maxW, Math.abs(w)); }));
    d2.forEach((row, j) => row.forEach((w, k) => { maxC2 = Math.max(maxC2, Math.abs(h1[j] * w)); maxW = Math.max(maxW, Math.abs(w)); }));
    const line = (x1, y1, x2, y2, w, c, max) => {
      const strength = learned || view === "born" ? Math.abs(w) / maxW : Math.abs(c) / max;
      if (learned && strength < 0.04) return null;
      return svg("line", {
        x1, y1, x2, y2, stroke: w >= 0 ? EXCITE : INHIBIT, "stroke-width": (0.4 + 2.2 * Math.abs(w) / maxW).toFixed(2),
        "stroke-opacity": ((learned ? 0.2 : 0.12) + 0.8 * Math.pow(strength, 0.7)).toFixed(2), "stroke-linecap": "round",
      });
    };
    d1.forEach((row, i) => row.forEach((w, j) => edges.push(line(xi, yIn(i), xh, yHid(j), w, b.inputs[i] * w, maxC1))));
    d2.forEach((row, j) => row.forEach((w, k) => edges.push(line(xh, j < nHid ? yHid(j) : H - 6, xo, yOut(k), w, h1[j] * w, maxC2))));
    const node = (x, y, r, fill, stroke, tip, extra = {}) => svg("circle", { cx: x, cy: y, r, fill, stroke, "stroke-width": 1.5, ...extra }, svg("title", {}, tip));
    const ink = C.ink || "#2b2a33", surface = C.surface || "#fff";
    const live = view === "now";
    const nodes = [
      ...meta.inputs.map((name, i) => {
        const v = Math.max(0, Math.min(1, b.inputs[i] / (INPUT_RANGE[name] || 1)));
        return [node(xi, yIn(i), 3.6, live ? alpha(ink.startsWith("#") ? ink : "#2b2a33", 0.1 + 0.85 * v) : surface, ink, `${name}: ${compact(b.inputs[i])}`),
          svg("text", { x: xi - 7, y: yIn(i) + 3, "text-anchor": "end", class: "net-label" }, name)];
      }),
      ...hidden.map((h, j) => node(xh, yHid(j), 6, live ? alpha(h >= 0 ? EXCITE : INHIBIT, Math.min(1, Math.abs(h))) : surface, ink,
        `hidden neuron ${j + 1}: ${h.toFixed(2)}`)),
      node(xh, H - 6, 3, surface, ink, "constant (bias)"),
      ...meta.options.map((name, k) => {
        const [e, label] = this.optionInfo(name, sp);
        const on = live && k === b.choice, ok = !live || b.mask[k];
        return [node(xo, yOut(k), on ? 7 : 5.5, on ? ink : surface, ink, `${label}: ${b.mask[k] ? (b.scores[k] ?? 0).toFixed(2) : "not possible now"}`,
          ok ? {} : { "stroke-dasharray": "2 2", "stroke-opacity": 0.5 }),
        svg("text", { x: xo + 11, y: yOut(k) + 4, class: `net-label${on ? " on" : ""}${ok ? "" : " off"}` }, `${e} ${label}`)];
      }),
    ];
    const caption = learned
      ? [el("span", {}, el("i", { style: `background:${EXCITE}` }), "strengthened"), el("span", {}, el("i", { style: `background:${INHIBIT}` }), "weakened"),
        el("span", { class: "muted" }, "by experience since birth; thicker = bigger change")]
      : view === "born"
        ? [el("span", {}, el("i", { style: `background:${EXCITE}` }), "excites"), el("span", {}, el("i", { style: `background:${INHIBIT}` }), "inhibits"),
          el("span", { class: "muted" }, "the wiring its genes gave it at birth")]
        : [el("span", {}, el("i", { style: `background:${EXCITE}` }), "excites"), el("span", {}, el("i", { style: `background:${INHIBIT}` }), "inhibits"),
          el("span", { class: "muted" }, "brighter = more signal now; thicker = stronger connection")];
    return el("figure", { class: "net" },
      svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": learned ? "How this animal's network has changed since birth" : "This animal's neural network" },
        svg("g", {}, edges), svg("g", {}, nodes.flat())),
      el("figcaption", { class: "legend" }, caption));
  }

  tab_genes(d, sp) {
    return [
      el("p", { class: "muted small" }, `Coat: ${d.coat}. Each gene has two copies, one from each parent; the dominant allele is shown first in the key.`),
      ...d.genes.map((g) => {
        const meta = sp.genes.find((x) => x.symbol === g.symbol);
        const homo = g.raw[0] === g.raw[1];
        return el("div", { class: "gene" },
          el("div", { class: "gene-top" },
            el("strong", {}, `${g.name} `, el("span", { class: "muted" }, `(${g.symbol})`)),
            el("span", { class: "geno-pills" }, el("span", { class: "allele" }, g.alleles[0]), el("span", { class: "allele" }, g.alleles[1]))),
          el("div", { class: "muted small" }, `${homo ? "Homozygous" : "Heterozygous"} · alleles ${meta.labels.join(" > ")}`),
          el("div", { class: "small" }, g.note));
      }),
    ];
  }

  tab_genome(d) {
    const kindColor = (locus, allele) => {
      if (locus.kind === "qtl") {
        const v = Math.max(-2, Math.min(2, allele)) / 2;
        return v >= 0 ? alpha("#e34948", 0.15 + 0.85 * v) : alpha("#2a78d6", 0.15 + 0.85 * -v);
      }
      if (locus.kind === "gene") return ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][this.alleleIndex(d, locus.label, allele)] || "#888";
      if (locus.kind === "load") return allele ? "#d03b3b" : "#d9d6cf";
      if (locus.kind === "brain") {
        const v = Math.max(-1, Math.min(1, allele / 0.25));
        return v >= 0 ? alpha("#b4549b", 0.2 + 0.8 * v) : alpha("#3d9a8b", 0.2 + 0.8 * -v);
      }
      const g = 90 + allele * 18;
      return `rgb(${g}, ${g}, ${g + 8})`;
    };
    const describe = (l, a) => l.kind === "qtl" ? `${l.label} QTL: ${a >= 0 ? "+" : ""}${a} SD`
      : l.kind === "gene" ? `${l.label} gene: ${a}` : l.kind === "load" ? `${l.label}: ${a ? "deleterious mutation" : "functional"}`
        : l.kind === "brain" ? `Brain wiring ${l.label}: this copy adds ${a >= 0 ? "+" : ""}${a} to one connection` : `${l.label} marker: allele ${a}`;
    const chroms = d.chromosomes.map((chrom, i) => el("div", { class: "chrom" },
      el("div", { class: "chrom-label" }, `Chr ${i + 1}`),
      el("div", { class: "chrom-pair" }, ["a", "b"].map((side) => el("div", { class: "strand" },
        chrom.map((locus) => el("span", {
          class: `locus ${locus.kind}`, style: `background:${kindColor(locus, locus[side])}`,
          title: `${side === "a" ? "Maternal" : "Paternal"} copy · ${describe(locus, locus[side])} · ${locus.pos} M`,
        }, locus.kind === "gene" ? String(locus[side]) : "")))))));
    return [
      el("p", { class: "muted small" }, `${d.chromosomes.length} chromosome pairs (top: from mother, bottom: from father). Hover a locus for details. Crossing over shuffles neighbouring blocks each generation.`),
      el("div", { class: "genome" }, chroms),
      el("div", { class: "legend" },
        el("span", {}, el("i", { style: "background:#e34948" }), "QTL raises trait"),
        el("span", {}, el("i", { style: "background:#2a78d6" }), "QTL lowers trait"),
        el("span", {}, el("i", { style: "background:#d03b3b" }), "deleterious mutation"),
        el("span", {}, el("i", { style: "background:#8a8a92" }), "neutral marker"),
        el("span", {}, el("i", { style: "background:#b4549b" }), el("i", { style: "background:#3d9a8b" }), "brain wiring (±)")),
      el("div", { class: "facts" },
        this.fact("Deleterious alleles", d.load), this.fact("…homozygous", d.load_hom),
        this.fact("Marker heterozygosity", `${d.het}/${d.markers}`), this.fact("Vigour", `${Math.round(d.vigor * 100)}%`)),
    ];
  }

  alleleIndex(d, symbol, label) {
    const sp = this.info.species.find((s) => s.name === d.species);
    const gene = sp.genes.find((g) => g.symbol === symbol);
    return gene ? gene.labels.indexOf(label) : 0;
  }

  tab_family(d) {
    const link = (ref, role) => !ref ? el("div", { class: "fact" }, el("span", {}, role), el("strong", { class: "muted" }, "founder"))
      : el("div", { class: "fact" }, el("span", {}, role),
        ref.alive ? el("button", { class: "link", onclick: () => this.app.select(ref.id) }, ref.name)
          : el("strong", { class: "muted" }, `${ref.name} ✝`));
    return [
      el("div", { class: "facts" }, link(d.mother, "Mother"), link(d.father, "Father"),
        d.partner ? link(d.partner, "Mate") : null,
        d.mate ? link(d.mate, "Father of litter") : null,
        this.fact("Offspring", d.children), this.fact("Generation", d.generation)),
      el("div", { class: "section-label" }, `Living offspring (${d.living_children.length})`),
      el("div", { class: "kids" }, d.living_children.length
        ? d.living_children.map((c) => el("button", { class: "kid", onclick: () => this.app.select(c.id) }, c.name))
        : el("span", { class: "muted small" }, "None alive")),
    ];
  }

  tab_lab(d, sp) {
    const edits = d.genes.map((g) => {
      const meta = sp.genes.find((x) => x.symbol === g.symbol);
      const pick = (current) => el("select", {}, meta.alleles.map((a, i) => el("option", { value: a, selected: a === current }, meta.labels[i])));
      const a = pick(g.raw[0]), b = pick(g.raw[1]);
      return el("div", { class: "edit-row" }, el("span", {}, `${g.name}`), a, b,
        el("button", { class: "pill small", onclick: () => this.app.lab("edit", { gene: g.symbol, alleles: [a.value, b.value] }) }, "Apply"));
    });
    return [
      el("div", { class: "lab-buttons" },
        el("button", { class: `pill${this.app.follow ? " accent" : ""}`, onclick: () => this.app.toggleFollow() }, "🎥 Follow"),
        el("button", { class: "pill", onclick: () => this.app.lab("clone") }, "🧬 Clone"),
        el("button", { class: `pill${this.app.breeding ? " accent" : ""}`, onclick: () => this.app.startBreeding() }, "💞 Breed with…"),
        el("button", { class: "pill danger", onclick: () => this.app.removeSelected() }, "🧹 Remove")),
      el("div", { class: "section-label" }, "🧪 Gene editor"),
      el("p", { class: "muted small" }, "Rewrite both copies of a colour gene. The coat changes immediately, and the edit is inherited by any offspring."),
      ...edits,
    ];
  }

  // ------------------------------------------------------------------ thought bubbles and labels

  bubbles(items) {
    const box = $("#bubbles");
    while (this.bubblePool.length < items.length) {
      const b = el("div", { class: "bubble" });
      box.append(b);
      this.bubblePool.push(b);
    }
    this.bubblePool.forEach((b, i) => {
      const it = items[i];
      if (!it) {
        b.style.display = "none";
        return;
      }
      b.style.display = "";
      if (b.textContent !== it.text) b.textContent = it.text;
      b.style.transform = `translate(${it.x}px, ${it.y}px) translate(-50%, -100%) scale(${it.scale})`;
    });
  }

  tag(pos, d) {
    const tag = $("#tag");
    if (!pos || !d || d.dead) {
      tag.hidden = true;
      return;
    }
    tag.hidden = false;
    const text = `${ACT_EMOJI[d.act] || ""} ${d.name}`;
    if (tag.textContent !== text) tag.textContent = text;
    tag.style.transform = `translate(${pos.x}px, ${pos.y}px) translate(-50%, -130%)`;
  }

  toast(message, kind = "info") {
    const t = $("#toast");
    t.textContent = message;
    t.className = `toast ${kind}`;
    t.hidden = false;
    clearTimeout(this._toast);
    this._toast = setTimeout(() => { t.hidden = true; }, 3500);
  }

  // ------------------------------------------------------------------ explanations

  setInsights(items) {
    this.insightLog = items.slice();
    this.unread = 0;
    this.report = null;
    this.popupQueue = [];
    $("#popups").replaceChildren();
    this.updateInsightCount();
    if (!$("#insights").hidden) this.renderInsights();
  }

  pushInsights(items) {
    const panelOpen = !$("#insights").hidden;
    for (const item of items) {
      this.insightLog.push(item);
      if (item.level === "major" && !panelOpen) this.unread++;
      const wanted = item.level === "major" || (item.level === "info" && this.prefs.tips);
      if (this.prefs.popups && wanted && !panelOpen) this.popupQueue.push(item);
      if (this.prefs.pause && item.level === "major" && this.app.running) this.app.togglePlay();
    }
    if (this.insightLog.length > 300) this.insightLog.splice(0, this.insightLog.length - 300);
    this.updateInsightCount();
    if (panelOpen) this.renderInsights();
    this.showPopups();
  }

  updateInsightCount() {
    const badge = $("#insight-count");
    badge.hidden = this.unread === 0;
    badge.textContent = this.unread > 9 ? "9+" : String(this.unread);
  }

  when(item) {
    const h = Math.floor(item.hours), m = Math.floor((item.hours - h) * 60);
    return `Day ${item.day} · ${String(h).padStart(2, "0")}:${String(m - (m % 5)).padStart(2, "0")} · ${title(item.season)}`;
  }

  insightCard(item, { popup = false, onClose } = {}) {
    const sp = this.info.species.find((s) => s.name === item.species);
    const actions = el("div", { class: "card-actions" },
      item.icon === "💀" && item.species ? el("button", {
        class: "pill small accent", onclick: () => { onClose?.(); this.askAI({ kind: "extinction", species: item.species }); },
      }, "🤖 Ask AI why") : null,
      item.chart ? el("button", { class: "pill small", onclick: () => this.openChart(item.chart) }, "📈 Show chart") : null,
      popup ? el("button", { class: "pill small", onclick: () => { onClose?.(); this.toggleInsights(true); } }, "All explanations") : null);
    return el("article", { class: `card ${item.level}` },
      el("div", { class: "card-icon" }, item.icon),
      el("div", { class: "card-main" },
        el("div", { class: "card-top" },
          el("strong", {}, item.title),
          popup ? el("button", { class: "card-close", "aria-label": "Dismiss", onclick: () => onClose?.() }, "✕") : null),
        el("div", { class: "card-meta" }, `${sp ? sp.icon + " " : ""}${this.when(item)}`),
        el("p", { class: "card-text" }, item.text),
        actions));
  }

  showPopups() {
    const box = $("#popups");
    while (box.children.length < 3 && this.popupQueue.length) {
      const item = this.popupQueue.shift();
      let timer;
      const close = () => {
        clearTimeout(timer);
        card.classList.add("leaving");
        setTimeout(() => { card.remove(); this.showPopups(); }, 220);
      };
      const card = this.insightCard(item, { popup: true, onClose: close });
      const life = item.level === "major" ? 16000 : 9000;
      const arm = () => { timer = setTimeout(close, life); };
      card.addEventListener("pointerenter", () => clearTimeout(timer));
      card.addEventListener("pointerleave", arm);
      box.prepend(card);
      arm();
    }
  }

  toggleInsights(open = $("#insights").hidden) {
    $("#insights").hidden = !open;
    if (open) {
      this.unread = 0;
      this.updateInsightCount();
      this.popupQueue = [];
      $("#popups").replaceChildren();
      this.renderInsights();
    }
  }

  renderInsights() {
    const panel = $("#insights");
    const pref = (key, label) => el("label", { class: "pref" },
      el("input", { type: "checkbox", checked: this.prefs[key], onchange: (e) => { this.prefs[key] = e.target.checked; } }), label);
    const list = this.insightLog.slice().reverse();
    panel.replaceChildren(
      el("div", { class: "insights-head" },
        el("h2", {}, "💡 What's happening, and why"),
        el("button", { class: "round small", "aria-label": "Close", onclick: () => this.toggleInsights(false) }, "✕")),
      el("p", { class: "muted small" }, "Every explanation comes from what the simulation recorded: causes of death, who bred, the genes they carried, food and weather. Nothing is guessed."),
      el("div", { class: "insights-tools" },
        el("button", { class: "pill accent", onclick: () => this.app.explain() }, "🔍 Explain right now"),
        pref("popups", "Pop-ups"), pref("tips", "Season tips"), pref("pause", "Pause on big changes"),
        el("label", { class: "pref" }, el("input", {
          type: "checkbox", checked: this.app.endOnExtinction,
          onchange: (e) => this.app.setEnding(e.target.checked),
        }), "🏁 End when a species dies out"),
        el("button", { class: "pill small", onclick: () => this.app.showSummary() }, "📜 Run summary")),
      el("div", { class: "insights-list" },
        this.report ? el("section", { class: "report-block" },
          el("div", { class: "section-label" }, `Right now · ${this.report.when}`),
          this.report.items.length ? this.report.items.map((i) => this.insightCard(i))
            : el("p", { class: "muted small" }, "Nothing has changed enough to explain yet. Let the simulation run for a day."),
          el("div", { class: "section-label" }, "Earlier")) : null,
        list.length ? list.map((i) => this.insightCard(i))
          : el("p", { class: "muted small" }, "Explanations appear here as things happen: population swings, alleles spreading or vanishing, traits evolving, and the start of each season.")));
  }

  showReport(items) {
    const env = this.app.env;
    this.report = { items, when: env ? this.when({ day: env.day, hours: env.hours, season: env.season }) : "" };
    this.toggleInsights(true);
    $("#insights .insights-list").scrollTop = 0;
  }

  // ------------------------------------------------------------------ run summary

  showSummary(sum) {
    const box = $("#summary");
    const info = this.info;
    const tpd = info.clock.ticks_per_day;
    const final = sum.final;
    const closeModal = () => { $("#summary-modal").hidden = true; };
    const pct = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(Math.round(x * 100))}%`;
    const causeBars = (deaths) => {
      const total = Object.values(deaths).reduce((a, b) => a + b, 0) || 1;
      return Object.entries(deaths).sort((a, b) => b[1] - a[1]).map(([cause, n]) =>
        el("div", { class: "cause" }, el("span", { class: "name" }, cause),
          el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(100 * n) / total}%` })),
          el("span", { class: "num" }, n.toLocaleString())));
    };
    const speciesCard = (sp) => {
      const meta = info.species.find((x) => x.name === sp.name);
      const fame = [];
      if (sp.records.oldest) fame.push(`👴 Oldest: ${sp.records.oldest.name} (${sp.records.oldest.days} days)`);
      if (sp.records.parent) fame.push(`👪 Most offspring: ${sp.records.parent.name} (${sp.records.parent.young})`);
      if (sp.records.hunter) fame.push(`🎯 Top hunter: ${sp.records.hunter.name} (${sp.records.hunter.kills} kills)`);
      return el("section", { class: "sum-species" },
        el("h3", {}, `${sp.icon} ${plural(sp.name)}`),
        el("div", { class: "facts" },
          this.fact("Peak", `${sp.peak} (day ${sp.peak_day})`), this.fact(sp.low_after_peak === false ? "Lowest" : "Lowest after peak", `${sp.low} (day ${sp.low_day})`),
          this.fact("Born", sp.births.toLocaleString()), this.fact("Generations", sp.generations),
          this.fact("Boom-bust cycles", sp.cycles), this.fact("At the end", sp.final)),
        el("div", { class: "section-label" }, "How they died"),
        el("div", { class: "bars-list" }, causeBars(sp.deaths)),
        sp.traits.length ? el("div", { class: "section-label" }, "Evolution (breeding values, start → end)") : null,
        el("div", { class: "evo" }, sp.traits.map((t) => el("div", { class: `evo-row ${Math.abs(t.change) >= 0.02 ? "moved" : ""}` },
          el("span", {}, meta?.traits[t.trait]?.label || t.label),
          el("span", { class: "muted" }, `${compact(t.start)} → ${compact(t.end)} ${t.unit}`),
          el("strong", {}, `${t.change > 0 ? "▲" : t.change < 0 ? "▼" : "·"} ${pct(t.change)}`)))),
        sp.alleles.length ? el("div", { class: "section-label" }, "Gene pool changes") : null,
        el("div", { class: "evo" }, sp.alleles.map((a) => el("div", { class: "evo-row moved" },
          el("span", {}, `${a.gene} ${a.allele}`),
          el("span", { class: "muted" }, `${Math.round(a.start * 100)}% → ${Math.round(a.end * 100)}%`),
          el("strong", {}, a.status === "lost" ? "lost" : a.status === "fixed" ? "fixed" : pct(a.end - a.start).replace("%", " pts"))))),
        sp.diversity ? el("p", { class: "muted small" }, `Genetic diversity (marker He): ${sp.diversity[0].toFixed(2)} → ${sp.diversity[1].toFixed(2)}`) : null,
        fame.length ? el("div", { class: "fame" }, fame.map((f) => el("span", {}, f))) : null);
    };
    const chart = el("canvas", { class: "sum-chart", role: "img", "aria-label": "Populations over the whole run" });
    box.replaceChildren(
      el("div", { class: "sum-head" },
        el("div", {},
          el("div", { class: "sum-kicker" }, final ? "🏁 Simulation over" : "📜 Run summary so far"),
          el("h2", {}, sum.headline),
          el("div", { class: "muted small" }, `${sum.days} days · ${sum.years} simulated years · seed ${sum.seed}`)),
        el("button", { class: "round small", "aria-label": "Close", onclick: closeModal }, "✕")),
      el("div", { class: "sum-body" },
        this.summaryStory(sum),
        el("figure", { class: "chart sum-figure" },
          el("figcaption", {}, el("span", {}, "Populations over the whole run"),
            el("span", { class: "legend" }, info.species.map((sp, i) => el("span", {}, el("i", { style: `background:${[C["series-1"], C["series-2"]][i] || C["series-1"]}` }), plural(sp.name))))),
          chart),
        el("div", { class: "sum-grid" }, sum.species.map(speciesCard)),
        sum.moments.length ? el("section", {},
          el("div", { class: "section-label" }, "Key moments"),
          el("ol", { class: "timeline" }, sum.moments.map((m) => el("li", {}, el("span", { class: "muted" }, `Day ${m.day}`), ` ${m.icon} ${m.title}`)))) : null,
        sum.events.length ? el("p", { class: "muted small" }, `Events you triggered: ${sum.events.map((e) => `${e.kind}${e.species ? ` (${e.species})` : ""} on day ${e.day}`).join(", ")}.`) : null),
      el("div", { class: "modal-actions" },
        el("button", { class: "pill", onclick: () => this.app.downloadCsv() }, "⬇ CSV"),
        final ? el("button", { class: "pill", onclick: () => { closeModal(); this.app.resume(); } }, "▶ Keep running") : null,
        final ? el("button", { class: "pill", onclick: () => { closeModal(); this.app.repopulate(); } }, "🐣 New population") : null,
        final ? el("button", { class: "pill accent", onclick: () => { closeModal(); $("#world-modal").hidden = false; } }, "🗺️ New world")
          : el("button", { class: "pill accent", onclick: closeModal }, "Close")));
    $("#summary-modal").hidden = false;
    requestAnimationFrame(() => {
      const series = sum.series;
      // One axis per chart: rabbits and foxes differ ~20-fold, so each is drawn
      // relative to its own peak (indexed), with the real peaks in the cards.
      const lines = info.species.map((sp, i) => {
        const ys = series.species[sp.name] || [];
        const peak = Math.max(1, ...ys);
        return { ys: ys.map((v) => v / peak), color: [C["series-1"], C["series-2"]][i] || C["series-1"] };
      });
      lineChart(chart, { xs: series.tick, series: lines, zero: true, yFormat: (v) => `${Math.round(v * 100)}%` });
    });
  }

  // The summary's story: written by the AI analyst when a key is set up, otherwise
  // the built-in notes with a pointer to the setup.
  summaryStory(sum) {
    const auto = sum.story.map((p) => el("p", {}, p));
    const st = this.ai.status;
    if (!st?.configured) {
      return el("section", { class: "sum-story" }, auto,
        el("div", { class: "ai-callout" }, "🔑 Want to know exactly why? Paste a Claude API key into ",
          el("code", {}, st?.env_file || "newForager/.env"),
          " and open this summary again: the AI analyst reads the run's records and map and proves why things happened."));
    }
    const answer = this.askAI({ kind: "summary" }, { open: false });
    return el("section", { class: "sum-story ai" },
      el("div", { class: "sum-ai-label" }, "🤖 What happened, and why · written by Claude from this run's records"),
      this.answerView(answer, { modal: true }),
      el("details", { class: "sum-auto" }, el("summary", {}, "Built-in notes (no AI)"), auto));
  }

  openChart(chart) {
    if (!chart) return;
    this.tab = chart.tab;
    if (chart.species) this.chartSel.species = chart.species;
    if (chart.gene) this.chartSel.gene = chart.gene;
    if (chart.trait) this.chartSel.trait = chart.trait;
    $("#drawer").hidden = false;
    document.body.classList.add("drawer-open");
    this.buildDrawer();
  }

  // ------------------------------------------------------------------ data drawer

  buildDrawer() {
    const tabs = [["ai", "🤖 AI analyst"], ["populations", "🐾 Populations"], ["deaths", "💀 Deaths & births"], ["traits", "📏 Traits"],
      ["genes", "🧬 Alleles"], ["diversity", "🌈 Diversity"], ["brains", "🧠 Brains"]];
    $("#drawer").classList.toggle("tall", this.tab === "ai");
    document.body.classList.toggle("drawer-tall", this.tab === "ai");
    $("#tabs").replaceChildren(...tabs.map(([id, label]) =>
      el("button", { class: this.tab === id ? "on" : "", onclick: () => { this.tab = id; this.buildDrawer(); } }, label)));
    const controls = $("#drawer-controls");
    controls.replaceChildren();
    const sel = this.chartSel;
    const sp = this.info.species.find((s) => s.name === sel.species) || this.info.species[0];
    if (this.tab !== "populations" && this.tab !== "ai") {
      controls.append(el("select", {
        "aria-label": "Species", onchange: (e) => {
          sel.species = e.target.value;
          const s2 = this.info.species.find((s) => s.name === sel.species);
          if (!s2.traits[sel.trait]) sel.trait = Object.keys(s2.traits)[0];
          sel.gene = s2.genes[0].symbol;
          this.buildDrawer();
        },
      }, this.info.species.map((s) => el("option", { value: s.name, selected: s.name === sel.species }, `${s.icon} ${plural(s.name)}`))));
    }
    if (this.tab === "traits") {
      controls.append(el("select", { "aria-label": "Trait", onchange: (e) => { sel.trait = e.target.value; this.buildDrawer(); } },
        Object.entries(sp.traits).map(([k, t]) => el("option", { value: k, selected: k === sel.trait }, t.label))));
    }
    if (this.tab === "genes") {
      controls.append(el("select", { "aria-label": "Gene", onchange: (e) => { sel.gene = e.target.value; this.buildDrawer(); } },
        sp.genes.map((g) => el("option", { value: g.symbol, selected: g.symbol === sel.gene }, `${g.name} (${g.symbol})`))));
    }

    const body = $("#drawer-body");
    const chart = (id, caption, legend = []) => el("figure", { class: "chart" },
      el("figcaption", {}, el("span", {}, caption), el("span", { class: "legend" }, legend.map(([color, label]) =>
        el("span", {}, el("i", { style: `background:${color}` }), label)))),
      el("canvas", { id, role: "img", "aria-label": caption }));
    const s1 = C["series-1"], s2 = C["series-2"], s3 = C["series-3"], s4 = C["series-4"];
    const spColor = (name) => [s1, s2][this.info.species.findIndex((s) => s.name === name)] || s1;
    if (this.tab === "ai") {
      body.replaceChildren(this.aiPanel());
      this.updateAIActions();
      this.updateAIAnswers();
      this.app.pullLedger(true);
      this.refreshAIStatus().then(() => this.updateAIActions());
      return;
    }
    if (this.tab === "deaths") {
      const causes = this.deathCauses(sp.name);
      this._causeKey = causes.map((c) => c.key).join();
      const meat = sp.diet !== "plants";
      body.replaceChildren(el("div", { class: "grid4" },
        chart("c-bd", `${sp.icon} Born and died, per day`, [[s1, "born"], [s2, "died"]]),
        chart("c-causes", `${sp.icon} What killed them (share of each day's deaths)`, causes.map((c, i) => [[s1, s2, s3, s4][i], c.label])),
        chart("c-adults", `${sp.icon} Adults old enough to breed`, [[s1, "females"], [s2, "males"]]),
        chart("c-food", meat ? `${sp.icon} Food per ${sp.name} (rabbits, plus voles counted as rabbits)` : `${sp.icon} Food: plant cover (share of capacity)`)),
      el("p", { class: "muted small note" }, "Recorded once a day. Numbers fall while the orange line is above the blue one. Check what killed them, and whether both sexes are left to breed: a population with no adult females can't recover however much food there is."));
      body.querySelectorAll("canvas").forEach((c) => this.hoverable(c));
      this.app.pullLedger(true);
      this.drawCharts();
      return;
    }
    if (this.tab === "populations") {
      body.replaceChildren(el("div", { class: "grid4" },
        ...this.info.species.map((s) => chart(`c-${s.name}`, `${s.icon} ${plural(s.name)}`)),
        chart("c-plants", "🌿 Plant biomass"),
        chart("c-voles", "🐭 Voles (estimated, foxes' staple prey)"),
        chart("c-phase", `Phase plane: ${plural(this.info.species[1]?.name || "").toLowerCase()} vs ${plural(this.info.species[0].name).toLowerCase()}`)));
    } else if (this.tab === "traits") {
      const t = sp.traits[sel.trait];
      body.replaceChildren(el("div", { class: "grid2" },
        chart("c-trait", `${t.label} (${t.unit}) · mean ± 1 SD`, [[spColor(sp.name), "phenotype"], [C["ink-2"], "breeding value"]]),
        chart("c-trait-sd", `${t.label}: variation (SD)`)));
    } else if (this.tab === "genes") {
      const g = sp.genes.find((x) => x.symbol === sel.gene);
      body.replaceChildren(el("div", { class: "grid2" },
        chart("c-freq", `${g.name} allele frequencies`, g.labels.map((l, i) => [[s1, s2, s3, s4][i], l])),
        chart("c-het", "Heterozygosity: observed vs Hardy–Weinberg expected", [[s1, "observed"], [s2, "expected"]])),
      el("p", { class: "muted small note" }, g.note));
    } else if (this.tab === "brains") {
      const neural = this.info.brain.mode === "neural";
      const learns = this.info.brain.learning;
      const both = this.info.species.map((s) => [spColor(s.name), plural(s.name)]);
      const prey = this.info.species.some((o) => o.prey.includes(sp.name));
      const eatWord = sp.diet === "plants" ? "eating" : "hunting";
      body.replaceChildren(el("div", { class: "grid3 brains" },
        chart("c-instinct", "Clear-cut decisions that match the founders' instincts", both),
        chart("c-differs", "Decide differently from how they were born (learned in life)", both),
        chart("c-eat", `${sp.icon} Start ${eatWord} at hunger: born with vs learned`, [[C["ink-2"], "born with (genes)"], [spColor(sp.name), "now (learned)"]]),
        chart("c-habit2", prey ? `${sp.icon} Flee when a predator is within × flight distance` : `${sp.icon} Start drinking at thirst`,
          [[C["ink-2"], "born with (genes)"], [spColor(sp.name), "now (learned)"]]),
        el("div", { class: "chart text", id: "brain-changes" })),
      el("p", { class: "muted small note" }, !neural
        ? "This world uses fixed instincts. Choose “Evolving neural networks” under 🗺️ New world to let behaviour evolve and learn."
        : `Two kinds of change. Evolution: brain wiring is inherited, so the “born with” lines shift over generations (a falling instinct line means brains are evolving away from the instincts the founders copied). ${learns
          ? "Learning: during its life each animal rewires itself from rewards and punishments, so “now” pulls away from “born with”. Learned habits are not inherited, but good learners leave more young."
          : "Lifetime learning is switched off in this world."} Close calls are left out of the comparisons.`));
    } else {
      body.replaceChildren(el("div", { class: "grid3" },
        chart("c-div", "Neutral marker diversity", [[s1, "expected He"], [s2, "observed Ho"]]),
        chart("c-vigor", "Mean genetic vigour"),
        chart("c-load", "Deleterious alleles per animal")));
    }
    body.querySelectorAll("canvas").forEach((c) => this.hoverable(c));
    this.drawCharts();
  }

  drawCharts() {
    if ($("#drawer").hidden || this.tab === "ai") return;
    if (this.tab === "deaths") {
      this.drawDeathCharts();
      return;
    }
    const h = this.app.hist;
    if (!h) return;
    const s1 = C["series-1"], s2 = C["series-2"], s3 = C["series-3"], s4 = C["series-4"];
    const xs = h.tick, hover = this.hover;
    const info = this.info;
    const sel = this.chartSel;
    const spColor = (name) => [s1, s2][info.species.findIndex((s) => s.name === name)] || s1;
    const q = (id) => document.getElementById(id);
    if (this.tab === "populations") {
      info.species.forEach((s) => lineChart(q(`c-${s.name}`), { xs, series: [{ ys: h.species[s.name].count, color: spColor(s.name) }], hover }));
      lineChart(q("c-plants"), { xs, series: [{ ys: h.plants, color: s3 }], hover });
      lineChart(q("c-voles"), { xs, series: [{ ys: h.voles || [], color: s4 }], hover, emptyText: "No vole data yet" });
      if (info.species.length > 1) {
        phaseChart(q("c-phase"), {
          xs: h.species[info.species[0].name].count, ys: h.species[info.species[1].name].count,
          xLabel: plural(info.species[0].name), yLabel: plural(info.species[1].name), color: s1, hover,
        });
      }
    } else if (this.tab === "traits") {
      const t = h.species[sel.species].traits[sel.trait];
      const lo = t.mean.map((m, i) => (m == null ? null : m - t.sd[i]));
      const hi = t.mean.map((m, i) => (m == null ? null : m + t.sd[i]));
      lineChart(q("c-trait"), { xs, zero: false, hover, series: [{ ys: t.mean, lo, hi, color: spColor(sel.species) }, { ys: t.genetic, color: C["ink-2"] }] });
      lineChart(q("c-trait-sd"), { xs, zero: true, hover, series: [{ ys: t.sd, color: spColor(sel.species) }] });
    } else if (this.tab === "genes") {
      const g = h.species[sel.species].genes[sel.gene];
      const sp = info.species.find((s) => s.name === sel.species);
      const gene = sp.genes.find((x) => x.symbol === sel.gene);
      stackedChart(q("c-freq"), { xs, hover, layers: gene.alleles.map((_, k) => ({ ys: g.freq.map((f) => (f ? f[k] : null)), color: [s1, s2, s3, s4][k] })) });
      lineChart(q("c-het"), { xs, hover, zero: true, series: [{ ys: g.ho, color: s1 }, { ys: g.he, color: s2 }] });
    } else if (this.tab === "brains") {
      const off = info.brain.mode === "neural" ? "Collecting decisions…" : "Brains are off: animals use fixed instincts";
      const pc = (v) => `${Math.round(v * 100)}%`;
      lineChart(q("c-instinct"), {
        xs, hover, zero: true, yFormat: pc, emptyText: off,
        series: info.species.map((s) => ({ ys: h.species[s.name].instinct || [], color: spColor(s.name) })),
      });
      lineChart(q("c-differs"), {
        xs, hover, zero: true, yFormat: pc, emptyText: info.brain.learning ? "Collecting…" : "Lifetime learning is off",
        series: info.species.map((s) => ({ ys: h.species[s.name].brain?.differs || [], color: spColor(s.name) })),
      });
      const br = h.species[sel.species].brain || {};
      const prey = info.species.some((o) => o.prey.includes(sel.species));
      lineChart(q("c-eat"), {
        xs, hover, zero: false, yFormat: pc, emptyText: off,
        series: [{ ys: br.eat_born || [], color: C["ink-2"] }, { ys: br.eat_now || [], color: spColor(sel.species) }],
      });
      lineChart(q("c-habit2"), {
        xs, hover, zero: false, yFormat: prey ? (v) => `${v.toFixed(2)}×` : pc, emptyText: off,
        series: prey ? [{ ys: br.flee_born || [], color: C["ink-2"] }, { ys: br.flee_now || [], color: spColor(sel.species) }]
          : [{ ys: br.drink_born || [], color: C["ink-2"] }, { ys: br.drink_now || [], color: spColor(sel.species) }],
      });
      this.renderBrainChanges();
    } else {
      const row = h.species[sel.species];
      lineChart(q("c-div"), { xs, hover, zero: true, series: [{ ys: row.diversity.he, color: s1 }, { ys: row.diversity.ho, color: s2 }] });
      lineChart(q("c-vigor"), { xs, hover, zero: false, series: [{ ys: row.vigor, color: spColor(sel.species) }] });
      lineChart(q("c-load"), { xs, hover, zero: true, series: [{ ys: row.load, color: s2 }] });
    }
  }

  renderBrainChanges() {
    const box = document.getElementById("brain-changes");
    if (!box) return;
    const state = this.app.frame?.brains;
    const neural = this.info.brain.mode === "neural";
    box.replaceChildren(...this.info.species.map((s, i) => {
      const st = state?.[i];
      return el("div", { class: "changes" },
        el("div", { class: "section-label" }, `${s.icon} ${plural(s.name)}: commonest changes of mind`),
        !neural ? el("p", { class: "muted small" }, "Brains are switched off in this world (fixed instincts).")
          : !st ? el("p", { class: "muted small" }, "Collecting decisions… (checked every 2 days)")
            : !st.changes.length ? el("p", { class: "muted small" }, "None: every choice matches the founders' instincts.")
              : el("ol", {}, st.changes.map((c) => el("li", {}, `instinct: ${c.instinct} → now: ${c.now} `,
                el("span", { class: "muted" }, `(${Math.round(c.share * 100)}% of changes)`)))),
        st ? el("p", { class: "muted small" },
          `Now ${Math.round(st.agreement * 100)}% of clear-cut decisions match instinct; founders matched ${Math.round(st.founders * 100)}%.`) : null);
    }));
  }

  hoverable(canvas) {
    const move = (e) => {
      const xs = this.tab === "deaths" ? this.app.ledger?.tick : this.app.hist?.tick;
      if (!xs) return;
      if (canvas._phase) {
        const r = canvas.getBoundingClientRect();
        const p = canvas._phase;
        let best = null, bd = 24 * 24;
        for (let i = 0; i < p.xs.length; i++) {
          const dd = (p.X(p.xs[i]) - (e.clientX - r.left)) ** 2 + (p.Y(p.ys[i]) - (e.clientY - r.top)) ** 2;
          if (dd <= bd) { best = i; bd = dd; }
        }
        this.hover = best;
      } else if (canvas._geo) {
        const g = canvas._geo;
        const r = canvas.getBoundingClientRect();
        const fx = Math.max(0, Math.min(1, (e.clientX - r.left - g.pad.l) / g.pw));
        this.hover = hoverIndex(xs, g.x0 + fx * (g.x1 - g.x0));
      }
      this.drawCharts();
      this.chartTip(e.clientX, e.clientY);
    };
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      $("#chart-tip").hidden = true;
      this.drawCharts();
    });
  }

  chartTip(cx, cy) {
    if (this.tab === "deaths") {
      this.deathTip(cx, cy);
      return;
    }
    const h = this.app.hist, i = this.hover, tip = $("#chart-tip");
    if (i == null || !h) { tip.hidden = true; return; }
    const tpd = this.info.clock.ticks_per_day;
    const rows = [el("div", { class: "when" }, `Day ${Math.floor((h.tick[i] + tpd / 4) / tpd) + 1} · tick ${h.tick[i].toLocaleString()}`)];
    const row = (color, value, label) => el("div", { class: "row" }, el("i", { style: `background:${color}` }), el("strong", {}, value), el("span", {}, label));
    const s1 = C["series-1"], s2 = C["series-2"], s3 = C["series-3"], s4 = C["series-4"];
    const sel = this.chartSel;
    if (this.tab === "populations") {
      this.info.species.forEach((s, k) => rows.push(row([s1, s2][k], compact(h.species[s.name].count[i]), plural(s.name).toLowerCase())));
      rows.push(row(s3, compact(h.plants[i]), "plant biomass"));
      if (h.voles) rows.push(row(s4, compact(h.voles[i]), "voles"));
    } else if (this.tab === "traits") {
      const t = h.species[sel.species].traits[sel.trait];
      rows.push(row(s1, t.mean[i] == null ? "–" : `${compact(t.mean[i])} ± ${compact(t.sd[i])}`, "phenotype"));
      rows.push(row(C["ink-2"], compact(t.genetic[i]), "breeding value"));
    } else if (this.tab === "genes") {
      const g = h.species[sel.species].genes[sel.gene];
      const gene = this.info.species.find((s) => s.name === sel.species).genes.find((x) => x.symbol === sel.gene);
      (g.freq[i] || []).forEach((f, k) => rows.push(row([s1, s2, s3, s4][k], `${Math.round(f * 100)}%`, gene.labels[k])));
      rows.push(row(s1, compact(g.ho[i]), "observed heterozygosity"), row(s2, compact(g.he[i]), "expected"));
    } else if (this.tab === "brains") {
      const pc = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
      this.info.species.forEach((s, k) => {
        rows.push(row([s1, s2][k], pc(h.species[s.name].instinct?.[i]), `${plural(s.name).toLowerCase()} follow instinct`));
        rows.push(row([s1, s2][k], pc(h.species[s.name].brain?.differs?.[i]), "decide differently than born"));
      });
      const br = h.species[sel.species].brain || {};
      rows.push(row(C["ink-2"], pc(br.eat_born?.[i]), "eat/hunt at hunger, born with"), row(s1, pc(br.eat_now?.[i]), "…now, learned"));
      if (br.flee_born?.[i] != null) {
        rows.push(row(C["ink-2"], `${br.flee_born[i].toFixed(2)}×`, "flee distance, born with"), row(s1, `${br.flee_now[i].toFixed(2)}×`, "…now, learned"));
      }
    } else {
      const r = h.species[sel.species];
      rows.push(row(s1, compact(r.diversity.he[i]), "expected He"), row(s2, compact(r.diversity.ho[i]), "observed Ho"),
        row(s3, compact(r.vigor[i]), "vigour"), row(s4, compact(r.load[i]), "deleterious alleles"));
    }
    this.placeTip(rows, cx, cy);
  }

  placeTip(rows, cx, cy) {
    const tip = $("#chart-tip");
    tip.replaceChildren(...rows);
    tip.hidden = false;
    const w = tip.offsetWidth, hh = tip.offsetHeight;
    tip.style.left = `${Math.min(innerWidth - w - 8, cx + 14)}px`;
    tip.style.top = `${Math.max(8, cy - hh - 12)}px`;
  }

  // ------------------------------------------------------------------ Deaths & births

  // The species' main causes of death over the run (at most four bands; the rest go in "other").
  deathCauses(name) {
    const L = this.app.ledger, s = L?.species[name];
    if (!s) return [];
    const totals = L.causes.map((c) => [c, s.causes[c].reduce((a, b) => a + b, 0)]).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
    const many = totals.length > 4;
    const out = totals.slice(0, many ? 3 : 4).map(([c]) => ({ key: c, label: CAUSE_LABEL[c] || c }));
    if (many) out.push({ key: "other", label: "other", others: totals.slice(3).map(([c]) => c) });
    return out;
  }

  causeSeries(s, c) {
    return c.key === "other" ? s.deaths.map((_, i) => c.others.reduce((a, o) => a + s.causes[o][i], 0)) : s.causes[c.key];
  }

  drawDeathCharts() {
    const L = this.app.ledger, sel = this.chartSel;
    const s = L?.species[sel.species];
    const q = (id) => document.getElementById(id);
    const s1 = C["series-1"], s2 = C["series-2"], s3 = C["series-3"], s4 = C["series-4"];
    const xs = L?.tick || [], hover = this.hover;
    const sp = this.info.species.find((x) => x.name === sel.species);
    const color = [s1, s2][this.info.species.indexOf(sp)] || s1;
    const none = { ys: [], color: s1 };
    lineChart(q("c-bd"), { xs, hover, series: s ? [{ ys: s.births, color: s1 }, { ys: s.deaths, color: s2 }] : [none] });
    const layers = s ? this.deathCauses(sel.species).map((c, k) => {
      const ys = this.causeSeries(s, c);
      return { color: [s1, s2, s3, s4][k], ys: s.deaths.map((tot, i) => (tot > 0 ? ys[i] / tot : null)) };
    }) : [];
    stackedChart(q("c-causes"), { xs: layers.length ? xs : [], layers, hover });
    lineChart(q("c-adults"), { xs, hover, series: s ? [{ ys: s.af, color: s1 }, { ys: s.am, color: s2 }] : [none] });
    lineChart(q("c-food"), {
      xs, hover, zero: true, yFormat: sp?.diet === "plants" ? (v) => `${Math.round(v * 100)}%` : undefined,
      series: [{ ys: s ? s.food : [], color }], emptyText: "None left to feed",
    });
  }

  deathTip(cx, cy) {
    const L = this.app.ledger, i = this.hover, sel = this.chartSel;
    const s = L?.species[sel.species];
    if (i == null || !s || i >= L.tick.length) { $("#chart-tip").hidden = true; return; }
    const s1 = C["series-1"], s2 = C["series-2"], s3 = C["series-3"], s4 = C["series-4"];
    const row = (color, value, label) => el("div", { class: "row" }, el("i", { style: `background:${color}` }), el("strong", {}, value), el("span", {}, label));
    const sp = this.info.species.find((x) => x.name === sel.species);
    const rows = [el("div", { class: "when" }, `Day ${L.day[i]} · ${s.n[i]} ${plural(sp.name).toLowerCase()}`),
      row(s1, compact(s.births[i]), "born"), row(s2, compact(s.deaths[i]), "died")];
    this.deathCauses(sel.species).forEach((c, k) => rows.push(row([s1, s2, s3, s4][k], compact(this.causeSeries(s, c)[i]), `of ${c.label}`)));
    rows.push(row(s1, String(s.af[i]), "adult females"), row(s2, String(s.am[i]), "adult males"));
    if (s.territories[i] != null) rows.push(row(C["ink-2"], `${s.territories[i]} (${s.paired[i]})`, "territories (pairs)"), row(C["ink-2"], String(s.floaters[i]), "floaters"));
    const food = s.food[i];
    rows.push(row(C["ink-2"], food == null ? "–" : sp.diet === "plants" ? `${Math.round(food * 100)}%` : compact(food),
      sp.diet === "plants" ? "plant cover" : `food per ${sp.name}`));
    this.placeTip(rows, cx, cy);
  }

  ledgerUpdated() {
    if ($("#drawer").hidden) return;
    if (this.tab === "ai") {
      document.getElementById("ai-glance")?.replaceChildren(...[this.glanceView()].flat());
    } else if (this.tab === "deaths") {
      if (this.deathCauses(this.chartSel.species).map((c) => c.key).join() !== this._causeKey) this.buildDrawer();
      else this.drawCharts();
    }
  }

  // ------------------------------------------------------------------ AI analyst

  async refreshAIStatus() {
    try {
      this.ai.status = await getJSON("/api/ai/status");
    } catch {
      // keep the last known status
    }
    return this.ai.status;
  }

  dayLabel(tick) {
    const tpd = this.info.clock.ticks_per_day;
    return `Day ${Math.floor((tick + tpd / 4) / tpd) + 1}`;
  }

  aiTitle(req) {
    if (req.kind === "summary") return "📜 Run summary";
    if (req.kind === "extinction") return `💀 Why did the ${plural(req.species).toLowerCase()} die out?`;
    return `💬 ${req.question}`;
  }

  // Ask the analyst. The same question at the same moment is answered once.
  askAI(req, { open = true } = {}) {
    const tick = this.app.frame?.tick;
    const same = this.ai.answers.find((a) => a.kind === req.kind && (a.species || null) === (req.species || null)
      && (a.question || "") === (req.question || "") && a.tick === tick && !a.error);
    if (same) {
      if (open) this.openAITab();
      return same;
    }
    const answer = { id: ++this.ai.seq, ...req, tick, title: this.aiTitle(req), text: "", meta: null, error: null, done: false, info: null, views: new Set() };
    this.ai.answers.unshift(answer);
    if (open) this.openAITab();
    streamAI(req, (ev) => {
      if (ev.meta) answer.meta = ev.meta;
      if (ev.text) answer.text += ev.text;
      if (ev.error) answer.error = ev.error;
      if (ev.done) answer.info = ev;
      this.paintAnswer(answer);
    }).catch((err) => {
      answer.error = err.message;
    }).finally(() => {
      if (!answer.text && !answer.error) answer.error = "The answer stopped before it began. Try again.";
      answer.done = true;
      this.paintAnswer(answer);
    });
    return answer;
  }

  openAITab() {
    if (this.tab === "ai" && !$("#drawer").hidden) {
      this.updateAIAnswers();
      return;
    }
    this.tab = "ai";
    $("#drawer").hidden = false;
    document.body.classList.add("drawer-open");
    this.buildDrawer();
  }

  paintAnswer(answer) {
    if (answer._raf) return;
    answer._raf = requestAnimationFrame(() => {
      answer._raf = 0;
      for (const v of answer.views) {
        if (v.isConnected) this.fillAnswer(v, answer);
        else answer.views.delete(v);
      }
    });
  }

  answerView(answer, { modal = false } = {}) {
    const v = el("article", { class: `ai-answer${modal ? " in-modal" : ""}` });
    v._modal = modal;
    answer.views.add(v);
    this.fillAnswer(v, answer);
    return v;
  }

  fillAnswer(v, a) {
    if (!v._parts) {
      v._parts = { head: el("header", {}), body: el("div", { class: "md" }), foot: el("footer", { class: "ai-foot" }) };
      v.append(...(v._modal ? [] : [v._parts.head]), v._parts.body, v._parts.foot);
    }
    const { head, body, foot } = v._parts;
    if (!v._modal) head.replaceChildren(el("strong", {}, a.title), a.tick != null ? el("span", { class: "muted small" }, this.dayLabel(a.tick)) : null);
    const kids = [];
    if (a.text) kids.push(renderMarkdown(a.text));
    if (a.error) kids.push(el("p", { class: "ai-error" }, `⚠️ ${a.error}`));
    if (!a.text && !a.error) kids.push(el("p", { class: "ai-wait" }, "🤖 Claude is reading this run's records and map…"));
    body.replaceChildren(...kids);
    v.classList.toggle("streaming", !a.done);
    if (a.meta && foot._for !== a.meta.evidence) {
      foot._for = a.meta.evidence;
      const m = a.meta;
      const gone = m.extinctions.length ? `, ${m.extinctions.length} extinction${m.extinctions.length > 1 ? "s" : ""}` : "";
      foot.replaceChildren(
        el("span", {}, `Based on ${m.days} days of daily records, the last ${m.deaths_logged} deaths${gone}, your sandbox actions and the map.`),
        el("a", { href: `/api/ai/evidence/${m.evidence}.txt`, target: "_blank", rel: "noopener" }, "📄 See the evidence"),
        el("details", { class: "ai-map" }, el("summary", {}, "🗺️ The map it looked at"),
          el("img", { src: `/api/ai/evidence/${m.evidence}.png`, alt: "Top-down map of the world given to the AI: terrain, animals, territories and recent deaths", loading: "lazy" })),
        el("span", { class: "ai-usage" }));
    }
    const usage = foot.querySelector(".ai-usage");
    if (usage && a.done && a.info) {
      const i = a.info;
      usage.textContent = i.cached ? "Saved answer: no new cost."
        : `${i.model || ""}${i.input_tokens ? ` · ${i.input_tokens.toLocaleString()} tokens read, ${(i.output_tokens || 0).toLocaleString()} written` : ""}${i.stop_reason === "max_tokens" ? " · cut short at the length limit" : ""}`;
    }
  }

  aiPanel() {
    return el("div", { class: "ai-panel" },
      el("div", { class: "ai-side" },
        el("div", { class: "section-label" }, "At a glance (last 3 days)"),
        el("div", { class: "glance", id: "ai-glance" }, this.glanceView()),
        el("div", { class: "section-label" }, "🤖 Ask the AI analyst"),
        el("div", { class: "ai-actions", id: "ai-actions" })),
      el("div", { class: "ai-answers", id: "ai-answers" }));
  }

  // Plain numbers anyone can read, straight from the daily records (no AI needed).
  glanceView() {
    const L = this.app.ledger;
    if (!L || L.tick.length < 1) return el("p", { class: "muted small" }, "Collecting the first day of records…");
    const n = L.tick.length, last = n - 1, from = Math.max(0, n - 3);
    const sum = (a) => a.slice(from).reduce((x, y) => x + (y || 0), 0);
    return this.info.species.map((sp) => {
      const s = L.species[sp.name];
      const now = s.n[last], before = s.n[Math.max(0, n - 4)];
      const born = sum(s.births), died = sum(s.deaths);
      const top = L.causes.map((c) => [c, sum(s.causes[c])]).sort((a, b) => b[1] - a[1])[0];
      const cause = died > 0 && top[1] > 0 ? `, mostly ${CAUSE_LABEL[top[0]] || top[0]} (${Math.round((100 * top[1]) / died)}%)` : "";
      const change = before ? (now - before) / before : null;
      const warn = [];
      if (now === 0) warn.push("💀 None left");
      else {
        if (s.af[last] === 0) warn.push("⚠️ No adult females: they can't breed");
        if (s.am[last] === 0) warn.push("⚠️ No adult males: they can't breed");
        if (now <= 4) warn.push("⚠️ Close to dying out");
      }
      const terr = s.territories[last] != null ? ` · ${s.territories[last]} territories (${s.paired[last]} pairs), ${s.floaters[last]} floaters` : "";
      return el("div", { class: "glance-row" },
        el("div", { class: "glance-top" }, el("strong", {}, `${sp.icon} ${plural(sp.name)}`), el("span", { class: "big" }, now.toLocaleString()),
          change != null && Math.round(change * 100) !== 0 ? el("span", { class: change < 0 ? "down" : "up" }, `${change < 0 ? "▼" : "▲"} ${Math.abs(Math.round(change * 100))}%`) : null),
        el("div", { class: "muted small" }, `Adults: ${s.af[last]} ♀ · ${s.am[last]} ♂${terr}`),
        el("div", { class: "muted small" }, `${Math.round(born).toLocaleString()} born, ${Math.round(died).toLocaleString()} died${cause}`),
        warn.length ? el("div", { class: "glance-warn" }, warn.join(" · ")) : null);
    });
  }

  updateAIActions() {
    const box = document.getElementById("ai-actions");
    if (!box) return;
    const st = this.ai.status;
    if (!st) {
      box.replaceChildren(el("p", { class: "muted small" }, "Checking the AI connection…"));
      return;
    }
    if (!st.configured) {
      box.replaceChildren(el("div", { class: "ai-setup" },
        el("strong", {}, "🔑 Add your Claude API key to switch on the analyst"),
        el("ol", {},
          el("li", {}, "Open ", el("code", {}, st.env_file)),
          el("li", {}, "Paste your key after ", el("code", {}, "ANTHROPIC_API_KEY="), " and save the file"),
          el("li", {}, "Click Check again. No restart needed.")),
        el("button", { class: "pill small", onclick: () => this.refreshAIStatus().then(() => this.updateAIActions()) }, "🔄 Check again")));
      return;
    }
    const gone = new Map();
    for (const e of st.extinctions) gone.set(e.species, e);
    const buttons = [el("button", { class: "pill accent", onclick: () => this.askAI({ kind: "summary" }) }, "📜 Summarize this run")];
    for (const e of gone.values()) {
      buttons.push(el("button", { class: "pill", onclick: () => this.askAI({ kind: "extinction", species: e.species }) },
        `💀 Why did the ${plural(e.species).toLowerCase()} die out? (day ${e.day})`));
    }
    const input = el("input", { type: "text", maxlength: 600, placeholder: "Ask anything, e.g. why do rabbits crash every winter?", "aria-label": "Question for the AI analyst" });
    const form = el("form", {
      class: "ai-ask", onsubmit: (ev) => {
        ev.preventDefault();
        const q = input.value.trim();
        if (q) {
          this.askAI({ kind: "question", question: q });
          input.value = "";
        }
      },
    }, input, el("button", { class: "pill accent", type: "submit" }, "Ask"));
    box.replaceChildren(el("div", { class: "ai-buttons" }, buttons), form,
      el("p", { class: "muted small" }, `Claude (${st.model}) reads this run's records and a map, then explains what happened and backs it up with days and numbers. Each answer is sent to Anthropic's API and uses your key's credit.`));
  }

  updateAIAnswers() {
    const box = document.getElementById("ai-answers");
    if (!box) return;
    if (!this.ai.answers.length) {
      box.replaceChildren(el("div", { class: "ai-intro" },
        el("h3", {}, "Why did that happen?"),
        el("p", {}, "The AI analyst reads everything this run recorded: births and deaths by cause for every day, who was left before an extinction and how they died, food, weather, territories, your sandbox actions and a map of the world. It explains what happened and proves why, quoting the days and numbers."),
        el("p", { class: "muted small" }, "Nothing is sent until you ask. Pick a button on the left or type a question.")));
      return;
    }
    box.replaceChildren(...this.ai.answers.map((a) => this.answerView(a)));
    box.scrollTop = 0;
  }
}

// Premade maps. Starting numbers are scaled to the land so the food supply is
// about the same per animal, and immigration to the length of the edge, where
// newcomers arrive. A fox territory needs roughly 28 x 28 cells, so on the tiny
// plot one fox pair fills it; populations that small swing wildly, so its run
// keeps going through extinctions and newcomers refill it.
const MAPS = {
  standard: { label: "🌳 Standard meadow (160 × 100)", width: 160, height: 100, water: 12, forest: 12, rock: 5, immigration: 0.001, initial: { rabbit: 160, fox: 14 } },
  small: { label: "🏡 Small valley (80 × 50): about 4 fox territories", width: 80, height: 50, water: 10, forest: 10, rock: 3, immigration: 0.0005, initial: { rabbit: 50, fox: 4 } },
  tiny: { label: "🔬 Tiny test plot (48 × 32): one fox pair and a few rabbits to watch closely", width: 48, height: 32, water: 8, forest: 6, rock: 0, immigration: 0.0006, keepGoing: true, initial: { rabbit: 24, fox: 2 } },
  lakes: { label: "🏞️ Lake district (160 × 100, lots of water)", width: 160, height: 100, water: 30, forest: 10, rock: 3, immigration: 0.001, initial: { rabbit: 120, fox: 10 } },
  plains: { label: "🌾 Open plains (160 × 100, little cover)", width: 160, height: 100, water: 6, forest: 3, rock: 2, immigration: 0.001, initial: { rabbit: 180, fox: 16 } },
  wild: { label: "⛰️ Wild uplands (200 × 130, forest and rock)", width: 200, height: 130, water: 12, forest: 25, rock: 12, immigration: 0.0013, initial: { rabbit: 200, fox: 18 } },
  custom: { label: "✏️ Custom (set the numbers below)" },
};

const CAUSE_LABEL = {
  predation: "predation", starvation: "starvation", dehydration: "thirst", "old age": "old age", disease: "disease",
  fighting: "fights", emigrated: "emigration", removed: "removal by you",
};
