// The 3D world: voxel terrain, water, grass, trees, sky and lighting.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export const T = { WATER: 0, SAND: 1, GRASS: 2, FOREST: 3, TREE: 4, ROCK: 5 };
export const WATER_TOP = 0.78;

const lin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
export function linear(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [lin(((n >> 16) & 255) / 255), lin(((n >> 8) & 255) / 255), lin((n & 255) / 255)];
}
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// Deterministic per-cell randomness so decorations don't flicker between rebuilds.
export function hash(i, salt = 0) {
  let x = (Math.imul(i ^ (salt * 0x9e3779b9), 2654435761) >>> 0) ^ 0x5bd1e995;
  x ^= x >>> 15;
  x = Math.imul(x, 2246822519) >>> 0;
  x ^= x >>> 13;
  return (x >>> 0) / 4294967296;
}

export function decode(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

const C = {
  grassLush: linear("#5fbf4a"),
  grassDry: linear("#b9c46a"),
  dirt: linear("#a88b5c"),
  forestFloor: linear("#4d8a3c"),
  sand: linear("#ead9a0"),
  bed: linear("#8fb3a8"),
  rock: linear("#9a9aa2"),
  rockDark: linear("#7c7c86"),
  snow: linear("#f4f7fb"),
  tundra: linear("#8e9a62"),
  steppe: linear("#cdbb7a"),
  autumn: linear("#c8a24a"),
  tuftSpring: linear("#77d65a"),
  tuftSummer: linear("#4fae3c"),
  tuftAutumn: linear("#b5a548"),
  trunk: linear("#7a5233"),
  conifer: linear("#2f7044"),
  leafSpring: linear("#86d26a"),
  leafSummer: linear("#4c9f3e"),
  leafAutumn: [linear("#e48a2c"), linear("#d4562a"), linear("#e9c23c")],
  water: linear("#3f97dc"),
};

const SKY = {
  day: new THREE.Color("#9fd8ff"),
  dusk: new THREE.Color("#ffb37d"),
  night: new THREE.Color("#0e1a38"),
};

function box(w, h, d, baseAtZero = true) {
  const g = new THREE.BoxGeometry(w, h, d);
  if (baseAtZero) g.translate(0, h / 2, 0);
  return g;
}

export class World3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = SKY.day.clone();
    this.scene.fog = new THREE.Fog(SKY.day.clone(), 140, 320);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 800);
    this.camera.position.set(0, 70, 85);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.46;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 260;
    this.controls.screenSpacePanning = false;
    this.controls.target.set(0, 0, 0);

    this.hemi = new THREE.HemisphereLight(0xdff1ff, 0x6b8a4a, 1.1);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0005;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);

    const gradient = new THREE.DataTexture(new Uint8Array([110, 185, 255]), 3, 1, THREE.RedFormat);
    gradient.minFilter = gradient.magFilter = THREE.NearestFilter;
    gradient.needsUpdate = true;
    this.gradient = gradient;

    this.raycaster = new THREE.Raycaster();
    this.groups = [];
    this.season = null;
    this.snow = 0;
    this._initCursor();
    this._initParticles();
  }

  toon(color, extra = {}) {
    return new THREE.MeshToonMaterial({ color, gradientMap: this.gradient, ...extra });
  }

  // ------------------------------------------------------------------ terrain

  setTerrain(info, terrain, keepView = false) {
    for (const obj of this.groups) {
      this.scene.remove(obj);
      obj.traverse?.((o) => {
        o.geometry?.dispose();
        o.material?.dispose?.();
      });
    }
    this.groups = [];
    const W = (this.W = info.width), H = (this.H = info.height);
    this.types = decode(terrain.types);
    const hb = decode(terrain.heights);
    this.heights = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) this.heights[i] = hb[i] / 20;
    this.fertility = decode(terrain.fertility);
    this.version = terrain.version;
    // Climate: each cell's temperature offset and dryness (engine/climate.py).
    const cl = terrain.climate;
    this.uniformClimate = !cl || cl.name === "temperate";
    this.climateOff = new Float32Array(W * H);
    this.climateDry = new Float32Array(W * H);
    if (cl && !this.uniformClimate) {
      const o = decode(cl.offset), d = decode(cl.dry);
      for (let i = 0; i < W * H; i++) {
        this.climateOff[i] = (o[i] / 255) * 2.1 - 1.2;
        this.climateDry[i] = d[i] / 255;
      }
    }

    const shadowSpan = Math.max(W, H) / 2 + 12;
    Object.assign(this.sun.shadow.camera, { left: -shadowSpan, right: shadowSpan, top: shadowSpan, bottom: -shadowSpan, near: 1, far: 500 });
    this.sun.shadow.camera.updateProjectionMatrix();

    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const v = new THREE.Vector3();
    const s = new THREE.Vector3();
    const N = W * H;

    // Ground columns.
    this.ground = new THREE.InstancedMesh(box(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0xffffff }), N);
    this.ground.receiveShadow = true;
    for (let i = 0; i < N; i++) {
      const x = i % W, y = (i / W) | 0;
      m.compose(v.set(x + 0.5 - W / 2, 0, y + 0.5 - H / 2), q.identity(), s.set(1, this.heights[i], 1));
      this.ground.setMatrixAt(i, m);
      this.ground.setColorAt(i, new THREE.Color(1, 1, 1));
    }
    this.add(this.ground);

    // Water surface.
    const waterCells = [];
    for (let i = 0; i < N; i++) if (this.types[i] === T.WATER) waterCells.push(i);
    const waterMat = new THREE.MeshLambertMaterial({ color: new THREE.Color().setRGB(...C.water), transparent: true, opacity: 0.82 });
    this.water = new THREE.InstancedMesh(box(1, 0.3, 1), waterMat, Math.max(1, waterCells.length));
    this.water.count = waterCells.length;
    waterCells.forEach((cell, k) => {
      const x = cell % W, y = (cell / W) | 0;
      m.compose(v.set(x + 0.5 - W / 2, WATER_TOP - 0.3, y + 0.5 - H / 2), q.identity(), s.set(1, 1, 1));
      this.water.setMatrixAt(k, m);
      const shade = 0.88 + 0.12 * hash(cell, 3);
      this.water.setColorAt(k, new THREE.Color().setRGB(...scale([1, 1, 1], shade)));
    });
    this.add(this.water);

    // Grass tufts, one per growing cell, resized as plants grow and are eaten.
    this.tuftCells = [];
    for (let i = 0; i < N; i++) if (this.types[i] === T.GRASS || this.types[i] === T.FOREST || this.types[i] === T.SAND) this.tuftCells.push(i);
    this.tufts = new THREE.InstancedMesh(box(0.62, 1, 0.62), new THREE.MeshLambertMaterial({ color: 0xffffff }), Math.max(1, this.tuftCells.length));
    this.tufts.count = this.tuftCells.length;
    this.tufts.castShadow = false;
    this.tufts.receiveShadow = true;
    this.tuftColor = new Float32Array(this.tuftCells.length * 3);
    this.add(this.tufts);

    // Flowers dot the meadows in spring and summer.
    this.flowerCells = this.tuftCells.filter((i) => this.types[i] === T.GRASS && hash(i, 7) < 0.035);
    this.flowers = new THREE.InstancedMesh(box(0.16, 0.16, 0.16), new THREE.MeshLambertMaterial({ color: 0xffffff }), Math.max(1, this.flowerCells.length));
    this.flowers.count = this.flowerCells.length;
    const petals = ["#ff6b8a", "#ffd23f", "#ffffff", "#b388ff", "#ff9f43"];
    this.flowerCells.forEach((cell, k) => {
      this.flowers.setColorAt(k, new THREE.Color(petals[Math.floor(hash(cell, 8) * petals.length)]));
    });
    this.add(this.flowers);

    this._buildTrees(m, q, v, s);
    this._buildRocks(m, q, v, s);
    this._buildBushes(terrain.bushes ? decode(terrain.bushes) : null, m, q, v, s);
    this.burrows = new THREE.InstancedMesh(box(0.7, 0.12, 0.7), new THREE.MeshLambertMaterial({ color: "#5a3d26" }), 2048);
    this.burrows.count = 0;
    this.burrows.receiveShadow = true;
    this.add(this.burrows);
    // Carcasses left by starvation and disease, which foxes scavenge.
    this.carrion = new THREE.InstancedMesh(box(0.5, 0.14, 0.26), new THREE.MeshLambertMaterial({ color: "#efe3cf" }), 400);
    this.carrion.count = 0;
    this.add(this.carrion);
    // Food a fox has buried: little mounds of fresh earth.
    this.cacheMounds = new THREE.InstancedMesh(box(0.55, 0.16, 0.55), new THREE.MeshLambertMaterial({ color: "#7a5636" }), 400);
    this.cacheMounds.count = 0;
    this.add(this.cacheMounds);

    this.season = null;
    if (!keepView) this.frameCamera(); // a repaint keeps the camera where it is
    this.clearPaintMarks();
  }

  add(obj) {
    this.scene.add(obj);
    this.groups.push(obj);
  }

  _buildTrees(m, q, v, s) {
    const W = this.W, H = this.H;
    this.trees = [];
    for (let i = 0; i < W * H; i++) {
      if (this.types[i] !== T.TREE) continue;
      // Conifers dominate where it is cold (taiga, mountain slopes).
      const conifer = hash(i, 11) < 0.4 + Math.min(0.55, Math.max(0, -this.climateOff[i]) * 1.2);
      this.trees.push({ cell: i, conifer, size: 0.8 + 0.5 * hash(i, 12), autumn: Math.floor(hash(i, 13) * 3) });
    }
    const n = Math.max(1, this.trees.length);
    this.trunks = new THREE.InstancedMesh(box(0.28, 1, 0.28), new THREE.MeshLambertMaterial({ color: new THREE.Color().setRGB(...C.trunk) }), n);
    this.leaves = new THREE.InstancedMesh(box(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0xffffff }), n * 3);
    this.trunks.castShadow = this.leaves.castShadow = true;
    this.trunks.receiveShadow = this.leaves.receiveShadow = true;
    this.trunks.count = this.trees.length;
    this.leaves.count = this.trees.length * 3;
    this.trees.forEach((t, k) => {
      const x = t.cell % W, y = (t.cell / W) | 0;
      const cx = x + 0.5 - W / 2 + (hash(t.cell, 14) - 0.5) * 0.3, cz = y + 0.5 - H / 2 + (hash(t.cell, 15) - 0.5) * 0.3;
      const base = this.heights[t.cell];
      const sz = t.size;
      const rot = q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), hash(t.cell, 16) * Math.PI);
      const trunkH = (t.conifer ? 0.55 : 1.0) * sz;
      this.trunks.setMatrixAt(k, m.compose(v.set(cx, base, cz), rot, s.set(1, trunkH, 1)));
      const parts = t.conifer
        ? [[1.1, 0.55, trunkH], [0.8, 0.5, trunkH + 0.5 * sz], [0.48, 0.45, trunkH + 0.95 * sz]]
        : [[1.25, 0.95, trunkH], [0.8, 0.45, trunkH + 0.9 * sz], [0, 0, 0]];
      t.leafMatrices = parts.map(([w, h, y0]) => m.compose(v.set(cx, base + y0, cz), rot, s.set(w * sz, h * sz, w * sz)).clone());
      t.leafMatrices.forEach((mat, j) => this.leaves.setMatrixAt(k * 3 + j, mat));
    });
    this.add(this.trunks);
    this.add(this.leaves);
  }

  _buildRocks(m, q, v, s) {
    const W = this.W, H = this.H;
    const spots = [];
    for (let i = 0; i < W * H; i++) {
      const t = this.types[i];
      if ((t === T.ROCK && hash(i, 21) < 0.35) || (t === T.GRASS && hash(i, 22) < 0.004)) spots.push(i);
    }
    this.boulders = new THREE.InstancedMesh(box(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0xffffff }), Math.max(1, spots.length));
    this.boulders.count = spots.length;
    this.boulders.castShadow = this.boulders.receiveShadow = true;
    this.boulderCells = spots;
    spots.forEach((i, k) => {
      const x = i % W, y = (i / W) | 0;
      const sz = 0.35 + 0.45 * hash(i, 23);
      q.setFromEuler(new THREE.Euler(hash(i, 24) * 0.4, hash(i, 25) * Math.PI, hash(i, 26) * 0.4));
      this.boulders.setMatrixAt(k, m.compose(
        v.set(x + 0.5 - W / 2 + (hash(i, 27) - 0.5) * 0.4, this.heights[i] - 0.05, y + 0.5 - H / 2 + (hash(i, 28) - 0.5) * 0.4),
        q, s.set(sz * 1.2, sz * 0.8, sz)));
      this.boulders.setColorAt(k, new THREE.Color().setRGB(...mix(C.rock, C.rockDark, hash(i, 29))));
    });
    this.add(this.boulders);
  }

  // Fruiting bushes (brambles, wild apples) at woodland edges, with berries
  // that ripen from late summer and disappear as foxes eat them.
  _buildBushes(mask, m, q, v, s) {
    const W = this.W, H = this.H;
    this.bushCells = [];
    if (mask) for (let i = 0; i < W * H; i++) if (mask[i]) this.bushCells.push(i);
    const n = Math.max(1, this.bushCells.length);
    this.bushes = new THREE.InstancedMesh(box(1, 1, 1), new THREE.MeshLambertMaterial({ color: 0xffffff }), n);
    this.bushes.count = this.bushCells.length;
    this.bushes.castShadow = this.bushes.receiveShadow = true;
    const leaf = new THREE.Color("#2f6b3a"), leafDark = new THREE.Color("#244f2c");
    this.bushCells.forEach((i, k) => {
      const x = i % W, y = (i / W) | 0;
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), hash(i, 51) * Math.PI);
      this.bushes.setMatrixAt(k, m.compose(v.set(x + 0.5 - W / 2, this.heights[i] + 0.22, y + 0.5 - H / 2), q,
        s.set(0.62 + 0.2 * hash(i, 52), 0.42 + 0.18 * hash(i, 53), 0.62 + 0.2 * hash(i, 54))));
      this.bushes.setColorAt(k, leaf.clone().lerp(leafDark, hash(i, 55)));
    });
    this.add(this.bushes);
    // Up to four berries per bush, shown according to how much ripe fruit it carries.
    this.berries = new THREE.InstancedMesh(box(0.2, 0.2, 0.2), new THREE.MeshLambertMaterial({ color: 0xffffff }), n * 4);
    this.berries.count = this.bushCells.length * 4;
    const kinds = [new THREE.Color("#6b2a8c"), new THREE.Color("#e2412f"), new THREE.Color("#8e2fa0")];
    this.bushCells.forEach((i, k) => {
      for (let j = 0; j < 4; j++) this.berries.setColorAt(k * 4 + j, kinds[Math.floor(hash(i, 56) * kinds.length)]);
    });
    this.add(this.berries);
    this.updateFruit(null);
  }

  updateFruit(fruit) {
    if (!this.berries) return;
    const W = this.W, H = this.H;
    const m = new THREE.Matrix4(), zero = new THREE.Matrix4().makeScale(0, 0, 0);
    this.bushCells.forEach((i, k) => {
      const shown = fruit ? Math.round((fruit[i] / 255) * 4) : 0;
      const x = i % W, y = (i / W) | 0;
      for (let j = 0; j < 4; j++) {
        if (j >= shown) {
          this.berries.setMatrixAt(k * 4 + j, zero);
          continue;
        }
        // On the surface of the bush: one on top, the rest round the sides.
        const a = (j / 4) * Math.PI * 2 + hash(i, 60) * Math.PI;
        const r = j === 0 ? 0.05 : 0.4;
        const up = j === 0 ? 0.56 : 0.3 + 0.15 * hash(i, 64 + j);
        m.makeTranslation(x + 0.5 - W / 2 + Math.cos(a) * r, this.heights[i] + up, y + 0.5 - H / 2 + Math.sin(a) * r);
        this.berries.setMatrixAt(k * 4 + j, m);
      }
    });
    this.berries.instanceMatrix.needsUpdate = true;
    this.berries.computeBoundingSphere();
  }

  // Start close enough to see the animals; ?zoom=N in the URL scales the distance.
  frameCamera() {
    const zoom = Number(new URLSearchParams(location.search).get("zoom")) || 1;
    const d = 46 * zoom;
    this.controls.target.set(0, 1, 0);
    this.camera.position.set(-d * 0.35, d * 0.62, d * 0.72);
    this.controls.update();
  }

  // Seasonal colours and plant cover. Called whenever a new plant grid arrives.
  // Snow on one cell: the same everywhere on a temperate map, otherwise set by
  // the local temperature, and thin where it is too dry for much to fall.
  snowAt(i, env) {
    if (this.uniformClimate) return env.snow;
    const t = env.temperature + this.climateOff[i];
    return Math.min(1, Math.max(0, (-t - 0.35) / 0.4)) * (1 - 0.8 * this.climateDry[i]);
  }

  updateGround(plants, env) {
    if (!this.ground) return;
    const W = this.W, H = this.H, N = W * H;
    const f = env.year_fraction;
    const autumn = smooth(0.45, 0.6, f) * (1 - smooth(0.75, 0.82, f));
    const spring = 1 - smooth(0.12, 0.3, f) + smooth(0.92, 1.0, f);
    const col = this.ground.instanceColor.array;
    for (let i = 0; i < N; i++) {
      const t = this.types[i];
      const shade = 0.93 + 0.1 * hash(i, 1);
      const snow = this.snowAt(i, env);
      let c;
      if (t === T.WATER) c = C.bed;
      else if (t === T.SAND) c = C.sand;
      else if (t === T.ROCK) {
        c = mix(C.rock, C.rockDark, hash(i, 2));
        if (this.heights[i] > 3.4) c = mix(c, C.snow, Math.max(0.55, snow));
      } else {
        const b = plants ? Math.min(1, plants[i] / 200) : 0.6;
        let green = t === T.FOREST ? C.forestFloor : mix(C.grassDry, C.grassLush, this.fertility[i] / 255);
        if (!this.uniformClimate) {
          green = mix(green, C.tundra, Math.min(0.8, Math.max(0, -this.climateOff[i] - 0.2)));  // tundra: olive, mossy
          green = mix(green, C.steppe, Math.min(0.85, this.climateDry[i]));  // dry country: straw-coloured
        }
        c = mix(C.dirt, mix(green, C.autumn, autumn * 0.6), Math.min(1, b * 1.3));
      }
      if (t !== T.WATER) c = mix(c, C.snow, snow * (t === T.ROCK ? 0.6 : 0.92));
      col[i * 3] = c[0] * shade;
      col[i * 3 + 1] = c[1] * shade;
      col[i * 3 + 2] = c[2] * shade;
    }
    this.ground.instanceColor.needsUpdate = true;

    // Tufts grow with biomass; colour follows the season.
    const tuft = mix(mix(C.tuftSummer, C.tuftSpring, spring), C.tuftAutumn, autumn);
    const mat = this.tufts.instanceMatrix.array;
    const tc = this.tuftColor;
    for (let k = 0; k < this.tuftCells.length; k++) {
      const i = this.tuftCells[k];
      const b = plants ? plants[i] / 255 : 0.5;
      // Only lush cells sprout a tuft; each is scattered and turned so the
      // meadow doesn't read as a grid.
      const snow = this.snowAt(i, env);
      const hgt = b < 0.3 || snow > 0.7 ? 0 : 0.04 + 0.3 * b;
      const x = i % W, y = (i / W) | 0;
      const o = k * 16;
      const jx = (hash(i, 4) - 0.5) * 0.5, jz = (hash(i, 5) - 0.5) * 0.5;
      const w = hgt > 0 ? 0.55 + 0.35 * hash(i, 17) : 0;
      const a = hash(i, 18) * Math.PI, ca = Math.cos(a) * w, sa = Math.sin(a) * w;
      mat[o] = ca; mat[o + 1] = 0; mat[o + 2] = -sa; mat[o + 3] = 0;
      mat[o + 4] = 0; mat[o + 5] = hgt; mat[o + 6] = 0; mat[o + 7] = 0;
      mat[o + 8] = sa; mat[o + 9] = 0; mat[o + 10] = ca; mat[o + 11] = 0;
      mat[o + 12] = x + 0.5 - W / 2 + jx; mat[o + 13] = this.heights[i]; mat[o + 14] = y + 0.5 - H / 2 + jz; mat[o + 15] = 1;
      const shade = 0.85 + 0.25 * hash(i, 6);
      const c = mix(tuft, C.snow, snow * 0.8);
      tc[k * 3] = c[0] * shade;
      tc[k * 3 + 1] = c[1] * shade;
      tc[k * 3 + 2] = c[2] * shade;
    }
    this.tufts.instanceMatrix.needsUpdate = true;
    if (!this.tufts.instanceColor) this.tufts.instanceColor = new THREE.InstancedBufferAttribute(tc, 3);
    else this.tufts.instanceColor.array.set(tc);
    this.tufts.instanceColor.needsUpdate = true;
    this.tufts.computeBoundingSphere();

    // Flowers bloom in spring and early summer.
    const bloom = env.season === "spring" || (env.season === "summer" && f < 0.4);
    const fm = new THREE.Matrix4();
    this.flowerCells.forEach((i, k) => {
      const x = i % W, y = (i / W) | 0;
      const b = plants ? plants[i] / 255 : 0.5;
      const on = bloom && b > 0.35 && this.snowAt(i, env) < 0.2;
      fm.makeScale(on ? 1 : 0, on ? 1 : 0, on ? 1 : 0);
      fm.setPosition(x + 0.5 - W / 2 + (hash(i, 9) - 0.5) * 0.5, this.heights[i] + 0.05 + 0.42 * b, y + 0.5 - H / 2 + (hash(i, 10) - 0.5) * 0.5);
      this.flowers.setMatrixAt(k, fm);
    });
    this.flowers.instanceMatrix.needsUpdate = true;
    this.flowers.computeBoundingSphere();

    this._seasonTrees(env, autumn, spring);
  }

  _seasonTrees(env, autumn, spring) {
    const key = `${env.season}:${Math.round(env.temperature * 10)}:${Math.round(autumn * 10)}`;
    if (key === this.season || !this.trees.length) return;
    this.season = key;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    const color = new THREE.Color();
    this.trees.forEach((t, k) => {
      for (let j = 0; j < 3; j++) {
        let c;
        let visible = !(j === 2 && !t.conifer);
        const snow = this.snowAt(t.cell, env);
        if (t.conifer) c = mix(C.conifer, C.snow, snow * (j === 2 ? 0.8 : 0.55));
        else {
          c = mix(mix(C.leafSummer, C.leafSpring, spring), C.leafAutumn[t.autumn], autumn);
          if (env.season === "winter" && snow > 0.3) visible = false; // bare deciduous trees
        }
        this.leaves.setMatrixAt(k * 3 + j, visible ? t.leafMatrices[j] : zero);
        this.leaves.setColorAt(k * 3 + j, color.setRGB(...scale(c, 0.9 + 0.2 * hash(t.cell, 30 + j))));
      }
    });
    this.leaves.instanceMatrix.needsUpdate = true;
    this.leaves.instanceColor.needsUpdate = true;
    this.boulders.instanceColor && this.boulderCells.forEach((i, k) => {
      const c = mix(mix(C.rock, C.rockDark, hash(i, 29)), C.snow, this.snowAt(i, env) * 0.7);
      this.boulders.setColorAt(k, color.setRGB(...c));
    });
    if (this.boulders.instanceColor) this.boulders.instanceColor.needsUpdate = true;
  }

  groundY(x, y) {
    const cx = Math.min(this.W - 1, Math.max(0, x | 0)), cy = Math.min(this.H - 1, Math.max(0, y | 0));
    return this.heights[cy * this.W + cx];
  }

  setBurrows(list) {
    const m = new THREE.Matrix4();
    const n = Math.min(list.length, 2048);
    for (let k = 0; k < n; k++) {
      const [x, y] = list[k];
      m.makeTranslation(x - this.W / 2, this.groundY(x, y) - 0.02, y - this.H / 2);
      this.burrows.setMatrixAt(k, m);
    }
    this.burrows.count = n;
    this.burrows.instanceMatrix.needsUpdate = true;
    this.burrows.computeBoundingSphere();
  }

  setCaches(list) {
    if (!this.cacheMounds || !list) return;
    const m = new THREE.Matrix4();
    const n = Math.min(list.length, 400);
    for (let k = 0; k < n; k++) {
      const [x, y] = list[k];
      m.makeTranslation(x - this.W / 2, this.groundY(x, y), y - this.H / 2);
      this.cacheMounds.setMatrixAt(k, m);
    }
    this.cacheMounds.count = n;
    this.cacheMounds.instanceMatrix.needsUpdate = true;
    this.cacheMounds.computeBoundingSphere();
  }

  setCarrion(list) {
    if (!this.carrion || !list) return;
    const m = new THREE.Matrix4();
    const n = Math.min(list.length, 400);
    for (let k = 0; k < n; k++) {
      const [x, y] = list[k];
      m.makeRotationY(hash(Math.round(x * 31 + y * 17), 40) * Math.PI);
      m.setPosition(x - this.W / 2, this.groundY(x, y), y - this.H / 2);
      this.carrion.setMatrixAt(k, m);
    }
    this.carrion.count = n;
    this.carrion.instanceMatrix.needsUpdate = true;
    this.carrion.computeBoundingSphere();
  }

  // ------------------------------------------------------------------ sky

  updateSky(env, hours, time) {
    const { sunrise, sunset } = env;
    const light = smooth(sunrise - 0.75, sunrise + 0.75, hours) * (1 - smooth(sunset - 0.75, sunset + 0.75, hours));
    const dusk = Math.max(
      1 - Math.min(1, Math.abs(hours - sunrise) / 1.4),
      1 - Math.min(1, Math.abs(hours - sunset) / 1.4),
    );
    const sky = SKY.night.clone().lerp(SKY.day, light).lerp(SKY.dusk, dusk * 0.55);
    this.scene.background.copy(sky);
    this.scene.fog.color.copy(sky);

    const span = Math.max(this.W || 100, this.H || 100);
    let theta, intensity, color;
    if (light > 0.02) {
      theta = Math.min(Math.PI, Math.max(0, ((hours - sunrise) / (sunset - sunrise)) * Math.PI));
      intensity = 0.4 + 2.0 * light;
      color = new THREE.Color("#fff1d6").lerp(new THREE.Color("#ffb36b"), dusk * 0.7);
    } else {
      // Moonlight: a dim blue key light so night is dark but readable.
      const nightLen = 24 - (sunset - sunrise);
      const into = ((hours - sunset + 24) % 24) / nightLen;
      theta = Math.min(Math.PI, Math.max(0, into * Math.PI));
      intensity = 0.55;
      color = new THREE.Color("#9db4ff");
    }
    this.sun.color.copy(color);
    this.sun.intensity = intensity;
    this.sun.position.set(-Math.cos(theta) * span, Math.sin(theta) * span * 0.9 + span * 0.15, span * 0.35);
    this.hemi.intensity = 0.35 + 0.85 * light;
    this.hemi.color.set(light > 0.3 ? "#dff1ff" : "#6f86c8");
    if (this.water) this.water.position.y = Math.sin(time * 1.3) * 0.02;
    return light;
  }

  // ------------------------------------------------------------------ cursor, picking, particles

  _initCursor() {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.92, 1, 48),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, depthTest: false, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.renderOrder = 10;
    ring.visible = false;
    this.cursor = ring;
    this.scene.add(ring);
    this.paintMarks = new THREE.InstancedMesh(new THREE.CircleGeometry(1, 24).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false }), 256);
    this.paintMarks.count = 0;
    this.paintMarks.renderOrder = 9;
    this.paintMarks.frustumCulled = false;
    this.scene.add(this.paintMarks);
  }

  // A painted spot shown straight away, until the rebuilt terrain replaces it.
  paintMark(x, y, radius, color) {
    const k = this.paintMarks.count;
    if (k >= this.paintMarks.instanceMatrix.count) return;
    const m = new THREE.Matrix4().makeScale(radius, 1, radius);
    m.setPosition(x - this.W / 2, this.groundY(x, y) + 0.12, y - this.H / 2);
    this.paintMarks.setMatrixAt(k, m);
    this.paintMarks.setColorAt(k, new THREE.Color(color));
    this.paintMarks.count = k + 1;
    this.paintMarks.instanceMatrix.needsUpdate = true;
    if (this.paintMarks.instanceColor) this.paintMarks.instanceColor.needsUpdate = true;
  }

  clearPaintMarks() {
    if (this.paintMarks) this.paintMarks.count = 0;
  }

  showCursor(point, radius, color) {
    if (!point) {
      this.cursor.visible = false;
      return;
    }
    this.cursor.visible = true;
    this.cursor.material.color.set(color);
    this.cursor.scale.set(radius, radius, radius);
    this.cursor.position.set(point.x - this.W / 2, this.groundY(point.x, point.y) + 0.08, point.y - this.H / 2);
  }

  ndc(event) {
    const r = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
  }

  // World (x, y) under the pointer, or null.
  pickGround(event) {
    if (!this.ground) return null;
    this.raycaster.setFromCamera(this.ndc(event), this.camera);
    const hit = this.raycaster.intersectObject(this.ground, false)[0];
    if (!hit) return null;
    return { x: hit.point.x + this.W / 2, y: hit.point.z + this.H / 2 };
  }

  _initParticles() {
    const n = 600;
    this.particles = new THREE.InstancedMesh(box(0.14, 0.14, 0.14, false), new THREE.MeshBasicMaterial({ color: 0xffffff }), n);
    this.particles.frustumCulled = false;
    this.particles.count = 0;
    this.pool = [];
    for (let i = 0; i < n; i++) this.pool.push({ life: 0 });
    this.particles.setColorAt(0, new THREE.Color());
    this.scene.add(this.particles);
  }

  burst(x, y, color, count = 10, up = 2.5) {
    const base = this.groundY(x, y) + 0.4;
    for (const p of this.pool) {
      if (count <= 0) break;
      if (p.life > 0) continue;
      const a = Math.random() * Math.PI * 2, sp = 0.8 + Math.random() * 1.6;
      Object.assign(p, {
        life: 0.9 + Math.random() * 0.5, x: x - this.W / 2, y: base, z: y - this.H / 2,
        vx: Math.cos(a) * sp, vy: up * (0.6 + Math.random() * 0.8), vz: Math.sin(a) * sp, color: new THREE.Color(color),
      });
      count--;
    }
  }

  updateParticles(dt) {
    const m = new THREE.Matrix4();
    let k = 0;
    for (const p of this.pool) {
      if (p.life <= 0) continue;
      p.life -= dt;
      p.vy -= 6 * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
      const s = Math.max(0, Math.min(1, p.life * 1.6));
      m.makeScale(s, s, s).setPosition(p.x, p.y, p.z);
      this.particles.setMatrixAt(k, m);
      this.particles.setColorAt(k, p.color);
      k++;
    }
    this.particles.count = k;
    this.particles.instanceMatrix.needsUpdate = true;
    if (this.particles.instanceColor) this.particles.instanceColor.needsUpdate = true;
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
