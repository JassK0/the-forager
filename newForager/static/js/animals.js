// Voxel animals drawn with one InstancedMesh per body part, coloured by genotype.
import * as THREE from "three";

const CAPACITY = 3000;

// [key, size, position, coat colour, group/animation, pivot]
// Models face +x; positions are in body units before the per-animal scale.
const MODELS = {
  rabbit: {
    scale: 0.95,
    parts: [
      ["body", [0.62, 0.4, 0.44], [0, 0.3, 0], "body"],
      ["haunch", [0.32, 0.34, 0.52], [-0.16, 0.26, 0], "body"],
      ["footL", [0.26, 0.08, 0.13], [-0.1, 0.04, 0.15], "body"],
      ["footR", [0.26, 0.08, 0.13], [-0.1, 0.04, -0.15], "body"],
      ["tail", [0.17, 0.17, 0.17], [-0.36, 0.4, 0], "tail"],
      ["stripe", [0.52, 0.05, 0.13], [0, 0.52, 0], "stripe", "stripe"],
      ["head", [0.34, 0.31, 0.32], [0.36, 0.52, 0], "head", "head"],
      ["eyeL", [0.07, 0.08, 0.04], [0.47, 0.57, 0.15], "eyes", "head"],
      ["eyeR", [0.07, 0.08, 0.04], [0.47, 0.57, -0.15], "eyes", "head"],
      ["nose", [0.06, 0.05, 0.08], [0.54, 0.49, 0], "nose", "head"],
      ["earL", [0.08, 0.36, 0.12], [0.3, 0.84, 0.085], "ears", "earL", [0.3, 0.66, 0.085]],
      ["earR", [0.08, 0.36, 0.12], [0.3, 0.84, -0.085], "ears", "earR", [0.3, 0.66, -0.085]],
    ],
    neck: [0.24, 0.45, 0],
  },
  fox: {
    scale: 1.05,
    parts: [
      ["body", [0.9, 0.34, 0.34], [0, 0.55, 0], "body"],
      ["chest", [0.24, 0.3, 0.36], [0.36, 0.5, 0], "chest"],
      ["stripe", [0.62, 0.05, 0.36], [0.08, 0.73, 0], "stripe", "stripe"],
      ["legFL", [0.11, 0.4, 0.11], [0.3, 0.2, 0.11], "legs", "legA", [0.3, 0.42, 0.11]],
      ["legFR", [0.11, 0.4, 0.11], [0.3, 0.2, -0.11], "legs", "legB", [0.3, 0.42, -0.11]],
      ["legBL", [0.11, 0.4, 0.11], [-0.32, 0.2, 0.11], "legs", "legB", [-0.32, 0.42, 0.11]],
      ["legBR", [0.11, 0.4, 0.11], [-0.32, 0.2, -0.11], "legs", "legA", [-0.32, 0.42, -0.11]],
      ["tail", [0.58, 0.2, 0.2], [-0.72, 0.6, 0], "tail", "tail", [-0.43, 0.6, 0]],
      ["tailtip", [0.2, 0.22, 0.22], [-1.08, 0.6, 0], "tailtip", "tail", [-0.43, 0.6, 0]],
      ["head", [0.36, 0.32, 0.34], [0.6, 0.72, 0], "head", "head"],
      ["muzzle", [0.24, 0.14, 0.18], [0.85, 0.65, 0], "muzzle", "head"],
      ["nose", [0.07, 0.07, 0.08], [0.98, 0.69, 0], "nose", "head"],
      ["eyeL", [0.06, 0.06, 0.04], [0.73, 0.78, 0.16], "eyes", "head"],
      ["eyeR", [0.06, 0.06, 0.04], [0.73, 0.78, -0.16], "eyes", "head"],
      ["earL", [0.1, 0.2, 0.11], [0.56, 0.96, 0.11], "ears", "earL", [0.56, 0.86, 0.11]],
      ["earR", [0.1, 0.2, 0.11], [0.56, 0.96, -0.11], "ears", "earR", [0.56, 0.86, -0.11]],
      ["food", [0.26, 0.13, 0.13], [0.98, 0.58, 0], "#b34a3c", "food"],
    ],
    neck: [0.42, 0.62, 0],
  },
};

const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);
const ZERO = new THREE.Matrix4().makeScale(0, 0, 0);

export class Animals {
  constructor(world3d, info, makeMaterial) {
    this.w3 = world3d;
    this.info = info;
    this.species = info.species.map((sp, si) => {
      const model = MODELS[sp.name] || MODELS.rabbit;
      const parts = model.parts.map(([key, size, pos, color, anim, pivot]) => {
        const geo = new THREE.BoxGeometry(...size);
        const mesh = new THREE.InstancedMesh(geo, makeMaterial(), CAPACITY);
        mesh.castShadow = true;
        mesh.receiveShadow = false;
        mesh.frustumCulled = false;
        mesh.count = 0;
        mesh.setColorAt(0, new THREE.Color());
        mesh.userData = { si, key };
        world3d.scene.add(mesh);
        const local = new THREE.Matrix4().makeTranslation(...pos);
        const p = pivot || (anim === "head" ? model.neck : null);
        return {
          key, size, pos: new THREE.Vector3(...pos), color, anim, mesh, local,
          pivot: p ? new THREE.Vector3(...p) : null,
        };
      });
      return { sp, model, parts, slots: new Map(), slotIds: [], free: [], coats: sp.coats };
    });
    this.state = new Map(); // id -> animation state
    this._m = new THREE.Matrix4();
    this._base = new THREE.Matrix4();
    this._head = new THREE.Matrix4();
    this._part = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._q2 = new THREE.Quaternion();
    this._v = new THREE.Vector3();
    this._s = new THREE.Vector3();
    this._c = new THREE.Color();
  }

  dispose() {
    for (const s of this.species) {
      for (const p of s.parts) {
        this.w3.scene.remove(p.mesh);
        p.mesh.geometry.dispose();
        p.mesh.material.dispose();
      }
    }
  }

  _slot(group, id, coat) {
    let slot = group.slots.get(id);
    if (slot === undefined) {
      slot = group.free.length ? group.free.pop() : group.slotIds.length;
      if (slot >= CAPACITY) return -1;
      group.slots.set(id, slot);
      group.slotIds[slot] = id;
      group.slotCoat = group.slotCoat || [];
      group.slotCoat[slot] = -1;
    }
    if (group.slotCoat[slot] !== coat) {
      group.slotCoat[slot] = coat;
      const colors = group.coats[coat].colors;
      for (const p of group.parts) {
        p.mesh.setColorAt(slot, this._c.set(colors[p.color] || (p.color.startsWith("#") ? p.color : colors.body)));
      }
      for (const p of group.parts) p.mesh.instanceColor.needsUpdate = true;
    }
    return slot;
  }

  // animals: [{id, si, x, y, h, act, flags, coat, size}] with interpolated positions.
  update(animals, acts, dt, time) {
    const seen = new Set();
    const burrows = [];
    const counts = this.species.map(() => 0);
    for (const a of animals) {
      const group = this.species[a.si];
      seen.add(a.id);
      const act = acts[a.act];
      let st = this.state.get(a.id);
      if (!st) {
        st = { x: a.x, y: a.y, gy: this.w3.groundY(a.x, a.y), phase: Math.random() * 6, speed: 0, seed: (a.id * 0.618) % 1 };
        this.state.set(a.id, st);
      }
      const moved = Math.hypot(a.x - st.x, a.y - st.y);
      st.x = a.x;
      st.y = a.y;
      st.speed += ((dt > 0 ? moved / dt : 0) - st.speed) * Math.min(1, dt * 8);
      st.phase += moved * (a.si === 0 ? 5.5 : 6.5);
      const targetY = this.w3.groundY(a.x, a.y);
      st.gy += (targetY - st.gy) * Math.min(1, dt * 12);

      const slot = this._slot(group, a.id, a.coat);
      if (slot < 0) continue;
      counts[a.si] = Math.max(counts[a.si], slot + 1);
      if (act === "hide" || a.flags & 32) {
        burrows.push([a.x, a.y]);
        for (const p of group.parts) p.mesh.setMatrixAt(slot, ZERO);
        continue;
      }
      this._pose(group, slot, a, act, st, time);
    }
    for (const group of this.species) {
      for (const [id, slot] of group.slots) {
        if (seen.has(id)) continue;
        group.slots.delete(id);
        group.free.push(slot);
        for (const p of group.parts) p.mesh.setMatrixAt(slot, ZERO);
        this.state.delete(id);
      }
    }
    this.species.forEach((group, si) => {
      const n = Math.max(counts[si], group.slotIds.length ? group.slotIds.length : 0);
      for (const p of group.parts) {
        p.mesh.count = Math.min(CAPACITY, n);
        p.mesh.instanceMatrix.needsUpdate = true;
      }
    });
    this.w3.setBurrows(burrows);
  }

  // A predator's capture lunge: a short leap played over the next half second.
  pounce(id) {
    const st = this.state.get(id);
    if (st) st.pounce = 1;
  }

  _pose(group, slot, a, act, st, time) {
    const isRabbit = group.sp.name === "rabbit";
    const moving = Math.min(1, st.speed / 1.2);
    const sleeping = act === "rest" || (act === "visit" && !isRabbit && moving < 0.05);
    const eating = act === "eat" || act === "drink" || (act === "forage" && moving < 0.1);
    const fleeing = act === "flee" || act === "chase";
    const scanning = (a.flags & 8) || act === "alert";
    const frozen = act === "freeze";
    const stalking = a.flags & 16;
    const digging = act === "cache" && moving < 0.05;
    const grooming = act === "groom";
    const playing = act === "play";
    const coat = group.coats[a.coat];
    const k = group.model.scale * a.size;

    // Whole-body transform: position, heading, hop or bob.
    let lift = 0, pitch = 0, crouch = 1;
    if (isRabbit) {
      const hop = Math.abs(Math.sin(st.phase));
      lift = hop * (fleeing ? 0.34 : playing ? 0.5 : 0.2) * moving;
      pitch = Math.cos(st.phase) * 0.28 * moving;
      if (!moving && act === "rest") crouch = 0.86;
      if (scanning && !moving) { pitch = 0.55; lift = 0.08; }      // rear up on the hind legs to look around
      if (grooming) { pitch = 0.4; lift = 0.04; }                   // sit up to wash
      if (frozen) { crouch = 0.78; pitch = -0.06; }                 // flatten and keep still
    } else {
      lift = Math.abs(Math.sin(st.phase * 2)) * 0.04 * moving;
      if (sleeping) lift = -0.27;
      if (stalking) { crouch = 0.8; lift = -0.08; }                 // creep low
      if (grooming) { pitch = 0.3; }
      if (playing) lift = Math.abs(Math.sin(time * 9 + st.seed * 5)) * 0.25;
    }
    if (st.pounce > 0) {
      const t = 1 - st.pounce;
      lift += Math.sin(t * Math.PI) * 0.9;
      pitch += (0.5 - t) * 0.8;
      st.pounce = Math.max(0, st.pounce - 0.035);
    }
    const v = this._v.set(a.x - this.w3.W / 2, st.gy + lift * k, a.y - this.w3.H / 2);
    this._q.setFromAxisAngle(Y, -a.h);
    this._q2.setFromAxisAngle(Z, pitch);
    this._q.multiply(this._q2);
    this._base.compose(v, this._q, this._s.set(k, k * crouch, k));

    // Head: dips to graze or drink, rests low when sleeping.
    const headPitch = digging ? -0.75 + Math.sin(time * 14 + st.seed * 10) * 0.18
      : eating ? -0.55 + Math.sin(time * 9 + st.seed * 10) * 0.08
      : grooming ? -0.3 + Math.sin(time * 11 + st.seed * 4) * 0.25
      : stalking ? -0.2 : sleeping ? -0.4 : scanning ? 0.15 : Math.sin(time * 1.3 + st.seed * 7) * 0.06;
    const neck = group.model.neck;
    this._head.makeTranslation(neck[0], neck[1], neck[2])
      .multiply(this._m.makeRotationZ(headPitch))
      .multiply(this._part.makeTranslation(-neck[0], -neck[1], -neck[2]));

    for (const p of group.parts) {
      let local = p.local;
      if ((p.anim === "stripe" && !coat.stripe) || (p.anim === "food" && !(a.flags & 64))) {
        p.mesh.setMatrixAt(slot, ZERO);
        continue;
      }
      if (p.anim === "legA" || p.anim === "legB") {
        if (sleeping) {
          p.mesh.setMatrixAt(slot, ZERO);
          continue;
        }
        const swing = Math.sin(st.phase + (p.anim === "legA" ? 0 : Math.PI)) * 0.75 * moving;
        local = this._pivoted(p, this._q2.setFromAxisAngle(Z, swing));
      } else if (p.anim === "tail") {
        const sway = sleeping ? 1.25 : Math.sin(time * 2.2 + st.seed * 9) * (act === "mark" ? 0.05 : 0.22);
        const raise = sleeping ? -0.25 : act === "mark" ? 0.9 : stalking ? -0.15 : fleeing ? 0.15 : 0.32 - 0.25 * moving;
        this._q2.setFromAxisAngle(Y, sway).multiply(new THREE.Quaternion().setFromAxisAngle(Z, raise));
        local = this._pivoted(p, this._q2);
      } else if (p.anim === "earL" || p.anim === "earR") {
        const back = fleeing || frozen ? -1.0 : scanning ? 0.15 : eating ? -0.35 : Math.sin(time * 3 + st.seed * 13) * 0.08;
        const splay = (p.anim === "earL" ? 1 : -1) * (isRabbit ? 0.12 : 0.05);
        this._q2.setFromAxisAngle(Z, back).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), splay));
        local = this._head.clone().multiply(this._pivoted(p, this._q2));
      } else if (p.anim === "head" || p.anim === "food") {
        local = this._head.clone().multiply(p.local);
      }
      this._m.multiplyMatrices(this._base, local);
      p.mesh.setMatrixAt(slot, this._m);
    }
  }

  _pivoted(p, q) {
    const pivot = p.pivot;
    const offset = this._v.copy(p.pos).sub(pivot).applyQuaternion(q).add(pivot);
    return new THREE.Matrix4().compose(offset.clone(), q.clone(), this._s.set(1, 1, 1));
  }

  // The id of the animal under the pointer, or null.
  pick(raycaster) {
    let best = null, bestD = Infinity;
    for (const group of this.species) {
      for (const p of group.parts) {
        if (!["body", "head", "haunch"].includes(p.key) || !p.mesh.count) continue;
        p.mesh.computeBoundingSphere();
        for (const hit of raycaster.intersectObject(p.mesh, false)) {
          if (hit.distance < bestD && group.slotIds[hit.instanceId] !== undefined) {
            bestD = hit.distance;
            best = group.slotIds[hit.instanceId];
          }
        }
      }
    }
    return best;
  }
}
