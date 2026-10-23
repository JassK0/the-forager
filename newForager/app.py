"""Web front end for the ecosystem simulation.

    python app.py            then open http://127.0.0.1:5000

The browser drives the clock: it asks for as many ticks as the chosen speed
calls for in real time and interpolates between the states it receives, so
animals move smoothly at 60 fps. Every run is reproducible from its seed.
"""

import random
import threading
import time

from flask import Flask, jsonify, render_template, request

from engine import World, build_config
from engine.world import EVENTS, PAINTABLE

MAX_TICKS_PER_REQUEST = 60

app = Flask(__name__)
_lock = threading.Lock()
_world = World(build_config({}), seed=1)


def _bad(message: str):
    return jsonify(error=message), 400


def _species_index(name) -> int | None:
    for i, s in enumerate(_world.species):
        if s.name == name:
            return i
    return None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/world")
def world_info():
    with _lock:
        return jsonify(_world.describe())


@app.get("/api/terrain")
def terrain():
    with _lock:
        return jsonify(_world.terrain_payload())


@app.post("/api/step")
def step():
    body = request.get_json(silent=True) or {}
    try:
        ticks = max(0, min(MAX_TICKS_PER_REQUEST, int(body.get("ticks", 1))))
        selected = body.get("selected")
        selected = None if selected is None else int(selected)
        plants = bool(body.get("plants", True))
        insights_since = body.get("insights_since")
        insights_since = None if insights_since is None else int(insights_since)
    except (TypeError, ValueError):
        return _bad("ticks, selected and insights_since must be integers")
    with _lock:
        start = time.perf_counter()
        for _ in range(ticks):
            _world.step()
        elapsed = time.perf_counter() - start
        frame = _world.frame(plants=plants, insights_since=insights_since)
        frame["ms_per_tick"] = 1000 * elapsed / ticks if ticks else None
        if selected is not None:
            frame["selected"] = _world.inspect(selected)
    return jsonify(frame)


@app.get("/api/animal/<int:uid>")
def animal(uid: int):
    with _lock:
        return jsonify(_world.inspect(uid))


@app.get("/api/insights")
def insights():
    """Everything the narrator has explained so far, oldest first."""
    with _lock:
        return jsonify(_world.narrator.log)


@app.get("/api/explain")
def explain():
    """An on-demand explanation of what is happening right now, and why."""
    with _lock:
        return jsonify(_world.narrator.report())


@app.get("/api/summary")
def summary():
    """A breakdown of the whole run so far (or of a finished run)."""
    with _lock:
        return jsonify(_world.summary())


@app.post("/api/run")
def run_settings():
    """Turn the end-on-extinction rule on or off, or resume a finished run."""
    body = request.get_json(silent=True) or {}
    with _lock:
        if body.get("resume"):
            _world.resume()
        if "end_on_extinction" in body:
            _world.end_on_extinction = bool(body["end_on_extinction"])
        return jsonify(end_on_extinction=_world.end_on_extinction, finished=_world.finished)


@app.get("/api/history")
def history():
    since = request.args.get("since", default=-1, type=int)
    with _lock:
        return jsonify(_world.history_since(since))


@app.post("/api/tool")
def tool():
    """Sandbox tools: spawn, remove, fertilise, paint terrain, events, repopulate."""
    body = request.get_json(silent=True) or {}
    kind = body.get("tool")
    try:
        x, y = float(body.get("x", 0)), float(body.get("y", 0))
        radius = max(0.5, min(25.0, float(body.get("radius", 3))))
    except (TypeError, ValueError):
        return _bad("x, y and radius must be numbers")
    with _lock:
        w = _world
        if kind == "spawn":
            si = _species_index(body.get("species"))
            if si is None:
                return _bad("unknown species")
            preset = body.get("preset") or None
            ids = w.spawn(si, x, y, int(body.get("count", 1)), preset)
            return jsonify(ids=ids)
        if kind == "remove":
            return jsonify(removed=w.remove_area(x, y, radius))
        if kind == "fertilise":
            w.fertilise(x, y, radius)
            return jsonify(ok=True)
        if kind == "paint":
            terrain_kind = body.get("terrain")
            if terrain_kind not in PAINTABLE:
                return _bad("unknown terrain type")
            return jsonify(changed=w.paint(x, y, radius, terrain_kind), version=w.terrain.version)
        if kind == "event":
            event = body.get("event")
            if event not in EVENTS:
                return _bad("unknown event")
            si = _species_index(body.get("species")) if event == "plague" else None
            if event == "plague" and si is None:
                return _bad("a plague needs a species")
            w.trigger(event, si)
            return jsonify(ok=True)
        if kind == "repopulate":
            w.repopulate()
            return jsonify(ok=True)
    return _bad("unknown tool")


@app.post("/api/lab")
def lab():
    """Genetics lab: breed, clone or gene-edit selected animals."""
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    with _lock:
        w = _world
        a = w.by_id.get(body.get("id"))
        if a is None:
            return _bad("that animal is no longer alive")
        if action == "breed":
            b = w.by_id.get(body.get("partner"))
            if b is None:
                return _bad("the partner is no longer alive")
            error = w.breed(a, b)
            return _bad(error) if error else jsonify(ok=True)
        if action == "clone":
            return jsonify(id=w.clone(a))
        if action == "edit":
            symbol = body.get("gene")
            alleles = body.get("alleles")
            gene = next((g for g in a.sp.genes if g.symbol == symbol), None)
            if gene is None or not isinstance(alleles, list) or len(alleles) != 2 \
                    or any(x not in gene.alleles for x in alleles):
                return _bad("invalid gene edit")
            w.edit_gene(a, symbol, (alleles[0], alleles[1]))
            return jsonify(w.inspect(a.id))
    return _bad("unknown lab action")


@app.post("/api/reset")
def reset():
    global _world
    body = request.get_json(silent=True) or {}
    try:
        seed = int(body.get("seed", random.randrange(1_000_000)))
        cfg = build_config(body.get("settings") or {})
    except (TypeError, ValueError) as err:
        return _bad(str(err))
    world = World(cfg, seed=seed)
    with _lock:
        _world = world
        return jsonify(_world.describe())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
