"""Web front end for the ecosystem simulation.

    python app.py            then open http://127.0.0.1:5000

The simulation runs on its own clock, in a background thread, at the speed the
browser asks for (or as fast as the computer allows, for "Max" and for
skipping ahead). The browser never waits for it: a few times a second it asks
for the latest state and moves the animals smoothly between states at 60 fps.
Speed never changes what happens, only how quickly it is computed: every tick
is the same calculation, so every run is reproducible from its seed.
"""

import json
import random
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from itertools import count

from flask import Flask, Response, jsonify, render_template, request

import ai
from engine import World, build_config
from engine.report import evidence as build_evidence, ledger_series
from engine.world import EVENTS, PAINTABLE

MAX_TICKS_PER_REQUEST = 60
BASE_TPS = 4  # ticks per second at 1x speed
CHUNK_SECONDS = 0.02  # the clock holds the world for at most this long at a time
MAX_SKIP_DAYS = 12 * 50  # longest single skip ahead: 50 years
AI_KINDS = ("summary", "extinction", "question")
KEEP_ANSWERS = 30  # finished AI answers kept, so asking the same thing twice costs nothing
KEEP_EVIDENCE = 12  # briefs kept so the user can see exactly what an answer was based on

app = Flask(__name__)
_lock = threading.Lock()
_waiting = 0  # requests waiting for the world, so the clock can step aside for them
_waiting_lock = threading.Lock()
_world = World(build_config({}), seed=1)
_world_serial = count(1)
_world_id = next(_world_serial)
_answers: OrderedDict = OrderedDict()  # (world, tick, kind, species, question) -> answer text
_evidence: OrderedDict = OrderedDict()  # id -> brief (text, png, meta)
_evidence_ids = count(1)


@contextmanager
def locked():
    """Exclusive access to the world for a request (the clock yields to waiting requests)."""
    global _waiting
    with _waiting_lock:
        _waiting += 1
    _lock.acquire()
    with _waiting_lock:
        _waiting -= 1
    try:
        yield
    finally:
        _lock.release()


class Clock:
    """Steps the world in a background thread at a target rate, flat out for "max",
    or flat out until a target tick when skipping ahead."""

    def __init__(self):
        self.running = False
        self.tps: float | None = 2 * BASE_TPS  # None: as fast as possible
        self.skip_from: int | None = None
        self.skip_to: int | None = None
        self.resume_after = False
        self.achieved = 0.0  # measured ticks per second
        self.wake = threading.Event()
        self.thread: threading.Thread | None = None

    def ensure_started(self) -> None:
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, name="simulation-clock", daemon=True)
            self.thread.start()

    def status(self, w: World) -> dict:
        skip = None
        if self.skip_to is not None:
            skip = {"from": self.skip_from, "to": self.skip_to}
        return {
            "running": self.running or skip is not None,
            "speed": None if self.tps is None else self.tps / BASE_TPS,
            "achieved": round(self.achieved / BASE_TPS, 2),
            "tick": w.tick,
            "skip": skip,
            "finished": bool(w.finished),
        }

    def _run(self) -> None:
        last = time.perf_counter()
        owed = 0.0
        window_ticks, window_start = 0, last
        while True:
            if not self.running and self.skip_to is None:
                self.achieved = 0.0
                self.wake.wait(0.5)
                self.wake.clear()
                last = window_start = time.perf_counter()
                owed, window_ticks = 0.0, 0
                continue
            now = time.perf_counter()
            flat_out = self.skip_to is not None or self.tps is None
            if flat_out:
                owed = float("inf")
            else:
                # Ticks owed for the time that has passed, without piling up a backlog.
                owed = min(owed + (now - last) * self.tps, 0.25 * self.tps + 1.0)
            last = now
            if owed < 1.0:
                time.sleep(min(0.02, (1.0 - owed) / self.tps))
                continue
            done = 0
            with _lock:
                w = _world
                start = time.perf_counter()
                while owed >= 1.0 and not w.finished and time.perf_counter() - start < CHUNK_SECONDS:
                    if self.skip_to is not None and w.tick >= self.skip_to:
                        break
                    w.step()
                    done += 1
                    owed -= 1.0
                finished = bool(w.finished)
                reached = self.skip_to is not None and w.tick >= self.skip_to
            if finished:
                self.running = False
                self.skip_to = self.skip_from = None
            elif reached:
                self.skip_to = self.skip_from = None
                self.running = self.resume_after
            window_ticks += done
            elapsed = time.perf_counter() - window_start
            if elapsed >= 0.5:
                self.achieved = window_ticks / elapsed
                window_ticks, window_start = 0, time.perf_counter()
            if _waiting:
                time.sleep(0.001)  # let a waiting request in


_clock = Clock()


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
    with locked():
        return jsonify(_world.describe())


@app.get("/api/terrain")
def terrain():
    with locked():
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
    with locked():
        start = time.perf_counter()
        for _ in range(ticks):
            _world.step()
        elapsed = time.perf_counter() - start
        frame = _world.frame(plants=plants, insights_since=insights_since)
        frame["ms_per_tick"] = 1000 * elapsed / ticks if ticks else None
        if selected is not None:
            frame["selected"] = _world.inspect(selected)
    return jsonify(frame)


@app.post("/api/frame")
def frame():
    """The latest state, without stepping (the clock steps the world)."""
    body = request.get_json(silent=True) or {}
    try:
        selected = body.get("selected")
        selected = None if selected is None else int(selected)
        insights_since = body.get("insights_since")
        insights_since = None if insights_since is None else int(insights_since)
    except (TypeError, ValueError):
        return _bad("selected and insights_since must be integers")
    with locked():
        out = _world.frame(plants=bool(body.get("plants", True)), insights_since=insights_since)
        if selected is not None:
            out["selected"] = _world.inspect(selected)
        out["clock"] = _clock.status(_world)
    return jsonify(out)


@app.route("/api/clock", methods=["GET", "POST"])
def clock():
    """Play, pause and speed (x1 = 4 ticks a second, or "max"), and skipping ahead.

    POST {"running": true, "speed": 5}, {"speed": "max"}, {"skip_days": 12} or {"cancel_skip": true}.
    """
    body = (request.get_json(silent=True) or {}) if request.method == "POST" else {}
    try:
        if "speed" in body:
            speed = body["speed"]
            _clock.tps = None if speed == "max" else max(0.05, min(1000.0, float(speed))) * BASE_TPS
        if "running" in body:
            _clock.running = bool(body["running"])
        if "skip_days" in body:
            days = max(0.0, min(float(MAX_SKIP_DAYS), float(body["skip_days"])))
            with locked():
                tick, tpd = _world.tick, _world.cfg.ticks_per_day
            if _clock.skip_to is None:
                _clock.resume_after = _clock.running
            _clock.skip_from, _clock.skip_to = tick, tick + int(days * tpd)
        if body.get("cancel_skip"):
            _clock.skip_to = _clock.skip_from = None
            _clock.running = _clock.resume_after
    except (TypeError, ValueError):
        return _bad("speed and skip_days must be numbers")
    if _clock.running or _clock.skip_to is not None:
        _clock.ensure_started()
        _clock.wake.set()
    with locked():
        return jsonify(_clock.status(_world))


@app.get("/api/animal/<int:uid>")
def animal(uid: int):
    with locked():
        return jsonify(_world.inspect(uid))


@app.get("/api/insights")
def insights():
    """Everything the narrator has explained so far, oldest first."""
    with locked():
        return jsonify(_world.narrator.log)


@app.get("/api/explain")
def explain():
    """An on-demand explanation of what is happening right now, and why."""
    with locked():
        return jsonify(_world.narrator.report())


@app.get("/api/summary")
def summary():
    """A breakdown of the whole run so far (or of a finished run)."""
    with locked():
        return jsonify(_world.summary())


@app.get("/api/ledger")
def ledger():
    """Day-by-day births, deaths by cause, breeding adults and food per head."""
    with locked():
        return jsonify(ledger_series(_world))


# ---------------------------------------------------------------- the AI analyst

@app.get("/api/ai/status")
def ai_status():
    conf = ai.settings()
    with locked():
        extinctions = [{"species": e["species"], "day": e["day"], "recolonised_day": e["recolonised_day"]}
                       for e in _world.ledger.extinctions]
        finished = bool(_world.finished)
    return jsonify(configured=bool(conf["key"]), model=conf["model"], env_file=str(ai.ENV_PATH),
                   extinctions=extinctions, finished=finished)


@app.post("/api/ai")
def ai_ask():
    """Stream an AI analysis as lines of JSON: {"meta"}, then {"text"} pieces, then {"done"} or {"error"}.

    POST {"kind": "summary"}, {"kind": "extinction", "species": "fox"} or {"kind": "question", "question": "..."}.
    """
    body = request.get_json(silent=True) or {}
    kind = body.get("kind", "summary")
    species = body.get("species")
    question = str(body.get("question") or "").strip()[:600]
    if kind not in AI_KINDS:
        return _bad("unknown kind of analysis")
    if kind == "question" and not question:
        return _bad("ask a question first")
    conf = ai.settings()
    if not conf["key"]:
        return jsonify(error=f"No AI key yet: paste your Claude API key into {ai.ENV_PATH} (ANTHROPIC_API_KEY=...).",
                       setup=True), 400
    with locked():
        if kind == "extinction" and not any(e["species"] == species for e in _world.ledger.extinctions):
            return _bad(f"no {species} extinction has been recorded in this run")
        key = (_world_id, _world.tick, kind, species if kind == "extinction" else None, question.lower())
        brief = build_evidence(_world, kind, species)
        finished = bool(_world.finished)
    eid = next(_evidence_ids)
    _evidence[eid] = brief
    while len(_evidence) > KEEP_EVIDENCE:
        _evidence.popitem(last=False)
    cached = _answers.get(key)

    def line(obj) -> str:
        return json.dumps(obj) + "\n"

    def generate():
        yield line({"meta": {**brief["meta"], "evidence": eid, "model": conf["model"], "cached": cached is not None}})
        if cached is not None:
            yield line({"text": cached})
            yield line({"done": True, "cached": True})
            return
        parts = []
        try:
            for what, value in ai.analyse(conf, kind, brief, species, question, finished):
                if what == "text":
                    parts.append(value)
                    yield line({"text": value})
                else:
                    if value.get("stop_reason") != "max_tokens":
                        _answers[key] = "".join(parts)
                        while len(_answers) > KEEP_ANSWERS:
                            _answers.popitem(last=False)
                    yield line({"done": True, **value})
        except ai.AIError as err:
            yield line({"error": str(err)})

    return Response(generate(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/ai/evidence/<int:eid>.txt")
def ai_evidence_text(eid: int):
    """Exactly what the AI was shown for an answer (the map image is separate)."""
    brief = _evidence.get(eid)
    if brief is None:
        return _bad("that evidence is no longer kept")
    return Response(brief["text"], mimetype="text/plain; charset=utf-8")


@app.get("/api/ai/evidence/<int:eid>.png")
def ai_evidence_map(eid: int):
    brief = _evidence.get(eid)
    if brief is None:
        return _bad("that evidence is no longer kept")
    return Response(brief["png"], mimetype="image/png")


@app.post("/api/run")
def run_settings():
    """Turn the end-on-extinction rule on or off, or resume a finished run."""
    body = request.get_json(silent=True) or {}
    with locked():
        if body.get("resume"):
            _world.resume()
        if "end_on_extinction" in body:
            _world.end_on_extinction = bool(body["end_on_extinction"])
        return jsonify(end_on_extinction=_world.end_on_extinction, finished=_world.finished)


@app.get("/api/history")
def history():
    since = request.args.get("since", default=-1, type=int)
    with locked():
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
    with locked():
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
    with locked():
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
    global _world, _world_id
    body = request.get_json(silent=True) or {}
    try:
        seed = int(body.get("seed", random.randrange(1_000_000)))
        cfg = build_config(body.get("settings") or {})
    except (TypeError, ValueError) as err:
        return _bad(str(err))
    world = World(cfg, seed=seed)
    with locked():
        _world = world
        _world_id = next(_world_serial)
        return jsonify(_world.describe())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
