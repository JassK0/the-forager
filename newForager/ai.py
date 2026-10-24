"""The AI analyst: Claude reads the run's records and explains what happened, and why.

The API key lives in `.env` next to this file (see `.env.example`). The file is
read on every request, so a key pasted in while the server runs works at once.
Requests go straight to the Claude Messages API with the standard library, so
there is nothing extra to install. Answers are streamed back as they are written.
"""

import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

ENV_PATH = pathlib.Path(__file__).with_name(".env")
DEFAULT_MODEL = "claude-opus-5-5"
API_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
TIMEOUT = 120  # seconds to wait for the next piece of an answer
MAX_TOKENS = {"summary": 3000, "extinction": 2500, "question": 1800}
WORDS = {"summary": 450, "extinction": 400, "question": 300}
PLACEHOLDERS = {"", "your-key-here", "sk-ant-...", "paste-your-key-here"}


class AIError(Exception):
    """Something went wrong talking to the AI, with a message fit to show the user."""


def read_env(path: pathlib.Path) -> dict:
    """KEY=value lines; blank lines and # comments are skipped, quotes are stripped."""
    out = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        out[key] = value
    return out


def settings() -> dict:
    """The key, model and endpoint, from .env first and then the environment."""
    env = read_env(ENV_PATH)

    def get(name: str, default: str = "") -> str:
        return env.get(name) or os.environ.get(name) or default

    key = get("ANTHROPIC_API_KEY")
    if key in PLACEHOLDERS:
        key = ""
    return {"key": key, "model": get("AI_MODEL", DEFAULT_MODEL), "base_url": get("ANTHROPIC_BASE_URL", API_URL).rstrip("/")}


# ---------------------------------------------------------------- prompts

SYSTEM = """You are the resident ecologist of Forager, an individual-based simulation of wild rabbits and red foxes. Every animal in it has its own genes, body, needs and a neural-network brain that evolves and learns; nothing is scripted, so populations rise, crash and sometimes die out for real ecological reasons. The person asking is curious but not a scientist. They want to understand what happened in their run and, above all, why.

You are given the simulation's own records: tables, logs, a region table and a map image. Treat them as the only source of truth.

How to reason:
- Explain causes, not just outcomes. Separate the immediate cause (what the last animals died of) from the underlying cause (why losses were not replaced: no adults of one sex, outside the breeding season, a food or prey crash, winter, a shortage of territories, emigration, the user's sandbox actions, and so on). Causes often chain together: say how one led to the next.
- Back every causal claim with specific evidence: the day or days and the numbers, for example "by day 41 only 1 adult female was left, and no young were born after day 38". Copy numbers exactly as recorded. Never invent animals, numbers or events.
- Test the obvious alternative explanations against the records and say briefly why each does or doesn't hold.
- If the records can't settle something, say so plainly.
- Mind the calendar: a year is 12 days and each season lasts 3 days (days 1-3 of each year are spring, 4-6 summer, 7-9 autumn, 10-12 winter). Day numbers count from the start of the run, so day 14 is day 2 of year 2 (spring). Check the breeding seasons before blaming a lack of births on anything else.
- "Emigrated" animals did not die: they left because they found no territory, but they are lost to this population all the same.

How to write:
- Plain, warm English for a non-scientist. Explain any ecology term in a few words the first time you use it.
- Markdown: short "## " headings, short paragraphs and bullet lists; bold the key numbers and days. No tables and no code blocks.
- Be concise and concrete. Don't repeat the question or these instructions."""


def _plural(name: str) -> str:
    return name + ("es" if name.endswith(("s", "x", "z", "ch", "sh")) else "s")


def task(kind: str, meta: dict, species: str | None = None, question: str | None = None, finished: bool = False) -> str:
    words = WORDS[kind]
    if kind == "extinction":
        f = meta.get("focus") or {}
        return (f"Explain why the {_plural(species or 'animal')} died out on day {f.get('day', '?')}. Use these sections:\n"
                "## Verdict: 1-2 sentences.\n"
                "## How it unfolded: a short timeline (bullets that start with the day) from the start of the decline "
                "to the last death.\n"
                "## The evidence: the decisive facts, each with its day and numbers.\n"
                "## Other explanations: the obvious alternatives, and whether the records support them.\n"
                "## Confidence: high, medium or low, and why.\n"
                f"Aim for about {words} words.")
    if kind == "question":
        return (f"The user asks: \"{question}\"\n"
                "Answer it directly in the first one or two sentences, then give the evidence from the records. "
                f"Use headings only if they help. Aim for at most {words} words.")
    gone = ""
    for e in meta.get("extinctions", []):
        gone += (f"\n## Why the {_plural(e['species'])} died out (day {e['day']}): the immediate cause, the underlying "
                 "cause and the evidence.")
    return ("Write the " + ("end-of-run report" if finished else "report on the run so far") + ". Use these sections:\n"
            "## In short: 2-3 sentences on the most important thing that happened, and why.\n"
            "## What happened: the run's phases in order, with days and numbers (booms, crashes, recoveries, "
            "extinctions).\n"
            "## Why: the main drivers, each proven with evidence from the records." + gone + "\n"
            "## How they evolved, and why: for each species, the biggest changes in inherited traits and in "
            "behaviour (learned or evolved), then why they happened. Use the selection tables (which animals "
            "bred and which died of what, compared with the average) to show what natural selection favoured, "
            "and say when the numbers are too small to tell selection from chance. If little changed, say so "
            "and why." + (" This run is in Super animals mode: make this the longest section, and also say "
                          "what the animals learned and passed on, and how strong the alphas became."
                          if meta.get("super") else "") + "\n"
            f"Aim for about {words + 150 * len(meta.get('extinctions', [])) + (200 if meta.get('super') else 100)} words.")


def messages(evidence: dict, instruction: str) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": evidence["text"]},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.b64encode(evidence["png"]).decode("ascii")}},
        {"type": "text", "text": instruction},
    ]}]


# ---------------------------------------------------------------- the API

def _explain_http_error(err: urllib.error.HTTPError, model: str) -> str:
    try:
        detail = json.loads(err.read().decode("utf-8", "replace")).get("error", {}).get("message", "")
    except (ValueError, AttributeError, OSError):
        detail = ""
    if err.code == 401:
        return "The API key in .env was rejected (401). Check ANTHROPIC_API_KEY, then try again."
    if err.code == 403:
        return f"This API key isn't allowed to do that (403). {detail}".strip()
    if err.code == 404:
        return f"The model \"{model}\" wasn't found (404). Set AI_MODEL in .env to a model your key can use."
    if err.code == 429:
        return "Rate limited by the AI service (429). Wait a moment and try again."
    if err.code in (500, 502, 503, 529):
        return f"The AI service is busy or unavailable ({err.code}). Try again in a minute."
    return f"The AI request failed ({err.code}): {detail or err.reason}"


def stream(conf: dict, system: str, msgs: list, max_tokens: int):
    """Yield ("text", piece) as the answer is written, then ("done", info)."""
    body = json.dumps({"model": conf["model"], "max_tokens": max_tokens, "system": system,
                       "messages": msgs, "stream": True}).encode("utf-8")
    req = urllib.request.Request(f"{conf['base_url']}/v1/messages", data=body, method="POST", headers={
        "content-type": "application/json", "x-api-key": conf["key"], "anthropic-version": API_VERSION,
        "accept": "text/event-stream",
    })
    try:
        res = urllib.request.urlopen(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as err:
        raise AIError(_explain_http_error(err, conf["model"])) from None
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        reason = getattr(err, "reason", err)
        raise AIError(f"Couldn't reach the AI service ({reason}). Check your internet connection.") from None
    info = {"model": conf["model"], "input_tokens": None, "output_tokens": None, "stop_reason": None}
    with res:
        try:
            for raw in res:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                kind = event.get("type")
                if kind == "message_start":
                    message = event.get("message", {})
                    info["model"] = message.get("model", info["model"])
                    info["input_tokens"] = message.get("usage", {}).get("input_tokens")
                elif kind == "content_block_delta" and event.get("delta", {}).get("type") == "text_delta":
                    yield "text", event["delta"].get("text", "")
                elif kind == "message_delta":
                    info["stop_reason"] = event.get("delta", {}).get("stop_reason")
                    info["output_tokens"] = event.get("usage", {}).get("output_tokens")
                elif kind == "error":
                    raise AIError(f"The AI service reported an error: {event.get('error', {}).get('message', 'unknown')}")
        except (TimeoutError, OSError) as err:
            raise AIError(f"The connection to the AI service dropped ({err}). Try again.") from None
    yield "done", info


def analyse(conf: dict, kind: str, evidence: dict, species: str | None = None, question: str | None = None,
            finished: bool = False):
    """Stream an analysis of the run: ("text", piece)... then ("done", info)."""
    instruction = task(kind, evidence["meta"], species, question, finished)
    yield from stream(conf, SYSTEM, messages(evidence, instruction), MAX_TOKENS[kind])
