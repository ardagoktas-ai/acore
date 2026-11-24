"""
Acore Studio - research prototype.

A person watches my Acore engine live in a micro-universe and asks it
questions. Every answer is generated from the engine's own internal model by
a rule-based language layer (intent patterns + templates). No LLM, no network.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _engine_path():
    p = os.environ.get("ACORE_PATH")
    if p:
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(HERE, "..", "..", "stage1_acore"))


ENGINE_PATH = _engine_path()
if ENGINE_PATH not in sys.path:
    sys.path.insert(0, ENGINE_PATH)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from acore_v5_0 import ACOREEngine  # noqa: E402
from worlds import SYMBOLS, build_world, list_worlds as _list_worlds  # noqa: E402

SUCCESS_REWARD = 100.0
DIR_NAMES = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
NAMES = {"#": "wall", ".": "floor", "A": "my body", "a": "my body carrying a key",
         "K": "key", "D": "door", "L": "lava", "F": "flicker wall", "C": "chest",
         "S": "switch", "G": "gate", "X": "exit", "T": "timer switch", "B": "decoy button",
         "P": "button", "Q": "button", "R": "button"}
PRESS_SYMBOLS = "PQRSTB"

CAPABILITIES = [
    "what opens the door / how do I open it / why does it open",
    "what opens the gate (cause and delay)",
    "where am I / what is my body / what do my actions do",
    "what am I unsure about (open hypotheses)",
    "is moving up/down/left/right (or action N) reversible",
    "what have I learned (stats and laws)",
    "help",
]

INTENTS = [
    ("help", [r"\bhelp\b", r"what can (?:you|i) (?:ask|do|answer)", r"\bcommands?\b",
              r"how do(?:es)? (?:this|it) work"]),
    ("door", [r"\bdoors?\b"]),
    ("gate", [r"\bgates?\b"]),
    ("reversible", [r"revers", r"\bundo", r"take (?:it )?back", r"\bpermanent",
                    r"can (?:i|you) go back"]),
    ("unsure", [r"\bunsure\b", r"uncertain", r"not sure", r"\bdoubt", r"hypothes",
                r"(?:don'?t|do not) (?:you |i )?know", r"\bconfus", r"ambigu", r"open questions?"]),
    ("body", [r"\bbody\b", r"where am i\b", r"where are you\b", r"\b(?:my|your) position\b",
              r"\bwhat am i\b", r"\bwho am i\b", r"which (?:cell|colou?r|symbol|one) am i",
              r"\bmyself\b", r"\byourself\b", r"\bself\b", r"\bmotor", r"what do (?:my|your) actions",
              r"\bcontrols?\b"]),
    ("learned", [r"\blearn(?:ed|t)?\b", r"\blaws?\b", r"\bbeliefs?\b", r"\bbelieve\b",
                 r"\bstat(?:s|istics)\b", r"\bsummary\b", r"\bprogress\b", r"\bdiscover",
                 r"what do (?:you|i) know\s*(?:so far)?\s*\??$", r"\bfigured out\b"]),
]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, float, str)):
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def _join(parts):
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _verb(sym):
    return "pressed" if sym in PRESS_SYMBOLS else "touched"


class Studio:
    def __init__(self, world="compound", seed=0):
        self.seed = seed
        self.agent = ACOREEngine(n_actions=4, seed=seed)
        self._palettes = []
        self.tick = 0
        self.episodes = 0
        self._in_episode = False
        self._load(world, seed)

    # ------------------------------------------------------------ world
    def _load(self, world, seed):
        env, pal, dirs = build_world(world, seed)
        self.world, self.world_seed = world, seed
        self.env, self.palette, self.dirs = env, pal, dirs
        if pal not in self._palettes:
            self._palettes.append(dict(pal))
        self.obs = env.reset()
        self._in_episode = False
        self._ep_steps = 0

    def _close_episode(self):
        if self._in_episode:
            self.agent.end_episode()
            self._in_episode = False
            self.episodes += 1

    def list_worlds(self):
        return _list_worlds()

    def new_world(self, world, seed=0):
        self._close_episode()
        self._load(world, seed)

    def reset_agent(self):
        self.agent = ACOREEngine(n_actions=4, seed=self.seed)
        self._palettes = [dict(self.palette)]
        self.tick = 0
        self.episodes = 0
        self.obs = self.env.reset()
        self._in_episode = False
        self._ep_steps = 0

    # ------------------------------------------------------------ running
    def run_episodes(self, k, max_steps=150):
        self._close_episode()
        out = []
        for _ in range(int(k)):
            obs = self.env.reset()
            steps = hazards = 0
            done = False
            while not done and steps < max_steps:
                a = self.agent.act(obs)
                nxt, r, done = self.env.step(a)
                hazards += r <= -10
                self.agent.observe(obs, a, nxt, r, done)
                obs, steps = nxt, steps + 1
            self.agent.end_episode()
            self.obs = obs
            self.tick += steps
            self.episodes += 1
            out.append({"steps": int(steps), "success": bool(done), "hazards": int(hazards)})
        return out

    def step(self, max_steps=150):
        if not self._in_episode:
            self.obs = self.env.reset()
            self._in_episode = True
            self._ep_steps = 0
        obs = self.obs
        a = self.agent.act(obs)
        nxt, r, done = self.env.step(a)
        self.agent.observe(obs, a, nxt, r, done)
        self.obs = nxt
        self._ep_steps += 1
        self.tick += 1
        if done or self._ep_steps >= max_steps:
            self.agent.end_episode()
            self._in_episode = False
            self.episodes += 1
        return {"action": int(a), "reward": float(r), "done": bool(done), "grid": self.grid()}

    def grid(self):
        inv = {v: k for k, v in self.palette.items()}
        return [[inv.get(v, "?") for v in row] for row in self.obs]

    def snapshot(self):
        try:
            stats = self.agent.stats()
        except Exception as err:
            stats = {"error": str(err)}
        stats = _jsonable(stats)
        if not isinstance(stats, dict):
            stats = {"summary": stats}
        return {"world": self.world, "tick": int(self.tick), "stats": stats, "laws": self.laws()}

    # ------------------------------------------------------------ model readers
    def _inv(self):
        """Engine value -> symbol. Earliest palette wins: the engine keeps the
        values of the world in which a role was first learned."""
        inv = {}
        for pal in reversed(self._palettes):
            inv.update({v: k for k, v in pal.items()})
        return inv

    def _sym_of_ctx(self, ctx, inv):
        try:
            return inv.get(dict(ctx).get((0, 1)), "?")
        except Exception:
            return "?"

    def _literals(self, formula, inv):
        lits = []
        for e, pol in formula:
            lits.append((self._sym_of_ctx(e[0], inv), bool(pol)))
        return sorted(lits, key=lambda l: (not l[1], l[0]))

    @staticmethod
    def _formula_text(lits, opens_if_true):
        text = "&".join(sorted((s if p else "~" + s) for s, p in lits))
        return text if opens_if_true else "NOT(" + text + ")"

    @staticmethod
    def _clause(lits):
        pos = [s for s, p in lits if p]
        neg = [s for s, p in lits if not p]
        parts = [f"{s} was {_verb(s)}" for s in pos]
        for s in neg:
            if pos and _verb(s) == _verb(pos[-1]):
                parts.append(f"{s} was not")
            else:
                parts.append(f"{s} was not {_verb(s)}")
        return _join(parts)

    def _latent_laws(self):
        """One entry per hidden-cause group: its MAP formula, verbalised."""
        inv = self._inv()
        out = []
        groups = getattr(self.agent, "latent_groups", {}) or {}
        for ctx, g in groups.items():
            try:
                target = self._sym_of_ctx(ctx, inv)
                hyp = next(h for h in g["hyps"] if h[0] == g["map"])
                o_true = hyp[1]
                opens_if_true = o_true is not None and o_true[2] == SUCCESS_REWARD
                lits = self._literals(g["map"], inv)
                n_hyps = len(g["hyps"])
                try:
                    relevant = bool(self.agent._policy_relevant(g))
                except Exception:
                    relevant = n_hyps > 1
            except Exception:
                continue
            name = NAMES.get(target, "object")
            clause = self._clause(lits)
            if not lits:
                sentence = f"the {name} ({target}) opens regardless of what was pressed"
            elif opens_if_true:
                sentence = f"the {name} ({target}) opens if {clause}"
            else:
                sentence = f"the {name} ({target}) opens unless {clause}"
            conf = "high" if n_hyps == 1 else ("medium" if not relevant else "low")
            out.append({"target": target, "sentence": sentence, "n_hyps": n_hyps,
                        "formula": self._formula_text(lits, opens_if_true), "confidence": conf,
                        "relevant": relevant})
        return out

    def _links(self):
        """Effect (old symbol, new symbol) -> list of (cause symbol, lag)."""
        inv = self._inv()
        out = {}
        try:
            groups = self.agent.link_groups()
        except Exception:
            return out
        for (old, new), hyps in groups.items():
            key = (inv.get(old, "?"), inv.get(new, "?"))
            items = []
            for h in hyps:
                try:
                    items.append((self._sym_of_ctx(h[0][0], inv), int(h[1])))
                except Exception:
                    continue
            out[key] = items
        return out

    @staticmethod
    def _link_sentence(old, new, cause, lag):
        name = NAMES.get(old, "object")
        effect = f"the {name} ({old}) opens" if new == "." else f"{old} turns into {new}"
        verb = _verb(cause)
        if lag == 0:
            return f"{effect} in the same step that {cause} is {verb}"
        unit = "step" if lag == 1 else "steps"
        return f"{effect} {lag} {unit} after {cause} is {verb}"

    def _motor(self):
        mm = getattr(self.agent, "motor_map", None)
        out = {}
        for a in range(4):
            d = None
            try:
                d = mm.get(a) if isinstance(mm, dict) else mm[a]
            except Exception:
                d = None
            if d is not None:
                out[a] = DIR_NAMES.get(tuple(d), str(d))
        return out

    def _body_known(self):
        try:
            return bool(self.agent.body_known())
        except Exception:
            return False

    def _engine_grid(self):
        raw = tuple(map(tuple, self.obs))
        try:
            return self.agent._translate(raw)
        except Exception:
            return self.agent._grid(self.obs)

    def laws(self):
        inv = self._inv()
        out = []
        if self._body_known():
            body = sorted(inv.get(v, "?") for v in getattr(self.agent, "self_values", set()))
            out.append(f"my body is shown as {' or '.join(body)}")
            motor = self._motor()
            if motor:
                out.append("motor map: " + ", ".join(f"action {a} = {d}" for a, d in sorted(motor.items())))
        for law in self._latent_laws():
            out.append(f"{law['sentence']} [MAP {law['formula']}, {law['n_hyps']} surviving]")
        for (old, new), items in self._links().items():
            if len(items) == 1:
                out.append(self._link_sentence(old, new, *items[0]))
            elif items:
                cands = ", ".join(f"{c} (delay {l})" for c, l in items)
                out.append(f"the {NAMES.get(old, 'object')} ({old}) change has rival causes: {cands}")
        vol = getattr(self.agent, "volatile", None)
        if vol:
            out.append(f"{len(vol)} cell(s) are noise to me (they change at random)")
        return out

    # ------------------------------------------------------------ language layer
    def ask(self, text):
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        intent = None
        if t:
            for name, pats in INTENTS:
                if any(re.search(p, t) for p in pats):
                    intent = name
                    break
        handler = {"help": self._a_help, "door": self._a_door, "gate": self._a_gate,
                   "reversible": self._a_reversible, "unsure": self._a_unsure,
                   "body": self._a_body, "learned": self._a_learned}.get(intent)
        if handler is None:
            return self._none("That question is outside my world. I am a research prototype "
                              "that only knows its micro-universe.")
        return handler(t)

    @staticmethod
    def _r(answer, source, confidence):
        return {"answer": answer, "source": source, "confidence": confidence}

    def _none(self, lead):
        return self._r(lead + " I can answer: " + "; ".join(CAPABILITIES) + ".", "none", "none")

    def _a_help(self, _t):
        return self._r("Ask me about my own model of this world: " + "; ".join(CAPABILITIES)
                       + ". Every answer is read from my internal state; if I have not learned "
                       "something I say so.", "help", "high")

    def _has(self, sym):
        return any(sym in row for row in self.env.layout)

    def _a_door(self, t):
        laws = [l for l in self._latent_laws() if l["target"] == "D"]
        if not laws:
            where = "" if self._has("D") else " There is no door in this world."
            return self._none("I have not learned a hidden rule for the door (D) yet." + where
                              + " Run some episodes and ask again.")
        law = laws[0]
        lead = "My best explanation is that " if re.search(r"\bwhy\b", t) else "From my hidden-cause model, "
        rivals = law["n_hyps"] - 1
        tail = (" No rival formula survives." if rivals == 0 else
                f" {rivals} rival formula(s) still survive"
                + (", but none of them would change what I do." if not law["relevant"] else "."))
        return self._r(f"{lead}{law['sentence']} (MAP formula {law['formula']}).{tail}",
                       "latent_groups", law["confidence"])

    def _a_gate(self, _t):
        items = self._links().get(("G", "."), [])
        if not items:
            where = "" if self._has("G") else " There is no gate in this world."
            return self._none("I have not learned what opens the gate (G) yet." + where)
        if len(items) == 1:
            return self._r("From my causal links, " + self._link_sentence("G", ".", *items[0]) + ".",
                           "link_groups", "high")
        cands = "; ".join(self._link_sentence("G", ".", c, l) for c, l in items)
        return self._r(f"I am not sure yet. {len(items)} hypotheses survive: {cands}.",
                       "link_groups", "low")

    def _a_body(self, _t):
        if not self._body_known():
            return self._none("I have not found my body yet: I need to move and watch which cell follows me.")
        inv = self._inv()
        body = sorted(inv.get(v, "?") for v in getattr(self.agent, "self_values", set()))
        pos = None
        try:
            pos = self.agent._locate_self(self._engine_grid())
        except Exception:
            pos = None
        where = f" I am at row {pos[0]}, column {pos[1]}." if pos is not None else \
            " I cannot locate myself in the current grid."
        motor = self._motor()
        mtxt = (" My actions: " + ", ".join(f"{a} moves me {d}" for a, d in sorted(motor.items())) + ".") \
            if motor else ""
        return self._r(f"My body is the cell shown as {' or '.join(body)}.{where}{mtxt}",
                       "self_values", "high")

    def _a_reversible(self, t):
        motor = self._motor()
        wanted = None
        m = re.search(r"\baction\s*(\d)\b", t)
        if m and int(m.group(1)) < 4:
            wanted = int(m.group(1))
        else:
            for word, d in (("up", "up"), ("north", "up"), ("down", "down"), ("south", "down"),
                            ("left", "left"), ("west", "left"), ("right", "right"), ("east", "right")):
                if re.search(r"\b" + word + r"\b", t):
                    wanted = next((a for a, n in motor.items() if n == d), None)
                    break
        labels = {}
        for a in range(4):
            try:
                labels[a] = str(self.agent.reversibility(self.obs, a))
            except Exception:
                labels[a] = "unknown"

        def name(a):
            return f"action {a}" + (f" ({motor[a]})" if a in motor else "")

        if wanted is not None:
            lab = labels[wanted]
            conf = "low" if lab == "unknown" else "high"
            return self._r(f"From where I stand now, {name(wanted)} is {lab}.", "reversibility", conf)
        listing = "; ".join(f"{name(a)}: {labels[a]}" for a in range(4))
        return self._r("I judge reversibility for my actions from where I stand now: " + listing + ".",
                       "reversibility", "medium")

    def _a_unsure(self, _t):
        inv = self._inv()
        lines = []
        try:
            for ctx, hyps in self.agent.latent_hypotheses().items():
                if len(hyps) <= 1:
                    continue
                target = self._sym_of_ctx(ctx, inv)
                forms = []
                for f, o_true in hyps:
                    opens = o_true is not None and o_true[2] == SUCCESS_REWARD
                    forms.append(self._formula_text(self._literals(f, inv), opens))
                lines.append(f"what opens {target}: {', '.join(sorted(forms))}")
        except Exception:
            pass
        for (old, new), items in self._links().items():
            if len(items) > 1:
                lines.append(f"what changes {old}: " + ", ".join(f"{c} with delay {l}" for c, l in items))
        try:
            n_struct = len(self.agent.structure_report())
        except Exception:
            n_struct = 0
        if n_struct > 1:
            lines.append(f"{n_struct} structural mappings of this world to known roles survive")
        if not lines:
            return self._r("I have no open hypotheses: every hidden-cause and causal-link group I hold "
                           "has a single survivor (or I have not formed any yet).",
                           "latent_hypotheses", "high")
        return self._r("I am unsure about: " + "; ".join(lines) + ".", "latent_hypotheses", "medium")

    def _a_learned(self, _t):
        snap = self.snapshot()
        stats = snap["stats"]
        stxt = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:8]) if stats else "no stats"
        laws = snap["laws"]
        ltxt = "; ".join(laws) if laws else "no laws yet"
        return self._r(f"After {self.tick} steps in {self.episodes} episode(s): {stxt}. Laws I believe: {ltxt}.",
                       "stats", "high" if laws else "low")


__all__ = ["Studio", "SYMBOLS"]
