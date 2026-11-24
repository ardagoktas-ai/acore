"""
A-CORE v1.9 — Multi-source synthesis of causal roles
====================================================
Input contract (unchanged): a raw 2D grid of integers, a scalar reward, a
done flag and n opaque motor commands. No names, no symbolic states.

Inherited: body discovery, canonical frame, one-shot schemas, volatility,
same-step and delayed causal links, working memory, compound hidden causes,
version spaces, value of information, experiment design, learned
reversibility, exhaust-before-commit, science episodes, role profiles, the
analogical lens, whole-structure hypotheses, elimination by prediction,
structural information gain and incremental imagination (v1.8).

NEW in v1.9
-----------
1. RETESTS ARE PER WORLD
   Retest bookkeeping (which context was tried under which memory state) is
   reset on entering a new world: a law borrowed by analogy is a prediction,
   not an experience. A pessimistic analogy ("this door needs a lever, and
   there is none") is therefore still tested once instead of trusted forever.

2. RE-ATTRIBUTION OF EVIDENCE
   When a structure dies, every occurrence THIS world produced under the
   borrowed roles is moved to the roles the new structure sees (with its
   memory snapshot translated). The source world's law is not polluted and
   what was learned here is not lost: aliasing at the new role induces its
   own hidden-cause group immediately.

3. PRESENCE-SCOPED EVIDENCE
   Every occurrence is tagged with the world it came from; for each world
   the agent knows which roles EXIST there. A literal about an object absent
   from a world can neither confirm nor refute anything in that world.

4. ROLE SYNTHESIS (multi-source analogy)
   On entering a new world, two known roles with the same profile and the
   same pair of outcomes, learned in DIFFERENT worlds, are offered as one
   kind of object. Their evidence is pooled with presence-scoped semantics;
   the vocabulary is the union of the ingredients of the laws each world
   taught. If a formula explains every world, a UNIFIED role is created
   (e.g. "door iff lever AND NOT alarm" from a lever world and an alarm
   world). Where the worlds leave the orientation open, ENABLING laws are
   preferred (the conjunction gates the special outcome). Schemas of the
   unified role are learned from augmented evidence so that plain evaluation
   equals scoped evaluation.

5. THE UNIFIED ROLE IN THE STRUCTURE SPACE
   It competes with its parts as a mapping target; its distance to a new
   object is that of its nearest part minus an Occam bonus, and mapping it
   makes its parts not "missing" (they are the same kind). Elimination by
   prediction still guards it: a wrong synthesis dies like any analogy.
"""

import itertools
import math
import os
import random
from collections import Counter

try:                       # optional: vectorised planning (pure-Python fallback below)
    import numpy as np
except ImportError:        # pragma: no cover
    np = None
from bisect import bisect_right
from dataclasses import dataclass, field

KEEP = "KEEP"
SELF = "SELF"
VOID = None
MEM = "M"

FORWARD = (0, 1)
HERE = (0, 0)


def canon(offset, motor):
    perp = (motor[1], -motor[0])
    return (offset[0] * perp[0] + offset[1] * perp[1],
            offset[0] * motor[0] + offset[1] * motor[1])


def uncanon(coff, motor):
    perp = (motor[1], -motor[0])
    a, b = coff
    return (a * perp[0] + b * motor[0], a * perp[1] + b * motor[1])


def add(p, o):
    return (p[0] + o[0], p[1] + o[1])


def sub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def formula_true(formula, memory):
    """A formula is a frozenset of literals (event, polarity): a conjunction."""
    return all((e in memory) == pol for e, pol in formula)


def event_target(e):
    """The object an event acted on (the value in front of the body)."""
    for k, v in e[0]:
        if k == (0, 1):
            return v
    return None


def formula_true_scoped(formula, memory, present):
    """Like formula_true, but a literal about an object that did not EXIST in the
    world where the memory was recorded can neither confirm nor refute anything:
    it is treated as satisfied."""
    return all((e in memory) == pol or event_target(e) not in present
               for e, pol in formula)


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


_ANY = object()

_NPU = None          # handle of the optional native library (code, not results)


def _native():
    """Lazily load my native kernels; None when unavailable or disabled."""
    global _NPU
    if _NPU is None:
        _NPU = False
        if np is not None and os.environ.get("ACORE_NATIVE", "1") != "0":
            try:
                from npu_driver import npu as mod
                mod.acore_value_iteration(np.zeros((1, 1)), np.zeros((1, 1)),
                                          np.zeros((1, 1), dtype=np.int64), np.zeros(1))
                _NPU = mod
            except Exception:
                _NPU = False
    return _NPU or None


class _LazyGrid:
    """Copy-on-write grid for imagination: only rows that change are copied."""
    __slots__ = ("base", "rows")

    def __init__(self, base):
        self.base = base
        self.rows = {}

    def get(self, r, c):
        row = self.rows.get(r)
        return row[c] if row is not None else self.base[r][c]

    def set(self, r, c, v):
        row = self.rows.get(r)
        if row is None:
            row = self.rows[r] = list(self.base[r])
        row[c] = v

    def freeze(self):
        if not self.rows:
            return self.base
        out = list(self.base)
        for r, row in self.rows.items():
            out[r] = tuple(row)
        return tuple(out)


@dataclass
class Schema:
    context: frozenset
    effect: frozenset
    self_delta: tuple
    reward: float
    done: bool
    exemplars: list = field(default_factory=list)
    uses: int = 1
    last_tick: int = 0
    sid: int = 0              # creation order: stable tie-break (keeps imagined edges reusable)

    @property
    def signature(self):
        return (self.effect, self.self_delta, self.reward, self.done)


@dataclass
class CausalLink:
    cause: tuple            # event descriptor
    lag: int                # steps between cause and effect
    old: object
    new: object
    applicable: int = 1
    joint: int = 1
    cells: set = field(default_factory=set)

    def precision(self):
        return self.joint / self.applicable

    def status(self, effect_count):
        recall = self.joint / max(1, effect_count)
        if self.precision() < 0.9 or recall < 0.9:
            return "rejected"
        return "candidate" if self.joint == 1 else "confirmed"


class ACOREEngine:
    VERSION = "5.0"

    def __init__(self, n_actions, gamma=0.95, curiosity_scale=6.0,
                 info_scale=12.0, action_cost=0.5, max_plan_nodes=4000,
                 max_rules=96, max_exemplars=512, volatility_threshold=2,
                 decay=0.995, wm_capacity=9, max_latents=4, max_occurrences=16,
                 max_lag=5, vocab_size=6, superstition_weight=0.25,
                 max_superstition_events=2, science_weight=4.0,
                 reversibility_aware=True, science=True, analogy=True,
                 analogy_threshold=2.0, new_world_fraction=0.3, struct_info=True,
                 max_struct_hyps=16, struct_candidates=3, mult_penalty=3.0,
                 unmapped_cost=8.0, synthesis=True, synthesis_bonus=1.0, seed=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.curiosity_scale = curiosity_scale
        self.info_scale = info_scale
        self.action_cost = action_cost
        self.max_plan_nodes = max_plan_nodes
        self.max_rules = max_rules
        self.max_exemplars = max_exemplars
        self.volatility_threshold = volatility_threshold
        self.decay = decay
        self.wm_capacity = wm_capacity
        self.max_latents = max_latents
        self.max_occurrences = max_occurrences
        self.max_lag = max_lag
        self.vocab_size = vocab_size
        self.superstition_weight = superstition_weight
        self.max_superstition_events = max_superstition_events
        self.science_weight = science_weight
        self.reversibility_aware = reversibility_aware   # ablation switch
        self.science = science                           # ablation switch
        # analogy: True = map roles, False = only re-find the body, None = off
        self.analogy_enabled = analogy
        self.analogy_threshold = analogy_threshold
        self.new_world_fraction = new_world_fraction
        self.struct_info = struct_info
        self.synthesis = synthesis
        self.synthesis_bonus = synthesis_bonus
        self.synthetic_roles = {}      # unified value -> {"parts", "law", "ctx"}
        self.world_id = 0
        self.world_present = {}        # world id -> internal values seen there
        self.card_world = {}           # role value -> world where its card was made
        self.occ_world = {}            # ctx -> [(signature, snapshot, world id)]
        self.max_world_occurrences = 24
        self.min_rival_weight = 0.01
        self.max_rivals = 6
        self.max_struct_hyps = max_struct_hyps
        self.struct_candidates = struct_candidates
        self.mult_penalty = mult_penalty
        self.unmapped_cost = unmapped_cost
        self.struct_hyps = []
        self._struct_map = 0
        self._struct_targets = []
        self.structures_eliminated = 0
        self.structure_switches = 0
        self.reattributed = 0
        self.lens = {}                 # raw value -> internal (known) value
        self.analogical = set()        # raw values bound by analogy
        self.tentative = set()         # bindings chosen among equally good rivals
        self.revoked = set()
        self.analogy_report = {}
        self.analogies_revoked = 0
        self._row_memo = {}
        self._flags_memo = {}
        self._flags_memo_version = -1
        self._schema_counter = 0
        self._info_version = 0
        self._lens_epoch = 0
        self._dirty_targets = set()
        self._dirty_any_body = set()
        self._dirty_all = True
        self._suspend_dirty = False
        self.edges_reused = 0
        self.edges_imagined = 0
        self.worlds_entered = 0
        self.known_values = set()
        self._rows_seen = set()
        self.value_profiles = {}       # internal value -> role profile
        self.floor_counts = {}
        self.wall_value = None
        self.body_start = None
        self._world_pending = False
        self._probe_blocked = set()
        self._probe_buffer = []
        self._episode_start = True
        self._lens_event_sources = {}
        self._info_cache = {}
        self._plan_tracked_cache = frozenset()
        self._plan_tracked_version = -1
        self._labels = {}
        self._graph = None
        self._evaluation = None
        self.graph_builds = 0
        self._novelty_cache = {}
        self.blocked_contexts = set()  # contexts that failed (blocked, not harmful)
        self.tried2 = {}               # (context, coarse memory) -> tries
        self.memkeys_seen = {}         # context -> coarse memory states it was tested in
        self.retests_pending = False
        self.deferred_commits = 0
        self.science_mode = False
        self.science_budget = {}       # ctx -> remaining science episodes
        self.science_episodes = 0
        self._scheduling = False
        self._episode_success = False
        self.rng = random.Random(seed)

        self.energy = 100.0
        self.max_energy = 100.0

        # Body
        self.motor_map = {}
        self.self_values = set()
        self.babble_tried = set()
        self.pending = []
        # Abstraction
        self.volatile = set()
        self.exo_changes = {}
        # Local schemas
        self.schemas = []
        self.exemplars = {}
        self.occurrences = {}
        # Remote (possibly delayed) links
        self.links = {}              # (cause, lag, old, new) -> CausalLink
        self.links_by_cause = {}     # cause -> set(keys)
        self.effect_counts = {}
        self._map_links = frozenset()
        self._map_links_by_cause = {}
        self._groups = {}
        self._open_groups = {}
        self.history = {}            # tick -> event descriptor (current episode)
        self.event_ticks = {}        # descriptor -> [ticks]
        self.experiment = None       # {"until": tick} while a delayed test runs
        self.experiments_run = 0
        # Working memory + latent version space
        self.wm = set()
        self.event_count = {}
        self.event_last = {}
        self.latents = []            # MAP hidden causes (drive the schemas)
        self.latent_groups = {}      # ctx -> {"vocab", "outcomes", "hyps", "map"}
        self.tried = {}

        self.model_version = 0
        self.curiosity_version = 0
        self.tick = 0
        self.compute_cost = 0
        self.replans = 0
        self.pruned_total = 0
        self.last_q = {}
        self._cache = None
        self._index = None
        self._index_version = -1
        self._cindex = None
        self._cindex_version = -1

    # ------------------------------------------------------------------ #
    # Grid helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _grid(obs):
        return tuple(tuple(row) for row in obs)

    @staticmethod
    def _val(grid, pos):
        r, c = pos
        if r < 0 or c < 0:
            return VOID
        try:
            return grid[r][c]
        except IndexError:
            return VOID

    @staticmethod
    def _diff(g0, g1):
        out = set()
        for r, (row0, row1) in enumerate(zip(g0, g1)):
            if row0 != row1:                       # rows are tuples: fast compare
                out.update((r, c) for c, (a, b) in enumerate(zip(row0, row1)) if a != b)
        return out

    def _locate_self(self, grid):
        """The unique cell showing one of my body values, or None (zero or many).
        Row-level counting keeps the per-cell work inside C."""
        sv = self.self_values
        if not sv:
            return None
        found = None
        for r, row in enumerate(grid):
            for v in sv:
                n = row.count(v)
                if n:
                    if n > 1 or found is not None:
                        return None
                    found = (r, row.index(v))
        return found

    def body_known(self):
        return bool(self.self_values) and len(self.motor_map) == self.n_actions

    def _mask(self, grid):
        """Remove volatile (noise) cells from a grid: they are not world state."""
        if not self.volatile:
            return grid
        return tuple(tuple(VOID if (r, c) in self.volatile else v
                           for c, v in enumerate(row))
                     for r, row in enumerate(grid))

    @staticmethod
    def _key(grid, pos, mem, pending):
        return grid, pos, mem, pending

    def _local_context(self, grid, p, motor):
        # in the canonical frame my cell is always HERE and the target FORWARD
        t = (p[0] + motor[0], p[1] + motor[1])
        vol = self.volatile
        items = []
        if p not in vol:
            items.append((HERE, self._val(grid, p)))
        if t not in vol:
            items.append((FORWARD, self._val(grid, t)))
        return frozenset(items)

    def _value_counts(self, grid):
        vol = self.volatile
        if not vol:
            counts = Counter()
            for row in grid:
                counts.update(row)
            return counts
        vol_rows = {}
        for r, c in vol:
            vol_rows.setdefault(r, set()).add(c)
        counts = Counter()
        for r, row in enumerate(grid):
            cs = vol_rows.get(r)
            if cs is None:
                counts.update(row)
            else:
                counts.update(v for c, v in enumerate(row) if c not in cs)
        return counts

    def _remote_values(self, grid, local):
        excluded = {}
        for r, c in local:
            excluded.setdefault(r, set()).add(c)
        for r, c in self.volatile:
            excluded.setdefault(r, set()).add(c)
        width = len(grid[0])
        out = set()
        for r, row in enumerate(grid):
            cs = excluded.get(r)
            if cs is None:
                out.update(row[:width])
            else:
                out.update(v for c, v in enumerate(row[:width]) if c not in cs)
        return out

    # ------------------------------------------------------------------ #
    # Body discovery
    # ------------------------------------------------------------------ #
    def _discover_body(self, g0, action, g1):
        changed = self._diff(g0, g1)
        if not self.self_values:
            counts = {}
            for row in g0:
                for v in row:
                    counts[v] = counts.get(v, 0) + 1
            candidates = []
            for p in changed:
                v = self._val(g0, p)
                if counts[v] != 1:
                    continue
                for q in changed:
                    if q != p and self._val(g1, q) == v and self._val(g1, p) != v \
                            and self._val(g0, q) != v:
                        candidates.append((v, p, q))
            if len(candidates) == 1:
                v, p, q = candidates[0]
                self.self_values.add(v)
                self.motor_map[action] = sub(q, p)
            return
        if action in self.motor_map:
            return
        p = self._locate_self(g0)
        if p is None:
            return
        for q in changed:
            if q != p and self._val(g1, q) == self._val(g0, p):
                self.motor_map[action] = sub(q, p)
                return

    def _track_self(self, g0, g1, p, motor):
        hit = self._locate_self(g1)
        if hit is not None:
            return hit
        t = add(p, motor)
        if self._val(g1, t) != self._val(g0, t) and self._val(g1, p) != self._val(g0, p):
            q = t
        else:
            q = p
        self.self_values.add(self._val(g1, q))
        return q

    # ------------------------------------------------------------------ #
    # Working memory
    # ------------------------------------------------------------------ #
    def _tracked(self):
        tracked = set()
        for f in self.latents:
            tracked |= {e for e, _pol in f}
        for g in self.latent_groups.values():
            tracked |= set(g["vocab"])
        return tracked

    def _plan_tracked(self):
        """Events imagination must keep track of: those of the MAP formulas, plus
        the events of competing formulas that can still change the policy."""
        if self._plan_tracked_version == self.model_version:
            return self._plan_tracked_cache
        tracked = set()
        for f in self.latents:
            tracked |= {e for e, _pol in f}
        for g in self.latent_groups.values():
            for f, _ot, _of in self._relevant_rivals(g):
                tracked |= {e for e, _pol in f}
            tracked |= self._superstition_events(g)
        self._plan_tracked_cache = frozenset(tracked)
        self._plan_tracked_version = self.model_version
        return self._plan_tracked_cache

    def _relevant_rivals(self, g):
        """Value of information: a competing formula matters only if it disagrees
        with the MAP formula on the memory the agent would create to exploit it
        (the MAP's positive literals)."""
        if len(g["hyps"]) < 2:
            return []
        map_hyp = next(h for h in g["hyps"] if h[0] == g["map"])
        exploit_memory = self._exploit_memory(map_hyp)
        map_outcome = map_hyp[1] if formula_true(map_hyp[0], exploit_memory) else map_hyp[2]
        return [h for h in g["hyps"]
                if (h[1] if formula_true(h[0], exploit_memory) else h[2]) != map_outcome]

    @staticmethod
    def _exploit_memory(hyp):
        """Smallest memory (over the formula's own events) for which the formula
        predicts its better-rewarded outcome: what exploiting it would create."""
        f, o_true, o_false = hyp
        want = o_true if o_true[2] >= o_false[2] else o_false
        events = sorted({e for e, _pol in f}, key=str)
        subsets = [frozenset(e for k, e in enumerate(events) if mask >> k & 1)
                   for mask in range(1 << len(events))]
        for m in sorted(subsets, key=len):
            if (o_true if formula_true(f, m) else o_false) == want:
                return m
        return frozenset()

    def _superstition_events(self, g):
        """Events of policy-IRRELEVANT rivals (possible superstitions) that are
        cheap to keep in imagination: at most `max_superstition_events`."""
        if self._policy_relevant(g) or len(g["hyps"]) < 2:
            return set()
        if not self.science_mode and not self._scheduling:
            return set()                    # tracked only during science episodes
        map_events = {e for e, _pol in g["map"]}
        extra = []
        for f, _ot, _of in sorted(g["hyps"], key=lambda h: self._complexity(h[0])):
            for e, _pol in f:
                if e not in map_events and e not in extra:
                    extra.append(e)
        return set(extra[:self.max_superstition_events])

    def _policy_relevant(self, g):
        return bool(self._relevant_rivals(g))

    def _flags_from(self, snapshot, latents=None):
        latents = self.latents if latents is None else latents
        return frozenset(i for i, f in enumerate(latents) if formula_true(f, snapshot))

    def _mem_items(self, flags, ctx):
        """Memory flags enter ONLY the schemas of the context whose aliasing
        created them; every other context stays memory-free (otherwise flags
        fragment unrelated schemas and turn known actions into 'unknown')."""
        own = {g["map"] for c, g in self.latent_groups.items() if c == ctx}
        return frozenset(((MEM, i), i in flags) for i, f in enumerate(self.latents) if f in own)

    def _is_salient(self, e):
        n = self.event_count.get(e, 0)
        return n <= 2 or n / max(1, self.tick) < 0.1

    def _salient(self, snapshot):
        return frozenset(e for e in snapshot if self._is_salient(e))

    def _remember(self, d):
        self.event_count[d] = self.event_count.get(d, 0) + 1
        self.event_last[d] = self.tick
        self.wm.add(d)
        tracked = self._tracked()
        while len(self.wm) > self.wm_capacity:
            evictable = [e for e in self.wm if e not in tracked]
            if not evictable:
                break
            worst = max(evictable, key=lambda e: (self.event_count[e], -self.event_last[e]))
            self.wm.discard(worst)

    def _rarity(self, e):
        return (self.event_count.get(e, 0), -self.event_last.get(e, 0))

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def _observe_internal(self, g0, action, g1, reward, done):
        self.tick += 1
        if done and reward > 0:
            self._episode_success = True
        self.energy = max(0.0, min(self.max_energy,
                                   self.energy + 0.1 * reward - self.action_cost))
        if not self.body_known():
            self._discover_body(g0, action, g1)
            self.pending.append((g0, action, g1, reward, done))
            self.pending = self.pending[-200:]
            if self.body_known():
                for item in self.pending:
                    self._learn(*item, replay=True)
                self.pending = []
            return
        self._learn(g0, action, g1, reward, done)

    def _learn(self, g0, action, g1, reward, done, replay=False):
        motor = self.motor_map.get(action)
        p = self._locate_self(g0)
        if motor is None or p is None:
            return
        body = self._val(g0, p)
        t = add(p, motor)
        q = self._track_self(g0, g1, p, motor)
        if q != p:                                   # the cell I left shows the floor
            floor = self._val(g1, p)
            self.floor_counts[floor] = self.floor_counts.get(floor, 0) + 1

        ctx_local = self._local_context(g0, p, motor)
        effect = []
        for c in (p, t):
            if c in self.volatile:
                continue
            old, new = self._val(g0, c), self._val(g1, c)
            token = KEEP if new == old else (SELF if new == body else new)
            effect.append((canon(sub(c, p), motor), token))
        signature = (frozenset(effect), canon(sub(q, p), motor),
                     round(float(reward), 2), bool(done))
        d = (ctx_local, signature)
        if q == p and reward > -10 and not done:
            self.blocked_contexts.add(ctx_local)

        # Start a delayed experiment if this event is informative
        if not replay and self.experiment is None and self._open_groups and \
                self._delay_info(d, self._remote_values(g0, {p, t, q})) > 0:
            self.experiment = {"until": self.tick + self.max_lag, "cause": d}
            self.experiments_run += 1          # only the evaluation changes, not the graph

        # 1) Remote and delayed causality
        self.history[self.tick] = d
        for old_tick in [k for k in self.history if k < self.tick - self.max_lag]:
            del self.history[old_tick]
        self.event_ticks.setdefault(d, []).append(self.tick)
        self._process_remote(g0, g1, {p, t, q})

        # 2) Working memory snapshot before this event
        snapshot = frozenset(self.wm)
        mk = (ctx_local, self._memkey(snapshot))
        if ctx_local in self.blocked_contexts:
            self.memkeys_seen.setdefault(ctx_local, set()).add(mk[1])
        if self.tried2.get(mk, 0) == 0:
            self.curiosity_version += 1
        self.tried2[mk] = self.tried2.get(mk, 0) + 1
        tried_key = (ctx_local, self._salient(snapshot))
        if self.tried.get(tried_key, 0) == 0:
            self.curiosity_version += 1
        self.tried[tried_key] = self.tried.get(tried_key, 0) + 1

        # 3) Latent version space and aliasing
        occs = self.occurrences.setdefault(ctx_local, [])
        repeated = (signature, snapshot) in occs
        if repeated:                                   # identical evidence: refresh only
            occs.remove((signature, snapshot))
        occs.append((signature, snapshot))
        del occs[:-self.max_occurrences]
        if not replay:
            log = self.occ_world.setdefault(ctx_local, [])
            if (signature, snapshot, self.world_id) not in log:
                log.append((signature, snapshot, self.world_id))
                del log[:-self.max_world_occurrences]
        # identical evidence cannot change the version space of this context
        changed = False if repeated else self._update_latent_group(ctx_local)
        flags = self._flags_from(snapshot)
        aliased = None
        for sig_old, snap_old in occs[:-1]:
            if sig_old != signature and self._flags_from(snap_old) == flags:
                aliased = (sig_old, snap_old)
                break
        if aliased is not None and self._induce_latent(ctx_local, snapshot, aliased[1],
                                                       signature, aliased[0]):
            self._minimize_latents()
            changed = True
        if changed:
            self._reconsolidate()
        else:
            self._integrate(ctx_local | self._mem_items(flags, ctx_local), signature)

        self._remember(d)

    # ---------------- remote / delayed links ---------------- #
    def _process_remote(self, g0, g1, local):
        now = self.tick
        remote = [c for c in self._diff(g0, g1) - local if c not in self.volatile]
        effects = {}
        for c in remote:
            effects.setdefault((self._val(g0, c), self._val(g1, c)), []).append(c)
        present = None                     # built only when a link needs it

        # update existing hypotheses whose cause happened exactly `lag` steps ago
        for lag in range(self.max_lag + 1):
            ev = self.history.get(now - lag)
            if ev is None:
                continue
            keys = self.links_by_cause.get(ev)
            if not keys:
                continue
            for key in keys:
                link = self.links[key]
                if link.lag != lag:
                    continue
                if present is None:
                    present = self._remote_values(g0, local)
                if link.old not in present:
                    continue
                self.compute_cost += 1
                link.applicable += 1
                e = (link.old, link.new)
                if e in effects:
                    link.joint += 1
                    link.cells.update(effects[e])

        # create new hypotheses: every recent event is a candidate cause
        for e, cells in effects.items():
            self.effect_counts[e] = self.effect_counts.get(e, 0) + 1
            explained = False
            for lag in range(self.max_lag + 1):
                ev = self.history.get(now - lag)
                if ev is None:
                    continue
                key = (ev, lag) + e
                if key not in self.links:
                    prior = bisect_right(self.event_ticks.get(ev, []), now - lag)
                    self.links[key] = CausalLink(ev, lag, e[0], e[1],
                                                 applicable=max(1, prior), cells=set(cells))
                    self.links_by_cause.setdefault(ev, set()).add(key)
                if self.links[key].status(self.effect_counts[e]) != "rejected":
                    explained = True
            if not explained:
                for c in cells:
                    self.exo_changes[c] = self.exo_changes.get(c, 0) + 1
                    if self.exo_changes[c] >= self.volatility_threshold and c not in self.volatile:
                        self.volatile.add(c)
                        self._dirty_all = True
                        self.model_version += 1

        # forget refuted / noise-only hypotheses
        for key, link in list(self.links.items()):
            n_e = self.effect_counts.get((link.old, link.new), 0)
            hopeless = (link.applicable >= 4 and link.precision() < 0.5) or \
                       (n_e >= 4 and link.joint / n_e < 0.5) or \
                       (link.cells and link.cells <= self.volatile)
            if hopeless:
                del self.links[key]
                self.links_by_cause[key[0]].discard(key)
        self._refresh_map_links()

    def link_groups(self):
        """effect (old, new) -> list of surviving hypotheses (version space)."""
        return self._groups

    def _compute_groups(self):
        groups = {}
        for key, link in self.links.items():
            if link.old in self.self_values:
                continue
            if link.status(self.effect_counts.get((link.old, link.new), 0)) != "rejected":
                groups.setdefault((link.old, link.new), []).append(key)
        return groups

    def _refresh_map_links(self):
        self._groups = self._compute_groups()
        self._open_groups = {e: keys for e, keys in self._groups.items() if len(keys) >= 2}
        chosen = set()
        for keys in self._groups.values():
            best = max(keys, key=lambda k: (self.links[k].joint, -self.links[k].applicable,
                                            -k[1], -self._rarity(k[0])[0]))
            chosen.add(best)
        chosen = frozenset(chosen)
        if chosen != self._map_links:
            self._map_links = chosen
            by_cause = {}
            for key in chosen:
                by_cause.setdefault(key[0], []).append(key)
            self._map_links_by_cause = by_cause
            self._dirty_all = True
            self.model_version += 1

    def _delay_info(self, d, remote_values):
        """Expected bits gained by producing event d now (delayed hypotheses)."""
        bits = 0.0
        for e, keys in self._open_groups.items():
            if e[0] not in remote_values:
                continue
            if not any(k[0] == d for k in keys):
                continue
            classes = {}
            for k in keys:
                cls = k[1] if k[0] == d else "none"
                classes[cls] = classes.get(cls, 0) + 1
            bits += entropy(classes.values())
        return bits

    def _is_open_cause(self, d):
        return any(any(k[0] == d for k in keys) for keys in self._open_groups.values())

    # ---------------- latent version space (compound causes) ---------------- #
    # A hidden cause is a FORMULA over episode memory: a conjunction of at most
    # two literals (event, polarity). Both orientations are kept
    # ("outcome1 iff f" and "outcome2 iff f"), so disjunctions are covered too:
    #   success iff (P and Q) | success iff (P and not Q) | success iff (P or Q)
    def _vocab_for(self, ctx, prefer=(), keep=()):
        """Rare remembered events that may matter for `ctx`. Events already in
        the vocabulary (`keep`) stay even when they stop being rare."""
        union = set()
        for _sig, snap in self.occurrences.get(ctx, []):
            union |= snap
        cands = [e for e in union if e[0] != ctx and (e in keep or self._is_salient(e))]
        cands.sort(key=lambda e: (e not in keep, e not in prefer, self.event_count.get(e, 0),
                                  -self.event_last.get(e, 0)))
        return cands[:self.vocab_size]

    def _enumerate_hypotheses(self, ctx, vocab, outcomes):
        occs = self.occurrences.get(ctx, [])
        o1, o2 = outcomes
        formulas = [frozenset([(e, pol)]) for e in vocab for pol in (True, False)]
        for i in range(len(vocab)):
            for j in range(i + 1, len(vocab)):
                for pi in (True, False):
                    for pj in (True, False):
                        formulas.append(frozenset([(vocab[i], pi), (vocab[j], pj)]))
        worlds = [frozenset(e for k, e in enumerate(vocab) if mask >> k & 1)
                  for mask in range(1 << len(vocab))]
        g = self.latent_groups.get(ctx)
        prior = g.get("prior", []) if g is not None else []
        skip = g.get("augmented", set()) if g is not None else set()
        if skip:
            occs = [o for o in occs if o not in skip]
        best_by_table = {}
        for f in formulas:
            for o_true, o_false in ((o1, o2), (o2, o1)):
                self.compute_cost += 1
                if not all(sig == (o_true if formula_true(f, snap) else o_false)
                           for sig, snap in occs):
                    continue
                if prior and not all(
                        sig == (o_true if formula_true_scoped(f, snap, present) else o_false)
                        for sig, snap, present in prior):
                    continue
                table = tuple((o_true if formula_true(f, w) else o_false) == o1 for w in worlds)
                old = best_by_table.get(table)
                if old is None or self._complexity(f, ctx) < self._complexity(old[0], ctx):
                    best_by_table[table] = (f, o_true, o_false)
        return list(best_by_table.values())

    def _complexity(self, f, ctx=None):
        """Evidence-weighted Occam order: fewer literals, positive literals, then
        the formula whose events were observed most often at this context (it has
        survived more chances to be refuted)."""
        if ctx is None:
            evidence = -sum(self.event_count.get(e, 0) for e, _pol in f)
        else:
            occs = self.occurrences.get(ctx, [])
            evidence = sum(1 for e, _pol in f for _sig, snap in occs if e in snap)
        return (len(f), sum(1 for _e, pol in f if not pol), -evidence)

    @staticmethod
    def _prior(f):
        return 1.0 if len(f) == 1 else 0.3

    def _set_group(self, ctx, vocab, outcomes, hyps):
        best = min(hyps, key=lambda h: self._complexity(h[0], ctx))[0]
        g = self.latent_groups.get(ctx)
        if g is None:
            self.latents.append(best)
        elif g["map"] != best:
            self.latents[self.latents.index(g["map"])] = best
        changed = g is None or g["map"] != best
        if g is None or g["hyps"] != hyps:
            self.model_version += 1               # information landscape changed
            self._info_version += 1
        group = self.latent_groups.setdefault(ctx, {})
        group.update({"vocab": vocab, "outcomes": outcomes, "hyps": hyps, "map": best})
        if changed and group.get("prior"):
            self._augment_unified(ctx)
        return changed

    def _induce_latent(self, ctx, snap_new, snap_old, sig_new, sig_old):
        if len(self.latents) >= self.max_latents or ctx in self.latent_groups:
            return False
        vocab = self._vocab_for(ctx, prefer=snap_new ^ snap_old)
        if not vocab:
            return False
        outcomes = (sig_new, sig_old)
        hyps = self._enumerate_hypotheses(ctx, vocab, outcomes)
        if not hyps:
            return False
        return self._set_group(ctx, vocab, outcomes, hyps)

    def _update_latent_group(self, ctx):
        g = self.latent_groups.get(ctx)
        if g is None:
            return False
        vocab = self._vocab_for(ctx, keep=set(g["vocab"]))
        hyps = self._enumerate_hypotheses(ctx, vocab, g["outcomes"])
        if not hyps:                                  # nothing in the language fits
            self.latents.remove(g["map"])
            del self.latent_groups[ctx]
            self.model_version += 1
            self._info_version += 1
            return True
        return self._set_group(ctx, vocab, g["outcomes"], hyps)

    def latent_hypotheses(self):
        """ctx -> list of (formula, outcome_if_true) still consistent with experience."""
        return {ctx: [(f, ot) for f, ot, _of in g["hyps"]]
                for ctx, g in self.latent_groups.items()}

    def _alias_free(self, latents):
        for occs in self.occurrences.values():
            seen = {}
            for sig, snap in occs:
                if seen.setdefault(self._flags_from(snap, latents), sig) != sig:
                    return False
        return True

    def _minimize_latents(self):
        for f in list(self.latents[:-1]):
            reduced = [l for l in self.latents if l != f]
            if self._alias_free(reduced):
                self.latents = reduced
                for ctx in [c for c, g in self.latent_groups.items() if g["map"] == f]:
                    del self.latent_groups[ctx]
                    self._info_version += 1

    def _latent_info(self, ctx, mem):
        """Expected information gain (bits) of testing `ctx` with imagined memory
        `mem`: entropy of the prior-weighted outcome of the surviving formulas."""
        g = self.latent_groups.get(ctx)
        if g is None or len(g["hyps"]) < 2:
            return 0.0
        relevant = self._policy_relevant(g)
        if not relevant and not self._superstition_events(g):
            return 0.0
        key = (ctx, mem)
        if key in self._info_cache:
            return self._info_cache[key]
        mass = {}
        for f, o_true, o_false in g["hyps"]:
            o = o_true if formula_true(f, mem) else o_false
            mass[o] = mass.get(o, 0.0) + self._prior(f)
        bits = entropy(mass.values())
        if not relevant:
            # knowledge for its own sake: cheap in normal episodes, strong in a
            # budgeted "science episode" scheduled after a success
            bits *= self.science_weight if self.science_mode else self.superstition_weight
        self._info_cache[key] = bits
        return bits

    def _reconsolidate(self):
        """Re-learn all stored occurrences under the new latent encoding.
        Diff-based: only schemas that actually changed invalidate imagined edges,
        and unchanged schemas keep their identity (creation order)."""
        before = {(sc.context, sc.signature): sc for sc in self.schemas}
        self._suspend_dirty = True
        self.schemas = []
        self.exemplars = {}
        for ctx_local, occs in self.occurrences.items():
            for sig, snap in occs:
                self._integrate(ctx_local | self._mem_items(self._flags_from(snap), ctx_local), sig)
        self._suspend_dirty = False
        after = {}
        for sc in self.schemas:
            old = before.get((sc.context, sc.signature))
            if old is not None:
                sc.sid = old.sid
            after[(sc.context, sc.signature)] = sc
        for key in set(before) ^ set(after):
            self._mark_dirty(before.get(key) or after[key])
        # memory flags may now mean a different formula: anything that reads a
        # flag must be re-imagined even if its text did not change
        for sc in list(before.values()) + self.schemas:
            if any(isinstance(k[0], str) for k, _v in sc.context):
                self._mark_dirty(sc)
        self._flags_memo = {}
        self.model_version += 1

    # ---------------- schemas ---------------- #
    def _integrate(self, context, signature):
        self.compute_cost += 1
        self.exemplars.pop(context, None)
        self.exemplars[context] = signature
        while len(self.exemplars) > self.max_exemplars:
            self.exemplars.pop(next(iter(self.exemplars)))
        best = self._match_context(context)
        if best is not None and best.signature == signature:
            best.uses += 1
            best.last_tick = self.tick
            return
        if best is not None:
            self.schemas.remove(best)
            self._mark_dirty(best)
            if len(best.exemplars) > 1:
                for ex in best.exemplars:
                    if ex != context and ex in self.exemplars:
                        self._insert(ex, self.exemplars[ex])
        self._insert(context, signature)
        self.model_version += 1
        if len(self.schemas) > self.max_rules:
            self.prune()

    def _consistent(self, context, signature):
        for ctx, sig in self.exemplars.items():
            self.compute_cost += 1
            if context <= ctx and sig != signature:
                return False
        return True

    def _insert(self, context, signature):
        for s in self.schemas:
            if s.signature != signature:
                continue
            inter = s.context & context
            if self._consistent(inter, signature):
                self._mark_dirty(s)
                s.context = inter
                self._mark_dirty(s)
                s.exemplars.append(context)
                s.uses += 1
                s.last_tick = self.tick
                return s
        effect, delta, reward, done = signature
        self._schema_counter += 1
        s = Schema(context=context, effect=effect, self_delta=delta, reward=reward,
                   done=done, exemplars=[context], last_tick=self.tick, sid=self._schema_counter)
        self.schemas.append(s)
        self._mark_dirty(s)
        return s

    @staticmethod
    def _more_specific(s, best):
        return best is None or len(s.context) > len(best.context) or (
            len(s.context) == len(best.context) and s.sid < best.sid)

    def _match_context(self, context):
        best = None
        for s in self.schemas:
            self.compute_cost += 1
            if s.context <= context and self._more_specific(s, best):
                best = s
        return best

    def _importance(self, s):
        recency = self.decay ** (self.tick - s.last_tick)
        return recency * (1 + math.log1p(s.uses)) * (1 + abs(s.reward) / 50.0)

    def prune(self):
        self.schemas.sort(key=self._importance, reverse=True)
        removed = len(self.schemas) - self.max_rules
        if removed > 0:
            del self.schemas[self.max_rules:]
            self._dirty_all = True
            self.pruned_total += removed
            self.model_version += 1

    # ------------------------------------------------------------------ #
    # Imagination
    # ------------------------------------------------------------------ #
    def _schema_index(self):
        """Sparse activation: only schemas about the value in front of me wake up."""
        if self._index_version != self.model_version or self._index is None:
            by_target, generic = {}, []
            for s in self.schemas:
                target = [v for k, v in s.context if k == FORWARD]
                if target:
                    by_target.setdefault(target[0], []).append(s)
                else:
                    generic.append(s)
            self._index = (by_target, generic)
            self._index_version = self.model_version
        return self._index

    def _compiled_index(self):
        """Schema index with each context pre-split into its requirement kinds:
        (length, sid, body requirement, memory-flag requirements, wider cells)."""
        if self._cindex is None or self._cindex_version != self.model_version:
            by_target, generic = self._schema_index()

            def comp(s):
                here, flag_reqs, others = _ANY, [], []
                for key, v in s.context:
                    if key == FORWARD:
                        continue
                    if key == HERE:
                        here = v
                    elif isinstance(key[0], str):
                        flag_reqs.append((key[1], v))
                    else:
                        others.append((key, v))
                return (len(s.context), s.sid, here, tuple(flag_reqs), tuple(others), s)

            self._cindex = ({t: [comp(s) for s in lst] for t, lst in by_target.items()},
                            [comp(s) for s in generic])
            self._cindex_version = self.model_version
        return self._cindex

    def _match_grid(self, grid, p, motor, flags):
        best = None
        by_target, generic = self._schema_index()
        candidates = by_target.get(self._val(grid, add(p, motor)), []) + generic
        here = self._val(grid, p)
        ahead = self._val(grid, (p[0] + motor[0], p[1] + motor[1]))
        for s in candidates:
            ok = True
            for key, v in s.context:
                if key == FORWARD:
                    if ahead != v:
                        ok = False
                        break
                elif key == HERE:
                    if here != v:
                        ok = False
                        break
                elif isinstance(key[0], str):
                    if (key[1] in flags) != v:
                        ok = False
                        break
                elif self._val(grid, add(p, uncanon(key, motor))) != v:
                    ok = False
                    break
            if ok and self._more_specific(s, best):
                best = s
        self.compute_cost += len(candidates)
        return best

    def _rewrite(self, g, old, new, protected):
        vol = self.volatile
        mod = g.rows
        for r, base_row in enumerate(g.base):
            row = mod.get(r)
            if row is None:
                row = base_row
            if old not in row:
                continue
            cols = [c for c, v in enumerate(row) if v == old]
            for c in cols:
                if (r, c) not in protected and (r, c) not in vol:
                    g.set(r, c, new)

    def _imagine_schema(self, here, ahead, p, motor, mem, cell):
        """The schema `predict` would select, from my two canonical cells, my
        memory and (only for wider contexts) a cell accessor."""
        mv = self.model_version
        if self._flags_memo_version != mv:
            self._flags_memo, self._flags_memo_version = {}, mv
        flags = self._flags_memo.get(mem)
        if flags is None:
            flags = self._flags_memo[mem] = self._flags_from(mem)
        by_target, generic = self._compiled_index()
        s = None
        blen, bsid = -1, 0
        for lst in (by_target.get(ahead), generic):
            if not lst:
                continue
            self.compute_cost += len(lst)
            for n, sid, hreq, freqs, others, sc in lst:
                if n < blen or (n == blen and sid > bsid):
                    continue
                if hreq is not _ANY and hreq != here:
                    continue
                ok = True
                for i, v in freqs:
                    if (i in flags) != v:
                        ok = False
                        break
                if ok and others:
                    for key, v in others:
                        if cell(add(p, uncanon(key, motor))) != v:
                            ok = False
                            break
                if ok:
                    s, blen, bsid = sc, n, sid
        return s

    def predict(self, grid, p, action, mem=frozenset(), pending=()):
        """Imagine `action`.
        Returns (grid', pos', reward, done, mem', pending', local_ctx, event) or None."""
        motor = self.motor_map.get(action)
        if motor is None:
            return None
        mv = self.model_version
        if self._flags_memo_version != mv:
            self._flags_memo, self._flags_memo_version = {}, mv
        flags = self._flags_memo.get(mem)
        if flags is None:
            flags = self._flags_memo[mem] = self._flags_from(mem)
        by_target, generic = self._compiled_index()
        val = self._val
        t = (p[0] + motor[0], p[1] + motor[1])
        pr, pc = p
        try:
            here = grid[pr][pc] if pr >= 0 and pc >= 0 else VOID
        except IndexError:
            here = VOID
        tr, tc = t
        try:
            ahead = grid[tr][tc] if tr >= 0 and tc >= 0 else VOID
        except IndexError:
            ahead = VOID
        s = None
        blen, bsid = -1, 0
        n_cand = 0
        for lst in (by_target.get(ahead), generic):
            if not lst:
                continue
            n_cand += len(lst)
            for n, sid, hreq, freqs, others, sc in lst:
                if n < blen or (n == blen and sid > bsid):
                    continue
                if hreq is not _ANY and hreq != here:
                    continue
                ok = True
                for i, v in freqs:
                    if (i in flags) != v:
                        ok = False
                        break
                if ok and others:
                    for key, v in others:
                        if val(grid, add(p, uncanon(key, motor))) != v:
                            ok = False
                            break
                if ok:
                    s, blen, bsid = sc, n, sid
        self.compute_cost += n_cand
        if s is None:
            return None
        body = here
        g = _LazyGrid(grid)                  # copy-on-write: only touched rows
        rows, cols = len(grid), len(grid[0])
        for coff, token in s.effect:
            if token == KEEP:
                continue
            r, c = p if coff == HERE else (t if coff == FORWARD else add(p, uncanon(coff, motor)))
            if 0 <= r < rows and 0 <= c < cols:
                g.set(r, c, body if token == SELF else token)
        delta = s.self_delta
        q = p if delta == HERE else (t if delta == FORWARD else add(p, uncanon(delta, motor)))
        vol = self.volatile
        if vol:
            items = []
            if p not in vol:
                items.append((HERE, here))
            if t not in vol:
                items.append((FORWARD, ahead))
            ctx_local = frozenset(items)
        else:
            ctx_local = frozenset(((HERE, here), (FORWARD, ahead)))
        d = (ctx_local, s.signature)
        # pending (delayed) effects tick down
        new_pending = []
        for remaining, old, new in pending:
            if remaining - 1 <= 0:
                self._rewrite(g, old, new, {q})
            else:
                new_pending.append((remaining - 1, old, new))
        # effects caused by this event (links indexed by their cause)
        caused = self._map_links_by_cause.get(d)
        if caused:
            protected = {p, t, q}
            for key in caused:
                link = self.links[key]
                if link.lag == 0:
                    self._rewrite(g, link.old, link.new, protected)
                else:
                    new_pending.append((link.lag, link.old, link.new))
        tracked = self._plan_tracked_cache if self._plan_tracked_version == mv \
            else self._plan_tracked()
        if d in tracked:
            mem = frozenset(mem | {d})
        return (g.freeze(), q, s.reward, s.done,
                mem, tuple(sorted(new_pending)) if new_pending else (), ctx_local, d)

    def _mark_dirty(self, schema):
        """A schema changed: imagined edges whose target cell shows its value
        must be re-imagined; every other edge of the cached graph stays valid."""
        if self._suspend_dirty:
            return
        here = ahead = _ANY
        for k, v in schema.context:
            if k == FORWARD:
                ahead = v
            elif k == HERE:
                here = v
        if ahead is _ANY:
            self._dirty_all = True               # a target-free schema can match anywhere
            return
        self._dirty_targets.add((here, ahead))   # (my body, what is in front of me)
        self._dirty_any_body.add(ahead) if here is _ANY else None

    def _is_memory_commit(self, ev, imagined_mem):
        """A rare, silent event done for the first time in this episode may be a
        hidden switch: once in episode memory it cannot be undone."""
        if ev in self.wm or ev in imagined_mem:
            return False                       # a repeat changes nothing
        return self._is_salient(ev)

    def _memkey(self, wm):
        """Coarse memory state for exploration bookkeeping: which known failure
        contexts (silent, blocked objects) were already touched this episode."""
        return frozenset(e[0] for e in wm if e[0] in self.blocked_contexts)

    def _coverage_deficit(self, ctx):
        """1 for a context tested under few memory states, -> 0 when well covered."""
        most = self._novelty_cache.get("_most")
        if most is None:
            most = self._novelty_cache["_most"] = max(
                (len(v) for v in self.memkeys_seen.values()), default=0)
        if most == 0:
            return 0.0
        return 1.0 - len(self.memkeys_seen.get(ctx, ())) / most

    def _followup_novelty(self, ev, salient):
        """Fraction of known failure contexts that, right after committing `ev`,
        could be tested under a memory state never seen before."""
        if not self.blocked_contexts:
            return 0.0
        nxt = salient | {ev[0]}
        cache = self._novelty_cache
        if (ev, salient) in cache:
            return cache[(ev, salient)]
        novel = sum(1 for c in self.blocked_contexts if self.tried2.get((c, nxt), 0) == 0)
        value = novel / len(self.blocked_contexts)
        cache[(ev, salient)] = value
        return value

    @staticmethod
    def _scc(order, succ):
        """Iterative Tarjan: strongly connected components of the imagined graph."""
        index, low, comp = {}, {}, {}
        stack, on_stack = [], set()
        counter = [0]
        for root in order:
            if root in index:
                continue
            work = [(root, 0)]
            index[root] = low[root] = counter[0]
            counter[0] += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                v, i = work[-1]
                nbrs = succ[v]
                if i < len(nbrs):
                    work[-1] = (v, i + 1)
                    w = nbrs[i]
                    if w not in index:
                        index[w] = low[w] = counter[0]
                        counter[0] += 1
                        stack.append(w)
                        on_stack.add(w)
                        work.append((w, 0))
                    elif w in on_stack:
                        low[v] = min(low[v], index[w])
                else:
                    work.pop()
                    if work:
                        low[work[-1][0]] = min(low[work[-1][0]], low[v])
                    if low[v] == index[v]:
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            comp[w] = v
                            if w == v:
                                break
        return comp

    def reversibility(self, obs, action):
        """Classify a motor command in the observed situation:
        'reversible' | 'irreversible' | 'unknown' | 'terminal'."""
        grid = self._grid(obs)
        p = self._locate_self(grid)
        mem = frozenset(self._plan_tracked() & self.wm)
        self._graph = None                       # classify from this exact situation
        key = self._plan(grid, p, mem, self._root_pending())
        return self._label(key, action)

    def _intuition(self, grid, p, motor):
        target_value = self._val(grid, add(p, motor))
        rewards = [s.reward for s in self.schemas if (FORWARD, target_value) in s.context]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def _root_pending(self):
        pending = []
        for key in self._map_links:
            link = self.links[key]
            if link.lag == 0:
                continue
            for t0, ev in self.history.items():
                remaining = t0 + link.lag - self.tick
                if ev == link.cause and remaining > 0:
                    pending.append((remaining, link.old, link.new))
        return tuple(sorted(pending))

    # ------------------------------------------------------------------ #
    # Planning = (1) imagined graph, rebuilt only when the MODEL changes,
    #            (2) cheap evaluation of bonuses + value iteration.
    # ------------------------------------------------------------------ #
    def _build_graph(self, grid, p, mem, pending):
        """Imagination proper: depends only on the model, so it is cached and
        reused while the agent moves inside it. Static parts of the value
        problem are stored as arrays; only exploration bonuses change later."""
        self.graph_builds += 1
        self._info_cache = {}
        self._flags_memo = {}
        grid = self._mask(grid)
        root = self._key(grid, p, mem, pending)
        tracked = self._plan_tracked()
        rivals_now = bool(self._struct_rivals())
        # INCREMENTAL: reuse the previous graph's edges unless the schemas about
        # their target value changed (or something global changed)
        old = self._graph
        if old is None or self._dirty_all or old.get("lens_epoch") != self._lens_epoch:
            old = None
        # a change of what imagination tracks only touches edges whose event
        # entered/left the tracked set, and nodes remembering a dropped event
        delta = (tracked ^ old["tracked"]) if old is not None else frozenset()
        dropped = (old["tracked"] - tracked) if old is not None else frozenset()
        dirty, dirty_any = self._dirty_targets, self._dirty_any_body
        same_info = old is not None and old.get("info_version") == self._info_version \
            and not rivals_now
        nodes = {root: (grid, p, mem, pending)}
        order = [root]
        preds, aheads, ahead_values = {}, {}, {}
        val = self._val
        predict = self.predict
        actions = range(self.n_actions)
        motors = [self.motor_map[a] for a in actions]
        latent_groups = self.latent_groups
        info_scale = self.info_scale
        zero_info = info_scale * 0.0
        max_nodes = self.max_plan_nodes
        imagined = 0
        i = 0
        while i < len(order):
            k = order[i]
            i += 1
            g, pos, m, pend = nodes[k]
            if old is not None and k in old["preds"] and not (
                    dirty & old["aheads"][k] or dirty_any & old["ahead_vals"][k]) and not (
                    delta and (m & dropped or any(
                        e[3] is not None and e[3][4] in delta for e in old["preds"][k]))):
                old_out = old["preds"][k]
                if same_info:                         # nothing to recompute: share it
                    for _a, k2, _r, _x in old_out:
                        if k2 is not None and k2 not in nodes:
                            nodes[k2] = old["nodes"][k2]
                            order.append(k2)
                    if len(nodes) <= self.max_plan_nodes:
                        preds[k] = old_out
                        aheads[k] = old["aheads"][k]
                        ahead_values[k] = old["ahead_vals"][k]
                        self.edges_reused += len(old_out)
                        continue
                out = []
                for a, k2, r, extra in old_out:
                    if k2 is not None and k2 not in nodes:
                        if len(nodes) < self.max_plan_nodes:
                            nodes[k2] = old["nodes"][k2]
                            order.append(k2)
                        else:
                            k2 = None
                    if extra is not None:
                        d, chain, blocked, ctx, ev, mm, _info = extra
                        extra = (d, chain, blocked, ctx, ev, mm,
                                 info_scale * self._latent_info(ctx, mm)
                                 if ctx in latent_groups else zero_info)
                    out.append((a, k2, r, extra))
                preds[k] = out
                aheads[k] = old["aheads"][k]
                ahead_values[k] = old["ahead_vals"][k]
                self.edges_reused += len(out)
                continue
            out = []
            here = val(g, pos)
            row_ahead = []
            pr, pc = pos
            for a in actions:
                motor = motors[a]
                row_ahead.append(val(g, (pr + motor[0], pc + motor[1])))
                pred = predict(g, pos, a, m, pend)
                imagined += 1
                if pred is None:
                    out.append((a, None, self._intuition(g, pos, motor), None))
                    continue
                g2, q, r, d, m2, pend2, ctx, ev = pred
                k2 = None
                if not d:
                    k2 = (g2, q, m2, pend2)
                    if k2 not in nodes:
                        if len(nodes) < max_nodes:
                            nodes[k2] = (g2, q, m2, pend2)
                            order.append(k2)
                        else:
                            k2 = None
                info = info_scale * self._latent_info(ctx, m) \
                    if ctx in latent_groups else zero_info
                out.append((a, k2, r, (d, m2 != m, q == pos, ctx, ev, m, info)))
            preds[k] = out
            ahead_values[k] = frozenset(row_ahead)
            aheads[k] = frozenset((here, v) for v in row_ahead)
        self.edges_imagined += imagined
        self._dirty_targets = set()
        self._dirty_any_body = set()
        self._dirty_all = False
        rivals = self._struct_rivals()
        if rivals:
            # edges where surviving structures predict different outcomes are
            # experiments: add their expected information gain (one-shot)
            touched = set()
            for _w, pi in rivals[1:]:
                touched |= set(pi)
            # memory only matters at hidden-cause test contexts (e.g. the door);
            # elsewhere an edge can only differ if its target value is remapped
            test_ctx = set(self.latent_groups)
            for k in order:
                g, pos, m, pend = nodes[k]
                new_out = []
                cache = {}
                for a, k2, r, extra in preds[k]:
                    if extra is not None:
                        d, chain, blocked, ctx, ev, mm, info = extra
                        ahead = dict(ctx).get(FORWARD)
                        if ahead in touched or ctx in test_ctx:
                            q = k2[1] if k2 is not None else (pos if blocked else None)
                            mine = (r, d, None if d else q)
                            bits = self._struct_eig(rivals, g, pos, a, m, pend, mine, cache)
                            if bits:
                                info += self.info_scale * bits
                                extra = (d, chain, blocked, ctx, ev, mm, info)
                    new_out.append((a, k2, r, extra))
                preds[k] = new_out
        # one pass: arrays for the value problem, the bump edges, and whether any
        # bump changes the imagined state without growing memory (e.g. a switch;
        # a bump that only lets time pass is a self-loop) -- only then is the
        # strongly-connected-component analysis needed
        index = {k: n for n, k in enumerate(order)}
        n, A = len(order), self.n_actions
        gamma = self.gamma
        R = [[0.0] * A for _ in range(n)]
        INFO = [[0.0] * A for _ in range(n)]
        DST = [[0] * A for _ in range(n)]
        CONT = [[0.0] * A for _ in range(n)]
        CHAIN = [[False] * A for _ in range(n)]
        unknown, raw_blocked, events = [], [], []
        want_events = bool(self._open_groups)
        needs_scc = False
        succ = {}
        for row, k in enumerate(order):
            Rr, Ir, Dr, Cr, Hr = R[row], INFO[row], DST[row], CONT[row], CHAIN[row]
            nxt = []
            for a, k2, r, extra in preds[k]:
                Rr[a] = r
                if extra is None:
                    unknown.append((row, a))
                    continue
                d, chain, is_blocked, ctx, ev, m, info = extra
                Ir[a] = info
                Hr[a] = chain
                if k2 is not None:
                    Dr[a] = index[k2]
                    Cr[a] = gamma
                    nxt.append(k2)
                if is_blocked:
                    moved = k2 is not None and k2[:3] != k[:3]
                    if moved and not chain:
                        needs_scc = True
                    raw_blocked.append((row, a, ctx, ev, m, d, k, k2, moved, chain))
                if want_events:
                    events.append((row, a, ev, k))
            succ[k] = nxt
        scc = self._scc(order, succ) if needs_scc else None
        blocked = [(row, a, ctx, ev, m, d or (moved and (chain or scc[k2] != scc[k])))
                   for row, a, ctx, ev, m, d, k, k2, moved, chain in raw_blocked]
        self._graph = {"version": self.model_version, "nodes": nodes, "order": order,
                       "index": index, "preds": preds, "scc": scc, "succ": succ, "aheads": aheads,
                       "ahead_vals": ahead_values,
                       "tracked": tracked, "lens_epoch": self._lens_epoch,
                       "info_version": self._info_version,
                       "R": R, "INFO": INFO, "DST": DST, "CONT": CONT, "CHAIN": CHAIN,
                       "unknown": unknown, "blocked": blocked, "events": events}
        if np is not None:
            for name in ("R", "INFO", "DST", "CONT", "CHAIN"):
                self._graph[name + "_np"] = np.array(self._graph[name])
        self._evaluation = None

    def _label(self, k, a):
        """Reversibility class of action `a` in imagined state `k`."""
        graph = self._graph
        for a2, k2, _r, extra in graph["preds"][k]:
            if a2 != a:
                continue
            if extra is None:
                return "unknown"
            d, _chain, _blocked, _ctx, ev, m, _info = extra
            if d:
                return "terminal"
            if self._is_memory_commit(ev, m):
                return "irreversible"
            if graph["scc"] is None:                 # computed only when asked
                graph["scc"] = self._scc(graph["order"], graph["succ"])
            if k2 is None or graph["scc"][k2] == graph["scc"][k]:
                return "reversible"
            return "irreversible"

    def _evaluate(self, root):
        """Exploration bonuses (curiosity, retry, reversibility gating, delayed
        information) on top of the cached static arrays, then value iteration."""
        self.replans += 1
        self._novelty_cache = {}
        graph = self._graph
        ratio = self.energy / self.max_energy
        bonus_unknown = self.curiosity_scale * ratio
        bonus_retry = 0.5 * self.curiosity_scale * ratio
        memkey = self._memkey(self.wm)
        aware = self.reversibility_aware
        # bonuses depend only on (context, event, memory, left): group the edges
        groups = graph.get("blocked_groups")
        if groups is None:
            groups = {}
            for row, a, ctx, ev, m, left in graph["blocked"]:
                groups.setdefault((ctx, ev, m, left), []).append((row, a))
            graph["blocked_groups"] = groups
        explore = []                                   # (cells, bonus, deferrable)
        any_reversible_retest = False
        tried2 = self.tried2
        for (ctx, ev, m, left), cells in groups.items():
            if tried2.get((ctx, memkey), 0) != 0:
                continue
            bonus = bonus_retry
            commit = self._is_memory_commit(ev, m)
            if commit and aware:
                bonus *= (1.0 + 2.0 * self._followup_novelty(ev, memkey)
                          + 4.0 * self._coverage_deficit(ctx))
            irreversible = commit or left
            if not irreversible:
                any_reversible_retest = True
            explore.append((cells, bonus, irreversible))
        pending = aware and any_reversible_retest      # exhaust before commit
        self.retests_pending = pending
        root_row = graph["index"][root]
        extra_info = {}
        for cells, bonus, irreversible in explore:
            if pending and irreversible:
                self.deferred_commits += sum(1 for row, _a in cells if row == root_row)
                continue
            for cell in cells:
                extra_info[cell] = bonus
        if self._open_groups:                          # delayed-cause experiments
            experiment_open = self.experiment is not None
            nodes = graph["nodes"]
            counts_cache = graph.setdefault("value_counts", {})
            by_ev = graph.get("events_by_ev")
            if by_ev is None:
                by_ev = {}
                for item in graph["events"]:
                    by_ev.setdefault(item[2], []).append(item)
                graph["events_by_ev"] = by_ev
            # only events that are candidate causes of an open group can carry
            # delayed information (or be penalised during a running experiment)
            open_causes = {key[0] for keys in self._open_groups.values() for key in keys}
            selected = [item for ev in open_causes for item in by_ev.get(ev, ())]
            for row, a, ev, k in selected:
                g, pos = nodes[k][0], nodes[k][1]
                if experiment_open:
                    extra_info[(row, a)] = extra_info.get((row, a), 0.0) - self.info_scale
                else:
                    counts = counts_cache.get(k)
                    if counts is None:
                        counts = counts_cache[k] = self._value_counts(g)
                    t = add(pos, self.motor_map[a])
                    here, ahead = self._val(g, pos), self._val(g, t)
                    remote = {v for v, n in counts.items()
                              if n - (here == v) - (ahead == v) > 0}
                    bits = self._delay_info(ev, remote)
                    if bits:
                        extra_info[(row, a)] = extra_info.get((row, a), 0.0) + self.info_scale * bits
        unknown_bonus = 0.0 if pending else bonus_unknown
        if pending:
            self.deferred_commits += sum(1 for row, _a in graph["unknown"] if row == root_row)
        if np is not None:
            V_base, V_total, R, INFO = self._solve_arrays(graph, extra_info, unknown_bonus)
        else:
            V_base, V_total, R, INFO = self._solve_lists(graph, extra_info, unknown_bonus)
        self._evaluation = {"key": (self.model_version, self.curiosity_version,
                                    self._salient(frozenset(self.wm)), memkey,
                                    self.experiment is not None),
                            "V_base": V_base, "V_total": V_total, "R": R, "INFO": INFO}

    def _solve_arrays(self, graph, extra_info, unknown_bonus):
        R = graph["R_np"].copy()
        INFO = graph["INFO_np"].copy()
        for row, a in graph["unknown"]:
            R[row, a] += unknown_bonus
        for (row, a), v in extra_info.items():
            INFO[row, a] += v
        DST, CONT, CHAIN = graph["DST_np"], graph["CONT_np"], graph["CHAIN_np"]
        n = R.shape[0]
        native = _native()
        if native is not None:
            D64 = graph.get("DST_i64")
            if D64 is None:
                D64 = graph["DST_i64"] = np.ascontiguousarray(DST, dtype=np.int64)
                graph["CONT_f64"] = np.ascontiguousarray(CONT, dtype=np.float64)
                graph["CHAIN_u8"] = np.ascontiguousarray(CHAIN, dtype=np.uint8)
            C64, CH8 = graph["CONT_f64"], graph["CHAIN_u8"]
            R64 = np.ascontiguousarray(R, dtype=np.float64)
            warm = graph.get("warm")
            V0_base = warm[0] if warm is not None else np.zeros(n)
            V_base = native.acore_value_iteration(R64, C64, D64, V0_base, 2000, 1e-3)
            if (INFO != 0).any():
                V0_total = warm[1] if warm is not None else V_base
                V_total = native.acore_value_iteration(
                    R64, C64, D64, V0_total, 2000, 1e-3,
                    INFO=np.ascontiguousarray(INFO, dtype=np.float64), CHAIN=CH8,
                    V_base=V_base)
            else:
                V_total = V_base
            graph["warm"] = (V_base, V_total)
            self.compute_cost += R.size
            return V_base, V_total, R, INFO

        def iterate(q_fn, V):
            for _ in range(2000):
                V_new = q_fn(V).max(axis=1)
                if np.max(np.abs(V_new - V)) < 1e-3:
                    return V_new
                V = V_new
            return V

        # warm start: on the same graph only a few bonuses changed since last time
        warm = graph.get("warm")
        V0_base = warm[0] if warm is not None else np.zeros(n)
        V_base = iterate(lambda V: R + CONT * V[DST], V0_base)
        has_info = INFO != 0
        if has_info.any():
            # one-shot information; memory-growing edges keep the full value
            base_cont = CONT * V_base[DST]
            V0_total = warm[1] if warm is not None else V_base
            V_total = iterate(lambda V: np.where(
                has_info, R + INFO + np.where(CHAIN, CONT * V[DST], base_cont),
                R + CONT * V[DST]), V0_total)
        else:
            V_total = V_base
        graph["warm"] = (V_base, V_total)
        self.compute_cost += R.size
        return V_base, V_total, R, INFO

    def _solve_lists(self, graph, extra_info, unknown_bonus):
        """Pure-Python fallback (no numpy)."""
        R = [row[:] for row in graph["R"]]
        INFO = [row[:] for row in graph["INFO"]]
        for row, a in graph["unknown"]:
            R[row][a] += unknown_bonus
        for (row, a), v in extra_info.items():
            INFO[row][a] += v
        DST, CONT, CHAIN = graph["DST"], graph["CONT"], graph["CHAIN"]
        n, A = len(R), self.n_actions

        def iterate(use_info, V_base=None):
            V = [0.0] * n
            for _ in range(2000):
                delta = 0.0
                for i in range(n - 1, -1, -1):
                    best = -1e18
                    for a in range(A):
                        if use_info and INFO[i][a]:
                            cont = V if CHAIN[i][a] else V_base
                            q = R[i][a] + INFO[i][a] + CONT[i][a] * cont[DST[i][a]]
                        else:
                            q = R[i][a] + CONT[i][a] * V[DST[i][a]]
                        best = max(best, q)
                    delta = max(delta, abs(best - V[i]))
                    V[i] = best
                if delta < 1e-3:
                    break
            return V

        V_base = iterate(False)
        V_total = iterate(True, V_base) if any(any(r) for r in INFO) else V_base
        return V_base, V_total, R, INFO

    def _plan(self, grid, p, mem, pending):
        """Make sure graph and evaluation are up to date for this situation."""
        key = self._key(self._mask(grid), p, mem, pending)
        graph = self._graph
        if graph is None or graph["version"] != self.model_version or key not in graph["nodes"]:
            self._build_graph(grid, p, mem, pending)
        ev_key = (self.model_version, self.curiosity_version, self._salient(frozenset(self.wm)),
                  self._memkey(self.wm), self.experiment is not None)
        if self._evaluation is None or self._evaluation["key"] != ev_key:
            self._evaluate(key)
        return key

    def _act_internal(self, grid):
        p = self._locate_self(grid) if self.self_values else None
        if not self.body_known() or p is None:
            return self._babble(grid)
        if self.experiment is not None and (self.tick >= self.experiment["until"]
                                            or not self._is_open_cause(self.experiment["cause"])):
            self.experiment = None             # evaluation key tracks this
        mem = frozenset(self._plan_tracked() & self.wm)
        key = self._plan(grid, p, mem, self._root_pending())
        ev, graph = self._evaluation, self._graph
        row = graph["index"][key]
        R, INFO, V_base, V_total = ev["R"], ev["INFO"], ev["V_base"], ev["V_total"]
        q_values = {}
        for a in range(self.n_actions):
            cont_v = V_base if (INFO[row][a] and not graph["CHAIN"][row][a]) else V_total
            q_values[a] = (float(R[row][a]) + float(INFO[row][a])
                           + graph["CONT"][row][a] * float(cont_v[graph["DST"][row][a]]))
        self.last_q = q_values
        best = max(q_values.values())
        return self.rng.choice([a for a, v in q_values.items() if abs(v - best) < 1e-9])

    # ------------------------------------------------------------------ #
    # ANALOGY: map a never-seen world onto known causal roles
    # ------------------------------------------------------------------ #
    def notice_world(self, obs):
        """Worlds change between episodes: check once, at the start of each
        episode, whether most of what I see is new. Returns the probe move
        while my body is still unknown in a new world, else None."""
        return self._notice_raw(self._grid(obs))

    def _notice_raw(self, raw):
        if self._episode_start and self.analogy_enabled is not None \
                and not self._world_pending and self.body_known():
            values = {v for row in raw for v in row}
            unseen = values - self.known_values
            if values and len(unseen) > self.new_world_fraction * len(values):
                self._enter_new_world()
        return self._probe_action(raw) if self._world_pending else None

    def act(self, obs):
        raw = self._grid(obs)
        probe = self._notice_raw(raw)
        if probe is not None:
            return probe
        grid = self._translate(raw)
        if self._episode_start:
            self._episode_start = False
            self._record_profiles(grid)
        return self._act_internal(grid)

    def observe(self, obs, action, next_obs, reward, done):
        raw0, raw1 = self._grid(obs), self._grid(next_obs)
        if self._world_pending:
            self._probe_buffer.append((raw0, action, raw1, reward, done))
            if not self._bind_new_world(raw0, action, raw1):
                self._probe_blocked.add((raw0, action))
                return                                  # body not found yet
            # replay the probing steps through the new lens (e.g. an alarm I bumped
            # while looking for my body is now remembered as an alarm)
            probes, self._probe_buffer = self._probe_buffer[:-1], []
            for p0, a0, p1, r0, d0 in probes:
                self._struct_update(p0, a0, p1, r0, d0)
                self._observe_internal(self._translate(p0), a0, self._translate(p1), r0, d0)
        self._remember_values(raw0)
        self._remember_values(raw1)
        g0, g1 = self._translate(raw0), self._translate(raw1)
        if self.struct_hyps and self.body_known():
            lens = self.lens
            self._struct_update(raw0, action, raw1, reward, done, g0, g1)
            if self.lens is not lens:                   # the structure switched
                g0, g1 = self._translate(raw0), self._translate(raw1)
        self._observe_internal(g0, action, g1, reward, done)

    def _probe_action(self, raw):
        """One safe-looking move to find my body: I do not know where I am, but my
        body is one of the unique cells; move in the direction in which most of
        those candidates would step onto the most common surface (likely floor)."""
        counts = {}
        for row in raw:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        surface = max(counts, key=counts.get)
        candidates = [(r, c) for r, row in enumerate(raw) for c, v in enumerate(row)
                      if counts[v] == 1]
        options = [a for a in range(self.n_actions) if a in self.motor_map
                   and (raw, a) not in self._probe_blocked] or list(self.motor_map)

        def score(a):
            m = self.motor_map[a]
            return sum(1 for cell in candidates if self._val(raw, add(cell, m)) == surface)

        best = max(score(a) for a in options)
        return self.rng.choice([a for a in options if score(a) == best])


    def _remember_values(self, raw):
        rows = self._rows_seen
        for row in raw:
            if row not in rows:
                rows.add(row)
                self.known_values.update(row)

    def _translate(self, raw):
        if not self.lens:
            return raw
        lens, memo = self.lens, self._row_memo
        if memo.get("_lens_size") != len(lens) or memo.get("_lens_id") != id(lens):
            memo.clear()
            memo["_lens_size"], memo["_lens_id"] = len(lens), id(lens)
        out = []
        for row in raw:
            tr = memo.get(row)
            if tr is None:
                tr = memo[row] = tuple(lens.get(v, v) for v in row)
            out.append(tr)
        return tuple(out)

    def _enter_new_world(self):
        self.world_id += 1
        self._world_pending = True
        self.tried2 = {}                                # retests are per world: a law
        self.memkeys_seen = {}                          # borrowed by analogy is untested here
        self._probe_blocked = set()
        self._probe_buffer = []
        self.worlds_entered += 1
        self.lens = {}
        self.analogical = set()
        self.tentative = set()
        self.revoked = set()
        self._lens_event_sources = {}
        self.volatile = set()                           # positions mean nothing here
        self.exo_changes = {}
        self._graph = None
        self._evaluation = None
        self._lens_epoch += 1
        self.model_version += 1

    def _bind_new_world(self, raw0, action, raw1):
        """Find my body by one move, then map every other value by role profile."""
        motor = self.motor_map[action]
        counts = {}
        for row in raw0:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        body = None
        for r, row in enumerate(raw0):
            for c, v in enumerate(row):
                q = (r + motor[0], c + motor[1])
                if counts[v] == 1 and self._val(raw1, q) == v and self._val(raw1, (r, c)) != v:
                    body, p = v, (r, c)
        if body is None:
            return False
        self._world_pending = False
        floor_raw = self._val(raw1, p)
        known = self.known_values
        lens = {}
        if body not in known:
            # my body in this world plays the role of my body in the known world
            lens[body] = self.body_start if self.body_start is not None \
                else next(iter(self.self_values))
        if floor_raw not in known:
            lens[floor_raw] = self._floor_internal()
        wall_raw = self._most_frequent(raw0, exclude={body, floor_raw})
        wall_int = self._wall_internal()
        if wall_raw is not None and wall_raw not in known and wall_int is not None:
            lens[wall_raw] = wall_int
        for v in {v for row in raw0 for v in row} & known:
            lens[v] = v                                  # perceptual continuity
        values = {v for row in raw0 for v in row}
        if self.analogy_enabled and self.synthesis:
            self._synthesize_roles()
        if self.analogy_enabled:
            self.struct_hyps = self._structural_hypotheses(raw0, p, floor_raw, wall_raw, lens)
        else:
            self.struct_hyps = [self._hypothesis(lens, {}, 0.0, values)]
        self._struct_targets = sorted(values - set(lens), key=str)
        self._set_struct_map(0, rebuild_memory=False)
        self.model_version += 1
        self._episode_start = True
        return True

    @staticmethod
    def _most_frequent(grid, exclude):
        counts = {}
        for row in grid:
            for v in row:
                if v not in exclude:
                    counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def _floor_internal(self):
        return max(self.floor_counts, key=self.floor_counts.get) if self.floor_counts else None

    def _wall_internal(self):
        return self.wall_value

    def _profile(self, grid, value, floor, wall, body_pos):
        """Role profile of a value, independent of its colour: how many copies,
        and what surrounds them (floor / wall / outside / other objects)."""
        cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == value]
        n = len(cells)
        mult = "one" if n == 1 else ("few" if n <= 4 else "many")
        hist = [0.0, 0.0, 0.0, 0.0]
        for cell in cells:
            for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = add(cell, d)
                v = floor if nb == body_pos else self._val(grid, nb)
                idx = 2 if v is VOID else (0 if v == floor else (1 if v == wall else 3))
                hist[idx] += 1.0 / n
        return mult, tuple(round(h, 2) for h in hist)

    def _record_profiles(self, grid):
        """At the start of each episode in a known world, store the role profile
        of every value I see (in my own internal vocabulary). In a world seen
        through an analogy only NEW roles get a card (existing cards are never
        overwritten: a wrong structure must not corrupt them)."""
        present = {v for row in grid for v in row}
        self.world_present[self.world_id] = present   # this world, as I see it NOW
        p = self._locate_self(grid)
        if p is not None:
            self.body_start = self._val(grid, p)
        floor = self._floor_internal()
        if p is None or floor is None:
            return
        self.wall_value = self._most_frequent(grid, exclude={self.body_start, floor})
        for v in present:
            if v in self.self_values or v == floor or v == self.wall_value:
                continue
            if isinstance(v, tuple):
                continue
            if self.struct_hyps and v in self.value_profiles:
                continue
            self.value_profiles[v] = self._profile(grid, v, floor, self.wall_value, p)
            self.card_world.setdefault(v, self.world_id)

    # ------------------------------------------------------------------ #
    # MULTI-SOURCE SYNTHESIS: one role, laws learned in different worlds
    # ------------------------------------------------------------------ #
    def _test_context_of(self, value):
        """The hidden-cause test context whose target is `value`, if any."""
        for ctx, g in self.latent_groups.items():
            if dict(ctx).get(FORWARD) == value and not g.get("prior"):
                return ctx, g
        return None, None

    def _synthesize_roles(self):
        """Two known roles with the same profile, each a hidden-cause test context
        with the same pair of outcomes, learned in DIFFERENT worlds, may be one
        kind of object. Their evidence is pooled with presence-scoped semantics;
        if some formula explains every world, a UNIFIED role is created whose
        law is that formula (e.g. lever from world A AND NOT alarm from world B)."""
        made = []
        values = sorted(self.value_profiles, key=str)
        for i, v1 in enumerate(values):
            for v2 in values[i + 1:]:
                if isinstance(v1, tuple) or isinstance(v2, tuple):
                    continue
                if self.card_world.get(v1) == self.card_world.get(v2):
                    continue
                (m1, h1), (m2, h2) = self.value_profiles[v1], self.value_profiles[v2]
                if m1 != m2 or sum(abs(a - b) for a, b in zip(h1, h2)) > self.analogy_threshold:
                    continue
                c1, g1 = self._test_context_of(v1)
                c2, g2 = self._test_context_of(v2)
                if g1 is None or g2 is None:
                    continue
                rew = lambda g: sorted(o[2] for o in g["outcomes"])
                if rew(g1) != rew(g2):
                    continue
                unified = ("unified", v1, v2)
                if unified in self.value_profiles:
                    continue
                if self._make_unified(unified, (v1, c1, g1), (v2, c2, g2)):
                    made.append(unified)
        return made

    def _make_unified(self, u, part1, part2):
        body = None
        pooled, vocab = [], []
        outcomes = None
        for v, ctx, g in (part1, part2):
            items = dict(ctx)
            body = items.get(HERE, body)
            sig_map = {}
            for o in g["outcomes"]:
                sig_map[o[2]] = o
            if outcomes is None:
                outcomes = g["outcomes"]
            # translate this role's evidence to the unified role
            for sig, snap, wid in self.occ_world.get(ctx, []):
                sig_u = outcomes[0] if sig[2] == outcomes[0][2] else outcomes[1]
                present = set(self.world_present.get(wid, set()))
                pooled.append((sig_u, snap, frozenset(present)))
            # the ingredients of the synthesis are the INGREDIENTS OF THE LAWS
            # each world taught (not every event that happened to be logged)
            for e, _pol in sorted(g["map"], key=str):
                if e not in vocab:
                    vocab.append(e)
        if not pooled or len(vocab) > self.vocab_size + 2:
            return False
        ctx_u = frozenset([(HERE, body), (FORWARD, u)])
        self.latent_groups[ctx_u] = {"vocab": vocab, "outcomes": outcomes, "hyps": [],
                                     "map": None, "prior": pooled, "augmented": set()}
        hyps = self._enumerate_hypotheses(ctx_u, vocab, outcomes)
        if not hyps:
            del self.latent_groups[ctx_u]
            return False
        # where both worlds leave the orientation open (an absent object makes a
        # literal vacuous), prefer ENABLING laws: the conjunction gates the
        # special outcome (reward / episode end), failure is the default
        best = min(hyps, key=lambda h: (len(h[0]), -h[1][2], not h[1][3],
                                        sum(1 for _e, pol in h[0] if not pol)))[0]
        g = self.latent_groups[ctx_u]
        g.update({"hyps": hyps, "map": best})
        self.latents.append(best)
        self.value_profiles[u] = self.value_profiles[part1[0]]
        self.card_world[u] = "synthesis"
        self.synthetic_roles[u] = {"parts": (part1[0], part2[0]), "law": best, "ctx": ctx_u}
        self._augment_unified(ctx_u)
        self._reconsolidate()
        self._info_version += 1
        return True

    def _augment_unified(self, ctx_u):
        """Schemas are learned from plain occurrences: give each pooled piece of
        evidence the events of the law's positive literals about objects absent
        from its world, so that plain evaluation equals scoped evaluation."""
        g = self.latent_groups[ctx_u]
        law = g["map"]
        occs = [o for o in self.occurrences.get(ctx_u, []) if o not in g["augmented"]]
        augmented = []
        for sig, snap, present in g["prior"]:
            extra = {e for e, pol in law if pol and event_target(e) not in present}
            augmented.append((sig, frozenset(snap | extra)))
        g["augmented"] = set(augmented)
        self.occurrences[ctx_u] = list(dict.fromkeys(augmented)) + occs

    # ------------------------------------------------------------------ #
    # STRUCTURAL ANALOGY: whole role mappings as competing hypotheses
    # ------------------------------------------------------------------ #
    def _hypothesis(self, base, assign, cost, values):
        lens = dict(base)
        lens.update(assign)
        for v in values:
            lens.setdefault(v, v)                    # unassigned: a new object
        return {"lens": lens, "assign": dict(assign), "cost": cost,
                "prior": math.exp(-cost), "alive": True, "mem": set(), "mem_before": set()}

    def _structural_hypotheses(self, raw, body_pos, floor_raw, wall_raw, base):
        """Enumerate whole role mappings. Each known role (a value with a stored
        profile) is given to one new value or left unmapped; mappings are
        one-to-one. Cost = profile distance + a penalty when multiplicities
        differ (relations matter more than attributes, so role swaps between
        objects of different multiplicity stay possible) + a penalty for every
        role left unmapped. Prior = exp(-cost)."""
        values = {v for row in raw for v in row}
        targets = sorted(values - set(base) - self.known_values, key=str)
        prof_t = {t: self._profile(raw, t, floor_raw, wall_raw, body_pos) for t in targets}
        roles = sorted(self.value_profiles, key=str)
        cands = {}
        for s_val in roles:
            mult_s, hist_s = self.value_profiles[s_val]
            options = []
            for t in targets:
                mult_t, hist_t = prof_t[t]
                c = sum(abs(a - b) for a, b in zip(hist_t, hist_s))
                if s_val in self.synthetic_roles:      # as close as its nearest part
                    for part in self.synthetic_roles[s_val]["parts"]:
                        hist_p = self.value_profiles[part][1]
                        c = min(c, sum(abs(a - b) for a, b in zip(hist_t, hist_p)))
                    c -= self.synthesis_bonus          # Occam on the ontology
                c += self.mult_penalty if mult_t != mult_s else 0.0
                options.append((c, str(t), t))
            options.sort()
            cands[s_val] = [(c, t) for c, _k, t in options[:self.struct_candidates]]
            cands[s_val].append((self.unmapped_cost, None))
        hyps = self._rank_structures(roles, cands, targets) if np is not None else None
        if hyps is None:
            hyps, seen = [], set()
            for combo in itertools.product(*[cands[r] for r in roles]):
                chosen = [t for _c, t in combo if t is not None]
                if len(set(chosen)) != len(chosen):
                    continue                             # one-to-one
                assign = {t: r for r, (_c, t) in zip(roles, combo) if t is not None}
                key = frozenset(assign.items())
                if key in seen:
                    continue
                seen.add(key)
                cost = sum(c for c, _t in combo)
                used = set(assign.values())
                for u, info in self.synthetic_roles.items():
                    if u in used:                        # its parts are not "missing":
                        cost -= self.unmapped_cost * sum(  # they are the same kind
                            1 for part in info["parts"] if part not in used)
                    elif any(part in used for part in info["parts"]):
                        cost -= self.unmapped_cost       # nor is the union of a used part
                hyps.append((cost, str(sorted(map(str, key))), assign))
        hyps.sort(key=lambda h: (h[0], h[1]))
        kept = hyps[:self.max_struct_hyps]
        if not any(not assign for _c, _k, assign in kept):
            # the null structure ("nothing here is like anything I know") is
            # always kept: if every analogy fails, the agent falls back on it
            kept.append((self.unmapped_cost * len(roles), "", {}))
        out = [self._hypothesis(base, assign, cost, values) for cost, _k, assign in kept]
        total = sum(h["prior"] for h in out)
        for h in out:
            h["prior"] /= total
        return out

    def _rank_structures(self, roles, cands, targets):
        """Exact vectorised equivalent of the brute-force product enumeration.
        Costs of every whole-role mapping are summed as arrays in the same
        left-to-right order. Python objects are built only for mappings whose
        cost can still enter the top list. Returns None when not applicable."""
        n_roles = len(roles)
        limit = self.max_struct_hyps
        if n_roles == 0 or limit < 1:
            return None
        sizes = [len(cands[r]) for r in roles]
        total = 1
        for s in sizes:
            total *= s
        if total > 2_000_000:
            return None
        tindex = {t: i for i, t in enumerate(targets)}
        idx = np.arange(total, dtype=np.int64)
        stride = total
        cost = np.zeros(total, dtype=np.float64)
        T = []
        for j, r in enumerate(roles):
            stride //= sizes[j]
            ch = (idx // stride) % sizes[j]
            cj = np.array([c for c, _t in cands[r]], dtype=np.float64)
            tj = np.array([-1 if t is None else tindex[t] for _c, t in cands[r]],
                          dtype=np.int64)
            cost = cost + cj[ch]
            T.append(tj[ch])
        valid = np.ones(total, dtype=bool)
        for j in range(n_roles):
            mapped = T[j] >= 0
            for k in range(j + 1, n_roles):
                valid &= ~((T[j] == T[k]) & mapped)
        rindex = {r: j for j, r in enumerate(roles)}
        uc = self.unmapped_cost
        for u, info in self.synthetic_roles.items():
            parts = info["parts"]
            any_part = np.zeros(total, dtype=bool)
            for part in parts:
                if part in rindex:
                    any_part |= T[rindex[part]] >= 0
            ju = rindex.get(u)
            if ju is None:
                cost = np.where(any_part, cost - uc, cost)
                continue
            missing = np.zeros(total, dtype=np.int64)
            for part in parts:
                if part in rindex:
                    missing += (T[rindex[part]] < 0)
                else:
                    missing += 1
            reduced = cost - uc * missing
            cost = np.where(T[ju] >= 0, reduced, np.where(any_part, cost - uc, cost))
        pos = np.flatnonzero(valid)
        if len(pos) > limit:
            vc = cost[pos]
            thr = np.partition(vc, limit - 1)[limit - 1]
            pos = pos[vc <= thr]
        T_lists = [tj.tolist() for tj in T]
        cost_list = cost.tolist()
        hyps = []
        for i in pos.tolist():
            assign = {}
            for j, r in enumerate(roles):
                tj = T_lists[j][i]
                if tj >= 0:
                    assign[targets[tj]] = r
            key = frozenset(assign.items())
            hyps.append((cost_list[i], str(sorted(map(str, key))), assign))
        return hyps

    def _alive_hyps(self):
        return [h for h in self.struct_hyps if h["alive"]]

    def _set_struct_map(self, index, rebuild_memory=True):
        """Perceive the world through hypothesis `index` from now on."""
        self._struct_map = index
        h = self.struct_hyps[index]
        self.lens = h["lens"]
        self.analogical = set(h["assign"])
        self._row_memo = {}
        self._lens_epoch += 1
        self._graph = None
        self._evaluation = None
        if rebuild_memory:
            # working memory = what happened this episode, as THIS structure sees it
            self.wm = set(h["mem_before"])
        self.model_version += 1

    def _describe(self, g0, action, g1, reward, done, p=None, q=None):
        """Event descriptor of a transition without learning from it."""
        motor = self.motor_map.get(action)
        if p is None:
            p = self._locate_self(g0)
        if motor is None or p is None:
            return None
        t = add(p, motor)
        body = self._val(g0, p)
        if q is None:
            if self._val(g1, t) != self._val(g0, t) and self._val(g1, p) != self._val(g0, p):
                q = t
            else:
                q = p
        effect = []
        for c in (p, t):
            if c in self.volatile:
                continue
            old, new = self._val(g0, c), self._val(g1, c)
            token = KEEP if new == old else (SELF if new == body else new)
            effect.append((canon(sub(c, p), motor), token))
        signature = (frozenset(effect), canon(sub(q, p), motor),
                     round(float(reward), 2), bool(done))
        return self._local_context(g0, p, motor), signature

    def _translate_with(self, raw, lens, memo):
        out = []
        for row in raw:
            tr = memo.get(row)
            if tr is None:
                tr = memo[row] = tuple(lens.get(v, v) for v in row)
            out.append(tr)
        return tuple(out)

    def _struct_update(self, raw0, action, raw1, reward, done, g0_map=None, g1_map=None):
        """Test every surviving structure on this transition: a structure whose
        (old) model predicts something else than what happened is eliminated.
        If the current one dies, switch to the most probable survivor and
        rebuild working memory through its eyes."""
        alive = self._alive_hyps()
        # my body is mapped identically by every structure: locate it once
        if g0_map is None:
            g0_map = self._translate(raw0)
        if g1_map is None:
            g1_map = self._translate(raw1)
        p = self._locate_self(g0_map)
        q1 = self._locate_self(g1_map)
        motor = self.motor_map.get(action)
        if p is None or motor is None:
            for h in alive:
                h["mem_before"] = set(h["mem"])
            return
        val = self._val
        t = (p[0] + motor[0], p[1] + motor[1])
        r0p, r0t, r1p, r1t = val(raw0, p), val(raw0, t), val(raw1, p), val(raw1, t)
        vol = self.volatile
        p_vol, t_vol = p in vol, t in vol
        rew = round(float(reward), 2)
        dn = bool(done)
        actual = (rew, dn)
        target_raw = r0t
        single = len(alive) <= 1 and not alive[0]["assign"]
        c_here = canon(HERE, motor)
        c_fwd = canon(sub(t, p), motor)
        died = False
        for h in alive:
            lens = h["lens"]
            h["mem_before"] = set(h["mem"])
            v0p, v0t = lens.get(r0p, r0p), lens.get(r0t, r0t)
            v1p, v1t = lens.get(r1p, r1p), lens.get(r1t, r1t)
            # only knowledge carried by THIS analogy can refute it: a wrong guess
            # about an object the structure leaves unmapped is just learning
            if not single and target_raw in h["assign"]:
                def cell(pos, lens=lens):
                    v = val(raw0, pos)
                    return v if v is None else lens.get(v, v)
                s = self._imagine_schema(v0p, v0t, p, motor, frozenset(h["mem"]), cell)
                if s is not None:
                    delta = s.self_delta
                    qp = p if delta == HERE else (
                        t if delta == FORWARD else add(p, uncanon(delta, motor)))
                    if (s.reward, s.done) != actual or (q1 is not None and q1 != qp):
                        h["alive"] = False
                        self.structures_eliminated += 1
                        died = True
                        continue
            q = q1
            if q is None:
                q = t if (v1t != v0t and v1p != v0p) else p
            effect = []
            items = []
            if not p_vol:
                effect.append((c_here, KEEP if v1p == v0p else (SELF if v1p == v0p else v1p)))
                items.append((HERE, v0p))
            if not t_vol:
                effect.append((c_fwd, KEEP if v1t == v0t else (SELF if v1t == v0p else v1t)))
                items.append((FORWARD, v0t))
            signature = (frozenset(effect), canon(sub(q, p), motor), rew, dn)
            h["mem"].add((frozenset(items), signature))
        if single:
            return
        if died:
            self.model_version += 1                  # information landscape changed
            self._info_version += 1
            if not self.struct_hyps[self._struct_map]["alive"]:
                best = max((i for i, h in enumerate(self.struct_hyps) if h["alive"]),
                           key=lambda i: self.struct_hyps[i]["prior"])
                before = dict(self.struct_hyps[self._struct_map]["assign"])
                self._set_struct_map(best)
                self.structure_switches += 1
                self._reattribute(before, self.struct_hyps[best]["assign"])

    def _reattribute(self, old_assign, new_assign):
        """A dead analogy filed this world's evidence under borrowed roles.
        Move every occurrence of THIS world to the roles the new structure sees,
        so what was learned here is not lost (and does not pollute the source)."""
        pi = {}
        for t in set(old_assign) | set(new_assign):
            a, b = old_assign.get(t, t), new_assign.get(t, t)
            if a != b:
                pi[a] = b
        if not pi:
            return
        moved = []
        for ctx in list(self.occ_world):
            log = self.occ_world[ctx]
            mine = [o for o in log if o[2] == self.world_id]
            ctx2 = frozenset((k, pi.get(v, v)) for k, v in ctx)
            if not mine or ctx2 == ctx:
                continue
            self.occ_world[ctx] = [o for o in log if o[2] != self.world_id]
            occs = self.occurrences.get(ctx, [])
            for sig, snap, wid in mine:
                if (sig, snap) in occs:
                    occs.remove((sig, snap))
                sig2 = self._remap_event((ctx, sig), pi)[1]
                snap2 = frozenset(self._remap_event(e, pi) for e in snap)
                moved.append((ctx2, sig2, snap2))
            if ctx in self.latent_groups:
                self._update_latent_group(ctx)
        changed = False
        for ctx2, sig2, snap2 in moved:
            occs = self.occurrences.setdefault(ctx2, [])
            if (sig2, snap2) not in occs:
                occs.append((sig2, snap2))
            log = self.occ_world.setdefault(ctx2, [])
            if (sig2, snap2, self.world_id) not in log:
                log.append((sig2, snap2, self.world_id))
            flags = self._flags_from(snap2)
            if ctx2 in self.latent_groups:
                changed |= self._update_latent_group(ctx2)
                continue
            for sig_old, snap_old in occs:
                if sig_old != sig2 and self._flags_from(snap_old) == flags:
                    if self._induce_latent(ctx2, snap2, snap_old, sig2, sig_old):
                        self._minimize_latents()
                        changed = True
                    break
        self.reattributed += len(moved)
        self._reconsolidate()

    def structure_report(self):
        """Surviving structures: (posterior, {new value: known role})."""
        alive = self._alive_hyps()
        z = sum(h["prior"] for h in alive)
        return [(round(h["prior"] / z, 3), dict(h["assign"])) for h in alive]

    # ---- information gain between surviving structures -------------------- #
    def _struct_rivals(self):
        """For every surviving rival of the current structure: its weight and the
        value permutation that turns my view of the world into its view."""
        alive = self._alive_hyps()
        if len(alive) < 2 or not self.struct_info:
            return []
        cur = self.struct_hyps[self._struct_map]
        z = sum(h["prior"] for h in alive)
        rivals = []
        for h in alive:
            if h is cur:
                continue
            pi = {}
            for t in self._struct_targets:
                a, b = cur["lens"].get(t, t), h["lens"].get(t, t)
                if a != b:
                    pi[a] = b
            if pi and h["prior"] / z >= self.min_rival_weight:
                rivals.append((h["prior"] / z, pi))
        rivals.sort(key=lambda r: -r[0])
        rivals = rivals[:self.max_rivals]
        return [(cur["prior"] / z, None)] + rivals if rivals else []

    @staticmethod
    def _remap_event(e, pi):
        ctx, (effect, delta, r, d) = e
        ctx2 = frozenset((k, pi.get(v, v)) for k, v in ctx)
        eff2 = frozenset((k, pi.get(v, v) if not isinstance(v, str) else v) for k, v in effect)
        return ctx2, (eff2, delta, r, d)

    def _match_local(self, here, ahead, flags):
        """Schema match from the two canonical cells + memory flags only."""
        by_target, generic = self._schema_index()
        best = None
        for sc in by_target.get(ahead, []) + generic:
            ok = True
            for key, v in sc.context:
                if key == FORWARD:
                    ok = ahead == v
                elif key == HERE:
                    ok = here == v
                elif isinstance(key[0], str):
                    ok = (key[1] in flags) == v
                else:
                    ok = False                       # wider contexts: not judged here
                if not ok:
                    break
            if ok and self._more_specific(sc, best):
                best = sc
        return best

    def _struct_eig(self, rivals, g, pos, a, m, pend, my_outcome, cache):
        """Expected information (bits) of doing `a` here, given the structures.
        A rival's outcome only needs my two cells and its memory flags, seen
        through its value permutation -- no full re-imagination."""
        motor = self.motor_map[a]
        t = (pos[0] + motor[0], pos[1] + motor[1])
        here, ahead = self._val(g, pos), self._val(g, t)
        mass = {my_outcome: rivals[0][0]}
        for i, (weight, pi) in enumerate(rivals[1:]):
            flags = cache.get(i)
            if flags is None:                        # cache is per imagined node
                flags = cache[i] = self._flags_from(
                    frozenset(self._remap_event(e, pi) for e in m))
            sc = self._match_local(pi.get(here, here), pi.get(ahead, ahead), flags)
            if sc is None:
                outcome = my_outcome
            else:
                q = pos if sc.self_delta == HERE else (t if sc.self_delta == FORWARD else None)
                outcome = (sc.reward, sc.done, None if sc.done else q)
            mass[outcome] = mass.get(outcome, 0.0) + weight
        return entropy(mass.values())

    def compact(self):
        """Drop rebuildable caches (imagined graph, evaluation) e.g. before copying."""
        if self.body_known():
            self.babble_tried = set()                # only needed while babbling
        self._row_memo = {}
        self._flags_memo = {}
        self._graph = None
        self._evaluation = None
        self._info_cache = {}
        self._novelty_cache = {}
        return self

    def _babble(self, grid):
        unknown = [a for a in range(self.n_actions) if a not in self.motor_map]
        fresh = [a for a in unknown if (grid, a) not in self.babble_tried]
        if fresh:
            a = self.rng.choice(fresh)
        else:
            known = [a for a in range(self.n_actions) if a in self.motor_map]
            a = self.rng.choice(known or list(range(self.n_actions)))
        self.babble_tried.add((grid, a))
        return a

    def end_episode(self):
        # Schedule a budgeted science episode: the last episode succeeded, so the
        # policy is secured, but harmless-looking rivals (possible superstitions)
        # remain unresolved.
        was_science = self.science_mode
        self.science_mode = False
        if was_science:
            self.model_version += 1
            self._info_version += 1
        if self.science and not was_science and self._episode_success:
            self._scheduling = True
            candidates = [(ctx, g) for ctx, g in self.latent_groups.items()
                          if self._superstition_events(g)]
            self._scheduling = False
            for ctx, g in candidates:
                if self.science_budget.get(ctx, 2) > 0:
                    self.science_budget[ctx] = self.science_budget.get(ctx, 2) - 1
                    self.science_mode = True
                    self.science_episodes += 1
                    self.model_version += 1
                    self._info_version += 1
                    break
        self._episode_success = False
        self._plan_tracked_version = -1
        self._episode_start = True
        for h in self.struct_hyps:               # each structure's view of this episode
            h["mem"], h["mem_before"] = set(), set()
        self.energy = self.max_energy
        self.wm = set()
        self.history = {}
        self.experiment = None
        self._evaluation = None

    def stats(self):
        return {
            "tick": self.tick,
            "schemas": len(self.schemas),
            "latents": len(self.latents),
            "delayed_links": sum(1 for k in self._map_links if k[1] > 0),
            "open_hypotheses": sum(len(v) for v in self.link_groups().values() if len(v) > 1)
                               + sum(len(g["hyps"]) for g in self.latent_groups.values()
                                     if len(g["hyps"]) > 1),
            "experiments": self.experiments_run,
            "volatile_cells": len(self.volatile),
            "replans": self.replans,
            "deferred_commits": self.deferred_commits,
            "science_episodes": self.science_episodes,
            "analogies": len(self.analogical),
            "structures_alive": len(self._alive_hyps()) if self.struct_hyps else 0,
            "structures_eliminated": self.structures_eliminated,
            "structure_switches": self.structure_switches,
            "unified_roles": len(self.synthetic_roles),
        }
