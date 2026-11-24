"""
Acore-WM data source: a self-contained re-implementation of the Acore
micro-universe physics (same rules as the A-CORE test suite), a random layout
generator and a fast trajectory generator (random + scripted policies,
multiprocessing).

Observations are grids of palette integers padded to MAX_H x MAX_W with PAD.
Palettes and action permutations are random per seed, so the world model has
to infer colour roles and motor meaning in context, as Acore does.

Self-check:  python envs.py
"""

import random
import time
from collections import deque
from multiprocessing import get_context

import numpy as np

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SYMBOLS = ["#", ".", "A", "a", "K", "D", "L", "F", "C", "S", "G", "X", "T", "B", "P", "Q", "R"]
TIMER_DELAY = 3
N_ACTIONS = 4
MAX_H, MAX_W = 8, 12
PAD = 0                                   # palette values are always >= 10
VALUE_BANDS = ((10, 100), (100, 200), (200, 300))
OBS_VOCAB = 300                           # observation token ids 0..299
REWARD_VALUES = (-1.0, -30.0, 100.0, 5.0)
REWARD_INDEX = {v: i for i, v in enumerate(REWARD_VALUES)}
CANDY = 5.0

DOOR_RULES = {
    "none": None,
    "lever": lambda m: "P" in m,
    "no_alarm": lambda m: "Q" not in m,
    "lever_not_alarm": lambda m: "P" in m and "Q" not in m,
    "lever_and_alarm": lambda m: {"P", "Q"} <= m,
    "lever_and_decoy": lambda m: {"P", "R"} <= m,
}


def make_palette(seed, lo=10, hi=100):
    values = random.Random(seed).sample(range(lo, hi), len(SYMBOLS))
    return dict(zip(SYMBOLS, values))


def make_action_perm(seed):
    dirs = DIRECTIONS[:]
    random.Random(seed + 1000).shuffle(dirs)
    return dirs


class GridWorld:
    """Same hidden physics as RawGridWorld in the test suite.
    State: (pos, has_key, gate_open, key_taken, timer, pressed)."""

    def __init__(self, layout, palette, action_dirs, seed=0, door_rule=None,
                 alarm_reward=-1.0, candy=False):
        self.layout = layout
        self.palette = palette
        self.dirs = action_dirs
        self.door_rule = door_rule
        self.alarm_reward = alarm_reward
        self.candy = candy
        self.rng = random.Random(seed)
        self._cache = {}
        for r, row in enumerate(layout):
            for c, ch in enumerate(row):
                if ch == "A":
                    self.start = (r, c)
        self.reset()

    def reset(self):
        self.state = (self.start, False, False, False, 0, frozenset())
        self.flicker_on = False
        self.trace = []
        return self.render()

    def cell(self, pos, state):
        ch = self.layout[pos[0]][pos[1]]
        if ch == "A" or (ch == "K" and state[3]) or (ch == "G" and state[2]) \
                or (ch == "Q" and pos in state[5]):
            return "."
        return ch

    def physics(self, state, d):
        pos, has_key, gate_open, key_taken, timer, pressed = state
        t = (pos[0] + d[0], pos[1] + d[1])
        ch = self.cell(t, state)
        reward, done, start_timer = -1.0, False, False
        if ch in "#FGB":
            pass
        elif ch == "L":
            reward = -30.0
        elif ch == "C":
            has_key = True
        elif ch == "S":
            gate_open = True
        elif ch == "T":
            start_timer = not gate_open and timer == 0
        elif ch in "PQR":
            pressed = pressed | {ch}
            if ch == "Q":
                reward = self.alarm_reward
                if self.candy:
                    pressed = pressed | {t}
        elif ch == "K":
            pos, has_key, key_taken = t, True, True
        elif ch == "D":
            if has_key or (self.door_rule is not None and self.door_rule(pressed)):
                pos, reward, done = t, 100.0, True
        elif ch == "X":
            pos, reward, done = t, 100.0, True
        else:
            pos = t
        if timer > 0:
            timer -= 1
            if timer == 0:
                gate_open = True
        if start_timer:
            timer = TIMER_DELAY
        return (pos, has_key, gate_open, key_taken, timer, pressed), reward, done, ch

    def render(self):
        """Padded (MAX_H, MAX_W) int16 array; treat as read-only."""
        key = (self.state, self.flicker_on)
        out = self._cache.get(key)
        if out is None:
            pos, has_key, _, key_taken, _, _ = self.state
            out = np.full((MAX_H, MAX_W), PAD, dtype=np.int16)
            for r, row in enumerate(self.layout):
                for c in range(len(row)):
                    ch = self.cell((r, c), self.state)
                    if (r, c) == pos:
                        out[r, c] = self.palette["a" if (has_key and key_taken) else "A"]
                    elif ch == "F":
                        out[r, c] = self.palette["F" if self.flicker_on else "#"]
                    else:
                        out[r, c] = self.palette[ch]
            self._cache[key] = out
        return out

    def step(self, action):
        self.state, reward, done, ch = self.physics(self.state, self.dirs[action])
        self.trace.append(ch)
        self.flicker_on = self.rng.random() < 0.5
        return self.render(), reward, done


def bump_path(env, symbol):
    """Directions (world physics, BFS) that walk next to `symbol` and bump it."""
    targets = {(r, c) for r, row in enumerate(env.layout) for c, ch in enumerate(row)
               if ch == symbol}
    if not targets:
        return None
    start = env.state
    frontier, seen = deque([(start, [])]), {start[0]}
    while frontier:
        state, path = frontier.popleft()
        pos = state[0]
        for d in DIRECTIONS:
            if (pos[0] + d[0], pos[1] + d[1]) in targets:
                return path + [d]
        for d in DIRECTIONS:
            s2, r, done, _ = env.physics(state, d)
            if s2[0] != pos and s2[0] not in seen and not done and r > -10:
                seen.add(s2[0])
                frontier.append((s2, path + [d]))
    return None


# ---------------------------------------------------------------------- #
# Random layouts
# ---------------------------------------------------------------------- #
def random_layout(rng):
    """Returns (layout rows, door rule name, candy flag)."""
    h, w = rng.randint(4, MAX_H), rng.randint(5, MAX_W)
    g = [["#"] * w for _ in range(h)]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            g[r][c] = "."
    gate_row = rng.randint(2, h - 3) if h >= 6 and rng.random() < 0.35 else None
    top_end = gate_row if gate_row is not None else h - 1
    upper = [(r, c) for r in range(1, top_end) for c in range(1, w - 1)]
    lower = []
    if gate_row is not None:
        g[gate_row] = ["#"] * w
        g[gate_row][rng.randint(1, w - 2)] = "G"
        lower = [(r, c) for r in range(gate_row + 1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(upper)
    rng.shuffle(lower)

    def put(ch, cells):
        if cells:
            r, c = cells.pop()
            g[r][c] = ch

    put("A", upper)
    rule, candy = "none", False
    if gate_row is not None:
        put(rng.choice("ST"), upper)
        goal = rng.choice("XD")
        put(goal, lower)
        if goal == "D":
            put(rng.choice("KC"), upper)
    else:
        goal = rng.choices(["X", "D", ""], [0.3, 0.5, 0.2])[0]
        if goal == "D":
            if rng.random() < 0.3:
                g[0][rng.randint(1, w - 2)] = "D"       # door in the outer wall
            else:
                put("D", upper)
            opener = rng.choice(["K", "C", "rule"])
            if opener == "rule":
                rule = rng.choice([k for k in DOOR_RULES if k != "none"])
                for ch in "PQR":
                    if ch in "PQ" or rng.random() < 0.5:
                        put(ch, upper)
                candy = rng.random() < 0.3
            else:
                put(opener, upper)
        elif goal == "X":
            put("X", upper)
    if rng.random() < 0.4:
        put("B", upper)
    for ch in rng.choices("QPR", k=rng.randint(0, 1)):
        put(ch, upper)
    for _ in range(rng.randint(0, 3)):
        put("L", upper if rng.random() < 0.7 or not lower else lower)
    for _ in range(rng.randint(0, 2)):
        put("#", upper)
    if rng.random() < 0.3:
        g[h - 1][rng.randint(0, w - 1)] = "F"
    return ["".join(row) for row in g], rule, candy


def make_world(seed):
    rng = random.Random(seed * 7919 + 13)
    layout, rule, candy = random_layout(rng)
    lo, hi = rng.choice(VALUE_BANDS)
    return GridWorld(layout, make_palette(seed, lo, hi), make_action_perm(seed), seed=seed,
                     door_rule=DOOR_RULES[rule], alarm_reward=CANDY if candy else -1.0,
                     candy=candy)


# ---------------------------------------------------------------------- #
# Trajectories
# ---------------------------------------------------------------------- #
def run_policy(env, rng, T, kind, eps=0.15):
    obs = np.zeros((T + 1, MAX_H, MAX_W), dtype=np.int16)
    act = np.zeros(T, dtype=np.int64)
    rew = np.zeros(T, dtype=np.int64)
    done = np.zeros(T, dtype=np.int64)
    mask = np.zeros(T, dtype=np.float32)
    obs[0] = env.reset()
    present = sorted({ch for row in env.layout for ch in row} & set("KCSTBPQRDXGL"))
    plan, a = [], rng.randrange(N_ACTIONS)
    for t in range(T):
        if kind == "random" or rng.random() < eps:
            a = rng.randrange(N_ACTIONS)
        elif kind == "sticky":
            if rng.random() > 0.7:
                a = rng.randrange(N_ACTIONS)
        else:
            if not plan:
                path = bump_path(env, rng.choice(present)) if present else None
                plan = list(path) if path else [rng.choice(DIRECTIONS)]
            a = env.dirs.index(plan.pop(0))
        o, r, dn = env.step(a)
        obs[t + 1], act[t], rew[t], done[t], mask[t] = o, a, REWARD_INDEX[r], int(dn), 1.0
        if dn:
            break
    return obs, act, rew, done, mask


def generate_sample(args):
    seed, T = args
    rng = random.Random(seed ^ 0x5EED)
    kind = rng.choices(["random", "sticky", "scripted"], [0.25, 0.15, 0.6])[0]
    return run_policy(make_world(seed), rng, T, kind)


def stack(samples):
    keys = ("obs", "act", "rew", "done", "mask")
    return {k: np.stack([s[i] for s in samples]) for i, k in enumerate(keys)}


def generate_batch(seeds, T, pool=None):
    args = [(s, T) for s in seeds]
    samples = pool.map(generate_sample, args) if pool else list(map(generate_sample, args))
    return stack(samples)


class TrajectoryStream:
    """On-the-fly batches with bounded prefetch. `consumed` (number of samples
    handed out) is enough to resume deterministically."""

    def __init__(self, batch_size, T, base_seed=0, workers=0, consumed=0, prefetch=4):
        self.batch_size, self.T, self.base_seed = batch_size, T, base_seed
        self.consumed = consumed
        self._next = consumed
        self.pool = get_context("spawn").Pool(workers) if workers > 0 else None
        self.pending = deque()
        if self.pool:
            for _ in range(prefetch):
                self._submit()

    def _seeds(self):
        s = range(self.base_seed + self._next, self.base_seed + self._next + self.batch_size)
        self._next += self.batch_size
        return s

    def _submit(self):
        args = [(s, self.T) for s in self._seeds()]
        self.pending.append(self.pool.map_async(generate_sample, args, chunksize=8))

    def next(self):
        if self.pool:
            samples = self.pending.popleft().get()
            self._submit()
            batch = stack(samples)
        else:
            batch = generate_batch(self._seeds(), self.T)
        self.consumed += self.batch_size
        return batch

    def close(self):
        if self.pool:
            self.pool.terminate()
            self.pool = None


# ---------------------------------------------------------------------- #
# Self-check against scripted sequences from the test suite
# ---------------------------------------------------------------------- #
U, D_, L_, R = (-1, 0), (1, 0), (0, -1), (0, 1)
KEY_CORRIDOR = ["######", "#KAD.#", "######"]
LAVA_CORNER = ["#####", "#.L.#", "#.AL#", "#...#", "#####"]
TIMER_ROOM = ["#######", "#A...T#", "###G###", "#..X..#", "#######"]


def _forced(env, seq):
    out = None
    for d in seq:
        out = env.step(env.dirs.index(d))
    return out


def self_check():
    pal, dirs = make_palette(0), make_action_perm(0)
    env = GridWorld(KEY_CORRIDOR, pal, dirs)
    _, r, done = _forced(env, [R, L_, R, R])
    assert done and r == 100.0, "key corridor"
    env = GridWorld(LAVA_CORNER, pal, dirs)
    _, r, _ = _forced(env, [U])
    assert r == -30.0, "lava"
    env = GridWorld(TIMER_ROOM, pal, dirs)
    o, _, _ = _forced(env, [R, R, R, R, L_, L_])
    assert o[2, 3] == pal["G"], "gate opened too early"
    o, _, _ = _forced(env, [R])
    assert o[2, 3] == pal["."], "gate not open after 3 steps"
    _, r, done = _forced(env, [D_, D_])
    assert done and r == 100.0, "timer exit"
    t0 = time.perf_counter()
    batch = generate_batch(range(64), 32)
    dt = time.perf_counter() - t0
    assert batch["obs"].shape == (64, 33, MAX_H, MAX_W)
    assert int(batch["obs"].max()) < OBS_VOCAB
    print(f"physics checks passed; 64 trajectories x 32 steps generated in {dt:.2f} s "
          f"(this machine, single process); mean episode length "
          f"{batch['mask'].sum(1).mean():.1f}")


if __name__ == "__main__":
    self_check()
