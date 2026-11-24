"""
Micro-universe tests for A-CORE v1.9
(multi-source synthesis: lever world A + alarm world B -> unseen world C).
Run:  python test_v1.py

The world only emits a grid of random integers; palette and motor meaning are
shuffled per seed.
"""

import copy
import gc
import pickle
import random
import time
from collections import deque

from acore_v1_9 import ACOREEngine, formula_true

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SYMBOLS = ["#", ".", "A", "a", "K", "D", "L", "F", "C", "S", "G", "X", "T", "B", "P", "Q", "R"]
TIMER_DELAY = 3


def make_palette(seed):
    values = random.Random(seed).sample(range(10, 100), len(SYMBOLS))
    return dict(zip(SYMBOLS, values))


def make_action_perm(seed):
    dirs = DIRECTIONS[:]
    random.Random(seed + 1000).shuffle(dirs)
    return dirs


class RawGridWorld:
    """
    Hidden physics (unknown to the agent):
      '#' wall: blocked, -1              'L' lava: pushed back, -30
      'K' visible key: picked up, body colour 'A'->'a'
      'C' chest: blocked, gives an INVISIBLE key
      'S' switch: blocked, opens every gate 'G' immediately
      'T' timer switch: blocked, opens every gate 'G' exactly 3 steps later
      'B' decoy button: blocked, does nothing at all
      'G' gate: blocked while closed
      'P','Q','R' buttons: blocked, -1, remembered by the world, NOTHING visible
      'D' door: opens (+100, done) if the agent has a key OR door_rule(pressed)
      'X' exit: +100, done
      'F' flicker wall: random colour changes (noise)
    State: (pos, has_key, gate_open, key_taken, timer, pressed)
    """

    def __init__(self, layout, palette, action_dirs, seed=0, door_rule=None, alarm_reward=-1.0,
                 candy=False):
        self.layout = layout
        self.door_rule = door_rule
        self.alarm_reward = alarm_reward
        self.candy = candy
        self.palette = palette
        self.dirs = action_dirs
        self.rng = random.Random(seed)
        for r, row in enumerate(layout):
            for c, ch in enumerate(row):
                if ch == "A":
                    self.start = (r, c)
        self.reset()

    def reset(self):
        self.state = (self.start, False, False, False, 0, frozenset())
        self.flicker_on = False
        self.trace = []                       # symbols the agent acted on
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
                if self.candy:                # a lure: eaten, it is gone
                    pressed = pressed | {t}   # (its cell is remembered as eaten)
        elif ch == "K":
            pos, has_key, key_taken = t, True, True
        elif ch == "D":
            if has_key or (self.door_rule is not None and self.door_rule(pressed)):
                pos, reward, done = t, 100.0, True
        elif ch == "X":
            pos, reward, done = t, 100.0, True
        else:
            pos = t
        if timer > 0:                         # countdown AFTER the action
            timer -= 1
            if timer == 0:
                gate_open = True
        if start_timer:
            timer = TIMER_DELAY
        return (pos, has_key, gate_open, key_taken, timer, pressed), reward, done, ch

    def render(self):
        key = (self.state, self.flicker_on)
        cache = self.__dict__.setdefault("_render_cache", {})
        if key not in cache:
            cache[key] = self._render()
        return [list(row) for row in cache[key]]

    def _render(self):
        pos, has_key, _, key_taken, _, _ = self.state
        obs = []
        for r, row in enumerate(self.layout):
            line = []
            for c in range(len(row)):
                ch = self.cell((r, c), self.state)
                if (r, c) == pos:
                    line.append(self.palette["a" if (has_key and key_taken) else "A"])
                elif ch == "F":
                    line.append(self.palette["F" if self.flicker_on else "#"])
                else:
                    line.append(self.palette[ch])
            obs.append(line)
        return obs

    def step(self, action):
        self.state, reward, done, ch = self.physics(self.state, self.dirs[action])
        self.trace.append(ch)
        self.flicker_on = self.rng.random() < 0.5
        return self.render(), reward, done

    def optimal_steps(self):
        start = (self.start, False, False, False, 0, frozenset())
        frontier, seen = deque([(start, 0)]), {start}
        while frontier:
            state, n = frontier.popleft()
            for d in DIRECTIONS:
                s2, r, done, _ = self.physics(state, d)
                if done:
                    return n + 1
                if s2 not in seen and r > -10:
                    seen.add(s2)
                    frontier.append((s2, n + 1))
        return None


OPEN_ROOM = ["#######",
             "#.....#",
             "#..A..#",
             "#.....#",
             "#######"]

LAVA_CORNER = ["#####",
               "#.L.#",
               "#.AL#",
               "#...#",
               "#####"]

KEY_CORRIDOR = ["######",
                "#KAD.#",
                "######"]

CHEST_CORRIDOR = ["######",
                  "#CAD.#",
                  "######"]

SWITCH_ROOM = ["#######",
               "#A...S#",
               "###G###",
               "#..X..#",
               "#######"]

TIMER_ROOM = ["#######",
              "#A...T#",
              "###G###",
              "#..X..#",
              "#######"]

DECOY_LAB = ["#######",          # no reward anywhere: only information drives behaviour
             "#.....#",
             "#BAT..#",
             "###G###",
             "#.....#",
             "#######"]

CHEST_DECOY = ["######",
               "#.B..#",
               "#.CAD#",
               "######"]

WORLD_FULL = ["##########F",
              "#A..L....C#",
              "#........L#",
              "#.L...B.T.#",
              "#####G#####",
              "#........D#",
              "###########"]

WORLD_FULL_2 = ["###########",
                "#C.......A#",
                "#.....L...#",
                "#.T.B..L..#",
                "###G#######",
                "#D........#",
                "###########"]


def run_episode(agent, env, max_steps=150):
    obs = env.reset()
    steps = hazards = 0
    done = False
    while not done and steps < max_steps:
        a = agent.act(obs)
        nxt, r, done = env.step(a)
        hazards += r <= -10
        agent.observe(obs, a, nxt, r, done)
        obs, steps = nxt, steps + 1
    agent.end_episode()
    return steps, done, hazards


def babble_until_body_known(agent, env, limit=60):
    obs = env.reset()
    for n in range(1, limit + 1):
        a = agent.act(obs)
        nxt, r, d = env.step(a)
        agent.observe(obs, a, nxt, r, d)
        obs = nxt
        if agent.body_known():
            agent.end_episode()
            return n
    return None


def trained_body(seed, **kw):
    pal, dirs = make_palette(seed), make_action_perm(seed)
    agent = ACOREEngine(n_actions=4, seed=seed, **kw)
    assert babble_until_body_known(agent, RawGridWorld(OPEN_ROOM, pal, dirs)) is not None
    return agent, pal, dirs


def forced(agent, env, obs, direction):
    a = env.dirs.index(direction)
    nxt, r, d = env.step(a)
    agent.observe(obs, a, nxt, r, d)
    return nxt, r, d


def forced_seq(agent, env, obs, seq):
    for d in seq:
        obs, r, done = forced(agent, env, obs, d)
    return obs, r, done


def path_to_bump(env, target_symbol):
    """Scripting helper (uses the WORLD's physics, not the agent): directions
    that walk to a cell next to `target_symbol` and bump into it."""
    targets = {(r, c) for r, row in enumerate(env.layout) for c, ch in enumerate(row)
               if ch == target_symbol}
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
    raise ValueError(target_symbol)


def press_sequence(agent, env, obs, symbols):
    r = done = None
    for sym in symbols:
        obs, r, done = forced_seq(agent, env, obs, path_to_bump(env, sym))
    return obs, r, done


def describe(formula, pal):
    inv = {v: k for k, v in pal.items()}
    parts = []
    for e, pol in sorted(formula, key=lambda l: str(l)):
        sym = inv.get(dict(e[0]).get((0, 1)), "?")
        parts.append(sym if pol else "~" + sym)
    return "&".join(sorted(parts))


def door_hypotheses(agent, pal, success_reward=100.0):
    """Readable 'door opens iff ...' formulas still in the version space."""
    out = []
    for ctx, hyps in agent.latent_hypotheses().items():
        if dict(ctx).get((0, 1)) != pal["D"]:
            continue
        for f, o_true in hyps:
            opens_if_true = o_true[2] == success_reward
            out.append(describe(f, pal) if opens_if_true else "NOT(" + describe(f, pal) + ")")
    return sorted(out)


def door_group(agent, pal):
    for ctx, g in agent.latent_groups.items():
        if dict(ctx).get((0, 1)) == pal["D"]:
            return g
    return None


def door_map(agent, pal, success_reward=100.0):
    """The MAP formula, written as 'door opens iff ...'."""
    g = door_group(agent, pal)
    if g is None:
        return None
    _f, o_true, _o_false = next(h for h in g["hyps"] if h[0] == g["map"])
    text = describe(g["map"], pal)
    return text if o_true[2] == success_reward else "NOT(" + text + ")"


def policy_resolved(agent, pal):
    """No surviving rival formula could change what the agent should do."""
    g = door_group(agent, pal)
    return g is not None and not agent._policy_relevant(g)


def with_body(grid, pos, value):
    return tuple(tuple(value if (r, c) == pos else v for c, v in enumerate(row))
                 for r, row in enumerate(grid))


def gate_hyps(agent, pal):
    return agent.link_groups().get((pal["G"], pal["."]), [])


def cause_target(agent, key, pal):
    inv = {v: k for k, v in pal.items()}
    return inv.get(dict(key[0][0]).get((0, 1)))


U, D_, L_, R = (-1, 0), (1, 0), (0, -1), (0, 1)


# ---------------------------------------------------------------------- #
# Regression (v1.2 / v1.3)
# ---------------------------------------------------------------------- #
def test_body_discovery(seed=0):
    pal, dirs = make_palette(seed), make_action_perm(seed)
    agent = ACOREEngine(n_actions=4, seed=seed)
    n = babble_until_body_known(agent, RawGridWorld(OPEN_ROOM, pal, dirs))
    assert n is not None and agent.self_values == {pal["A"]}
    assert all(agent.motor_map[a] == dirs[a] for a in range(4))
    return f"body found in {n} steps"


def test_one_shot_hazard_with_rotation_transfer(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(LAVA_CORNER, pal, dirs)
    obs, r, _ = forced(agent, env, env.reset(), R)
    grid = agent._grid(obs)
    pred = agent.predict(grid, agent._locate_self(grid), dirs.index(U))
    assert pred is not None and pred[2] == -30.0
    assert dirs[agent.act(obs)] in [D_, L_]
    return "ok"


def test_visible_carrying_state(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(KEY_CORRIDOR, pal, dirs)
    _, _, done = forced_seq(agent, env, env.reset(), [R, L_, R, R])
    assert done and pal["a"] in agent.self_values and not agent.latents
    return "ok"


def test_same_step_link(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(SWITCH_ROOM, pal, dirs)
    forced_seq(agent, env, env.reset(), [R] * 4)
    hyps = gate_hyps(agent, pal)
    assert len(hyps) == 1 and hyps[0][1] == 0 and cause_target(agent, hyps[0], pal) == "S"
    return "ok"


def test_hidden_state(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(CHEST_CORRIDOR, pal, dirs)
    forced_seq(agent, env, env.reset(), [R, L_, R])
    assert len(agent.latents) == 1
    agent.end_episode()
    steps, ok, _ = run_episode(agent, RawGridWorld(CHEST_CORRIDOR, pal, dirs))
    assert ok and steps == 2
    return "ok"


# ---------------------------------------------------------------------- #
# NEW in v1.4
# ---------------------------------------------------------------------- #
def test_delayed_link(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(TIMER_ROOM, pal, dirs)
    obs = env.reset()
    obs, _, _ = forced_seq(agent, env, obs, [R, R, R, R])     # press timer (no visible change)
    obs, _, _ = forced_seq(agent, env, obs, [L_, L_])          # gate still closed
    assert obs[2][3] == pal["G"]
    obs, _, _ = forced(agent, env, obs, R)                      # 3rd step: gate opens
    assert obs[2][3] == pal["."]
    hyps = gate_hyps(agent, pal)
    assert len(hyps) == 1, f"version space: {[(cause_target(agent, k, pal), k[1]) for k in hyps]}"
    assert cause_target(agent, hyps[0], pal) == "T" and hyps[0][1] == 3, "wrong cause or lag"
    obs, r, done = forced_seq(agent, env, obs, [D_, D_])       # walk through to the exit
    assert done and r == 100.0
    # Imagination: after pressing, the gate opens exactly on the 3rd step
    agent.end_episode()
    g = agent._grid(RawGridWorld(TIMER_ROOM, pal, dirs).render())
    g = with_body(with_body(g, (1, 1), pal["."]), (1, 4), pal["A"])
    state = (g, (1, 4), frozenset(), ())
    opened_at = None
    for step, d in enumerate([R, L_, R, L_, R], start=1):
        g2, q, _, _, m, pend, _, _ = agent.predict(state[0], state[1], dirs.index(d), state[2], state[3])
        state = (g2, q, m, pend)
        if g2[2][3] == pal["."] and opened_at is None:
            opened_at = step - 1      # steps AFTER the press
    assert opened_at == 3, f"imagined gate opened after {opened_at} steps"
    # Plan to the exit from scratch: press, wait, go
    steps, ok, _ = run_episode(agent, env)
    assert ok and steps == env.optimal_steps(), f"{steps} vs opt {env.optimal_steps()}"
    return f"cause=T lag=3 learned in one shot; plan uses the timer ({steps} steps, optimal)"


def test_delayed_experiment_design(seed=0, info_scale=12.0):
    agent, pal, dirs = trained_body(seed, info_scale=info_scale)
    env = RawGridWorld(DECOY_LAB, pal, dirs)
    # Confounded observation: decoy and timer pressed back to back
    obs, _, _ = forced_seq(agent, env, env.reset(), [L_, R, U, D_, U])
    assert obs[3][3] == pal["."], "test invalid: gate should be open"
    hyps = gate_hyps(agent, pal)
    names = sorted((cause_target(agent, k, pal), k[1]) for k in hyps)
    n_start = len(names)
    assert ("B", 4) in names and ("T", 3) in names, f"version space too small: {names}"
    agent.end_episode()
    # Free behaviour in the same lab: no reward exists, only information
    env.reset()
    obs = env.render()
    for step in range(1, 41):
        a = agent.act(obs)
        nxt, r, done = env.step(a)
        agent.observe(obs, a, nxt, r, done)
        obs = nxt
        hyps = gate_hyps(agent, pal)
        if len(hyps) == 1:
            break
    names = [(cause_target(agent, k, pal), k[1]) for k in hyps]
    assert names == [("T", 3)], f"not resolved: {names}"
    presses = [i for i, ch in enumerate(env.trace) if ch in "BT"]
    first = presses[0]
    confounded = any(env.trace[j] in "BT" for j in range(first + 1, min(first + 6, len(env.trace))))
    assert not confounded, f"confounded experiment: {env.trace}"
    return f"{n_start} hypotheses -> 1 after {step} steps, clean test of '{env.trace[first]}'"


def test_latent_experiment_design(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(CHEST_DECOY, pal, dirs)
    # door fails, then BOTH decoy and chest are touched, then door opens
    forced_seq(agent, env, env.reset(), [R, U, L_, D_, L_, R])
    start = door_hypotheses(agent, pal)
    assert "B" in start and "C" in start, f"hypotheses: {start}"
    agent.end_episode()
    episodes, traces = resolve_freely(agent, env, pal, max_episodes=4)
    assert episodes is not None and door_map(agent, pal) == "C", \
        f"not resolved: map={door_map(agent, pal)} {door_hypotheses(agent, pal)}"
    assert "B" not in traces[-1][2], f"still touching the decoy: {traces[-1][2]}"
    return (f"{start} -> MAP 'C' in {episodes} episode(s); left (policy-irrelevant): "
            f"{door_hypotheses(agent, pal)}")


COMPOUND_LAB = ["#########",
                "#P.....Q#",
                "#...A...#",
                "#R.....D#",
                "#########"]

COMPOUND_WORLD = ["##########F",
                  "#P...L...Q#",
                  "#....A....#",
                  "#..L...L..#",
                  "#R.......D#",
                  "###########"]


def resolve_freely(agent, env, pal, max_episodes=8):
    """Let the agent act on its own until one hypothesis is left."""
    traces = []
    for episode in range(1, max_episodes + 1):
        steps, ok, _ = run_episode(agent, env)
        traces.append((steps, ok, list(env.trace)))
        if policy_resolved(agent, pal):
            return episode, traces
    return None, traces


def test_compound_and(seed=0, info_scale=12.0):
    agent, pal, dirs = trained_body(seed, info_scale=info_scale)
    env = RawGridWorld(COMPOUND_LAB, pal, dirs, door_rule=lambda m: {"P", "Q"} <= m)
    obs = env.reset()
    obs, r, _ = press_sequence(agent, env, obs, ["D"])             # fails
    obs, r, done = press_sequence(agent, env, obs, ["P", "Q", "R", "D"])
    assert done, "test invalid"
    start = door_hypotheses(agent, pal)
    assert "P&Q" in start and len(start) >= 4, f"version space: {start}"
    agent.end_episode()
    episodes, traces = resolve_freely(agent, env, pal)
    if info_scale == 0:
        return episodes, door_map(agent, pal)
    assert episodes is not None and door_map(agent, pal) == "P&Q", \
        f"not resolved: map={door_map(agent, pal)} {door_hypotheses(agent, pal)}"
    return (f"{len(start)} hypotheses -> MAP 'P&Q' in {episodes} episode(s); "
            f"left: {door_hypotheses(agent, pal)}")


def test_compound_and_not(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(COMPOUND_LAB, pal, dirs, door_rule=lambda m: "P" in m and "Q" not in m)
    obs = env.reset()
    obs, _, _ = press_sequence(agent, env, obs, ["D"])                  # fails
    obs, _, done = press_sequence(agent, env, obs, ["Q", "P", "R", "D"])  # fails too
    assert not done
    agent.end_episode()
    obs = env.reset()
    obs, _, done = press_sequence(agent, env, obs, ["P", "R", "D"])     # opens
    assert done
    start = door_hypotheses(agent, pal)
    assert "P&~Q" in start and "R&~Q" in start, f"version space: {start}"
    agent.end_episode()
    episodes, traces = resolve_freely(agent, env, pal)
    assert episodes is not None and door_map(agent, pal) == "P&~Q", \
        f"not resolved: map={door_map(agent, pal)} {door_hypotheses(agent, pal)}"
    for _steps, _ok, trace in traces:
        assert "Q" not in trace, f"pressed the forbidden button: {trace}"
    test_trace = traces[-1][2]
    before_door = set(test_trace[:test_trace.index("D")]) & {"P", "R"}
    assert len(before_door) == 1, f"non-discriminating experiment: {test_trace}"
    steps, ok, _ = run_episode(agent, env)
    assert ok and steps == env.optimal_steps(), f"{steps} vs opt {env.optimal_steps()}"
    return (f"{start} -> MAP 'P&~Q' in {episodes} episode(s); experiment touched only "
            f"{sorted(before_door)} and never Q; then optimal ({steps} steps)")


def test_autonomous_compound_world(seed=0):
    pal, dirs = make_palette(seed), make_action_perm(seed)
    agent = ACOREEngine(n_actions=4, seed=seed)
    env = RawGridWorld(COMPOUND_WORLD, pal, dirs, seed=seed,
                       door_rule=lambda m: "P" in m and "Q" not in m)
    log = [run_episode(agent, env) for _ in range(12)]
    opt = env.optimal_steps()
    assert all(s == opt and h == 0 for s, _, h in log[-3:]), f"not optimal ({opt}): {log}"
    hyps = door_hypotheses(agent, pal)
    assert "Q" not in env.trace, "exploits by pressing Q"
    return (f"episodes={[(s, int(ok)) for s, ok, _ in log]} opt={opt} | "
            f"MAP={door_map(agent, pal)} | version space={hyps}")


REVERSIBILITY_LAB = ["#########",
                     "#K..A..S#",
                     "#P.....R#",
                     "####G####",
                     "#.......#",
                     "#########"]


def place(env, pos):
    env.state = (pos,) + env.state[1:]
    return env.render()


def test_reversibility_classification(seed=0):
    agent, pal, dirs = trained_body(seed)
    env = RawGridWorld(REVERSIBILITY_LAB, pal, dirs)
    obs = env.reset()
    obs, _, _ = press_sequence(agent, env, obs, ["S", "P", "K"])   # experience once
    obs, _, _ = forced_seq(agent, env, obs, [R, L_, D_, U])         # move with the key
    agent.end_episode()
    U_, D__ = dirs.index(U), dirs.index(D_)
    Lr, Rr = dirs.index(L_), dirs.index(R)
    env.reset()
    labels = {
        "move": agent.reversibility(place(env, (1, 4)), Lr),
        "switch (gate opens)": agent.reversibility(place(env, (1, 6)), Rr),
        "button, 1st time": agent.reversibility(place(env, (2, 2)), Lr),
        "never-touched object": agent.reversibility(place(env, (2, 6)), Rr),
        "key (consumed, body changes)": agent.reversibility(place(env, (1, 2)), Lr),
    }
    expected = {"move": "reversible", "switch (gate opens)": "irreversible",
                "button, 1st time": "irreversible", "never-touched object": "unknown",
                "key (consumed, body changes)": "irreversible"}
    assert labels == expected, f"{labels}"
    # pressing the button again in the same episode is a harmless repeat
    obs = place(env, (2, 2))
    forced(agent, env, obs, L_)
    assert agent.reversibility(place(env, (2, 2)), Lr) == "reversible", "repeat not reversible"
    return labels


def test_superstition_science(seed=0, science=True):
    """Confounded history: Q and R were always pressed together, so 'P and not Q'
    and 'P and not R' explain everything and lead to the same policy."""
    agent, pal, dirs = trained_body(seed, science=science)
    env = RawGridWorld(COMPOUND_LAB, pal, dirs, door_rule=lambda m: "P" in m and "Q" not in m)
    obs = env.reset()
    obs, _, _ = press_sequence(agent, env, obs, ["D", "Q", "R", "P", "D"])   # fails
    agent.end_episode()
    obs = env.reset()
    _, _, done = press_sequence(agent, env, obs, ["P", "D"])                  # opens
    assert done
    start = door_hypotheses(agent, pal)
    assert "P&~Q" in start and "P&~R" in start, f"version space: {start}"
    assert policy_resolved(agent, pal), "rivals should be policy-irrelevant here"
    agent.end_episode()
    for _ in range(4):
        run_episode(agent, env)
    final = door_hypotheses(agent, pal)
    if not science:
        return final
    assert final == ["P&~Q"], f"superstition survived: {final}"
    steps, ok, _ = run_episode(agent, env)
    assert ok and steps == env.optimal_steps()
    return f"{start} -> {final} after {agent.science_episodes} science episode(s); then optimal"


def test_autonomous_full_world_and_transfer(seed=0):
    pal, dirs = make_palette(seed), make_action_perm(seed)
    agent = ACOREEngine(n_actions=4, seed=seed)
    env1 = RawGridWorld(WORLD_FULL, pal, dirs, seed=seed)
    log = [run_episode(agent, env1) for _ in range(10)]
    opt1 = env1.optimal_steps()
    assert all(s == opt1 and h == 0 for s, _, h in log[-3:]), f"not optimal ({opt1}): {log}"
    assert (0, 10) in agent.volatile, "flicker not abstracted"
    hyps = gate_hyps(agent, pal)
    assert len(hyps) == 1 and cause_target(agent, hyps[0], pal) == "T" and hyps[0][1] == 3
    env2 = RawGridWorld(WORLD_FULL_2, pal, dirs, seed=seed)
    steps, ok, haz = run_episode(agent, env2)
    opt2 = env2.optimal_steps()
    assert ok and haz == 0 and steps <= opt2 + 2, f"transfer: {steps} vs opt {opt2}, haz {haz}"
    return (f"episodes={[(s, int(ok), h) for s, ok, h in log]} opt={opt1} | "
            f"new world {steps} (opt {opt2}) | {agent.stats()}")


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# v1.7: analogical transfer to a never-seen world
# ---------------------------------------------------------------------- #
def make_palette_in(seed, lo, hi):
    values = random.Random(seed).sample(range(lo, hi), len(SYMBOLS))
    return dict(zip(SYMBOLS, values))


ANALOGY_SRC = ["#####D#####",
               "#.........#",
               "#.L..P..L.#",
               "#....A....#",
               "#.Q.L.L.Q.#",
               "#....L....#",
               "###########"]

ANALOGY_TGT = ["###########",
               "#.L.......#",
               "#....Q..L.D",
               "#.L..A....#",
               "#...P...L.#",
               "#.....Q.L.#",
               "###########"]

ANALOGY_AMBIG = ["###########",        # a decoy that looks exactly like the lever
                 "#.L.......#",
                 "#....Q..L.D",
                 "#.L..A.R..#",
                 "#...P...L.#",
                 "#.....Q.L.#",
                 "###########"]

_SOURCE_CACHE = {}


def rule_p_not_q(m):
    return "P" in m and "Q" not in m


def train_source(seed, episodes=12, **kw):
    pal, dirs = make_palette(seed), make_action_perm(seed)
    agent = ACOREEngine(n_actions=4, seed=seed, **kw)
    env = RawGridWorld(ANALOGY_SRC, pal, dirs, seed=seed, door_rule=rule_p_not_q)
    log = [run_episode(agent, env) for _ in range(episodes)]
    return agent, pal, dirs, env, log


def trained_source(seed, **kw):
    """Train once per seed, hand out independent copies (keeps the suite fast)."""
    key = (seed, tuple(sorted(kw.items())))
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = train_source(seed, **kw)
    agent, pal, dirs, env, log = _SOURCE_CACHE[key]
    return pickle.loads(pickle.dumps(agent.compact())), pal, dirs, env, log


TARGET_STEPS = 80     # cap for episodes in never-seen worlds (successes take 7-29 steps)


def target_env(layout, seed, dirs, rule=rule_p_not_q):
    pal_t = make_palette_in(seed + 500, 100, 200)          # disjoint colours
    return RawGridWorld(layout, pal_t, dirs, seed=seed, door_rule=rule), pal_t


def test_analogical_transfer_first_try(seed=0):
    agent, pal, dirs, src_env, log = trained_source(seed)
    source_rule = door_map(agent, pal)
    env, pal_t = target_env(ANALOGY_TGT, seed, dirs)
    steps, ok, hazards = run_episode(agent, env)
    roles = roles_found(agent, pal, pal_t)
    assert {k: roles[k] for k in "PQDL"} == {"P": "P", "Q": "Q", "D": "D", "L": "L"}, f"roles {roles}"
    assert ok and hazards == 0 and "Q" not in env.trace, f"first try failed: {steps} {env.trace}"
    assert steps <= env.optimal_steps() + 1, f"{steps} vs opt {env.optimal_steps()}"
    return (f"source rule {source_rule}; never-seen colours + layout: roles {roles} -> "
            f"first try {steps} steps (opt {env.optimal_steps()}), 0 hazards")


def test_analogy_ablation(seed=0):
    results = {}
    for label, analogy in (("analogy", True), ("no analogy", False)):
        # same source training; analogy only matters once a new world appears
        agent, pal, dirs, _src, _log = trained_source(seed)
        agent.analogy_enabled = analogy
        env, _ = target_env(ANALOGY_TGT, seed, dirs)
        results[label] = [run_episode(agent, env, TARGET_STEPS)[:3] for _ in range(2)]
    pal, dirs = make_palette(seed), make_action_perm(seed)
    fresh, _, _ = trained_body(seed)
    env, _ = target_env(ANALOGY_TGT, seed, dirs)
    results["fresh"] = [run_episode(fresh, env, TARGET_STEPS)[:3] for _ in range(2)]
    on, off, new = results["analogy"], results["no analogy"], results["fresh"]
    assert on[0][1] and on[0][2] == 0
    assert sum(s for s, _, _ in on) < sum(s for s, _, _ in off), results
    assert sum(s for s, _, _ in on) < sum(s for s, _, _ in new), results
    return {k: [(s, int(ok), h) for s, ok, h in v] for k, v in results.items()}


def test_ambiguous_analogy(seed=0):
    agent, pal, dirs, _src, _log = trained_source(seed)
    env, pal_t = target_env(ANALOGY_AMBIG, seed, dirs)
    steps, ok, hazards = run_episode(agent, env)
    roles = roles_found(agent, pal, pal_t)
    assert ok and hazards == 0 and "Q" not in env.trace, f"{steps} {env.trace}"
    assert roles["P"] == "P", f"real lever lost: {roles}"
    touched_decoy = "R" in env.trace
    if touched_decoy:
        assert roles["R"] == "new", f"decoy not revoked: {roles}"
    story = "decoy tried, its structure eliminated" if touched_decoy else "lever tried first"
    return (f"lever look-alike decoy (2 rival structures): "
            f"{''.join(c for c in env.trace if c not in '.#')} -> {story}; solved in {steps} steps")


def roles_found(agent, pal_src, pal_tgt):
    inv_s = {v: k for k, v in pal_src.items()}
    inv_t = {v: k for k, v in pal_tgt.items()}
    return {inv_t[t]: inv_s.get(s, "new") for t, s in agent.lens.items() if t in inv_t}


ANALOGY_INVERTED = ["###########",      # same look, opposite physics: the 'alarms' are keys
                    "#.L.......#",
                    "#....Q..L.D",
                    "#.L..A....#",
                    "#...P...L.#",
                    "#.....Q.L.#",
                    "###########"]


def named_structures(agent, pal_s, pal_t, top=6):
    inv_s = {v: k for k, v in pal_s.items()}
    inv_t = {v: k for k, v in pal_t.items()}
    out = []
    for w, assign in agent.structure_report()[:top]:
        out.append((w, {inv_t.get(t, t): inv_s.get(s_, s_) for t, s_ in sorted(assign.items(), key=str)}))
    return out


def test_structural_hypothesis_space(seed=0):
    agent, pal, dirs, _src, _log = trained_source(seed)
    env, pal_t = target_env(ANALOGY_AMBIG, seed, dirs)
    obs = env.reset()
    a = agent.act(obs)                       # the probe move binds the new world
    nxt, r, d = env.step(a)
    agent.observe(obs, a, nxt, r, d)
    structs = [assign for _w, assign in named_structures(agent, pal, pal_t, top=50)]
    lever_is_P = any(m.get("P") == "P" and m.get("Q") == "Q" for m in structs)
    lever_is_R = any(m.get("R") == "P" and m.get("Q") == "Q" for m in structs)
    swapped = any(m.get("P") == "Q" and m.get("Q") == "P" for m in structs)
    null = any(not m for m in structs)
    assert lever_is_P and lever_is_R and swapped and null, structs[:6]
    top = named_structures(agent, pal, pal_t, top=3)
    return f"{len(structs)} whole-role structures, e.g. {top}"


def test_inverted_world_role_swap(seed=0):
    agent, pal, dirs, src_env, _log = trained_source(seed)
    rule_before = door_map(agent, pal)
    if rule_before != "P&~Q":
        # a role swap needs both roles: this source never learned what the alarm does
        return f"not applicable: source rule is {rule_before} (alarm never tried in the source)"
    env, pal_t = target_env(ANALOGY_INVERTED, seed, dirs,
                            rule=lambda m: "Q" in m and "P" not in m)   # opposite physics
    log = [run_episode(agent, env, TARGET_STEPS) for _ in range(4)]
    roles = roles_found(agent, pal, pal_t)
    assert any(ok for _, ok, _ in log[:2]), f"no adaptation by episode 2: {log}"
    assert roles["P"] == "Q" and roles["Q"] == "P", f"role swap not found: {roles}"
    assert all(ok and s == env.optimal_steps() for s, ok, _ in log[-2:]), log
    back = [run_episode(agent, src_env) for _ in range(2)]
    opt = src_env.optimal_steps()
    assert all(ok and s <= opt + 1 and h == 0 for s, ok, h in back), f"source damaged: {back}"
    assert door_map(agent, pal) == rule_before
    return (f"inverted world {[(s, int(ok)) for s, ok, _ in log]} -> structure "
            f"'P is the alarm, Q is the lever' ({named_structures(agent, pal, pal_t, 1)[0][0]:.2f});"
            f" back in source {[s for s, _, _ in back]} (opt {opt}), rule {rule_before}")


def test_alien_world_is_contained(seed=0):
    """No known structure fits: the door needs the lever AND the decoy."""
    agent, pal, dirs, src_env, _log = trained_source(seed)
    rule_before = door_map(agent, pal)
    env, pal_t = target_env(ANALOGY_AMBIG, seed, dirs, rule=lambda m: {"P", "R"} <= m)
    log = [run_episode(agent, env, TARGET_STEPS) for _ in range(5)]
    assert sum(ok for _, ok, _ in log) >= 3, f"never adapted: {log}"
    back = [run_episode(agent, src_env) for _ in range(2)]
    opt = src_env.optimal_steps()
    assert all(ok and s <= opt + 1 and h == 0 for s, ok, h in back), f"source damaged: {back}"
    assert door_map(agent, pal) == rule_before
    return (f"alien rule: {[(s, int(ok)) for s, ok, _ in log]}, structures eliminated "
            f"{agent.structures_eliminated} -> back in source {[s for s, _, _ in back]} "
            f"(opt {opt}), rule {rule_before}")


def test_structural_information_ablation(seed=0):
    """Same ambiguous world with and without information gain between structures."""
    out = {}
    for label, flag in (("with IG", True), ("no IG", False)):
        agent, pal, dirs, _src, _log = trained_source(seed)
        agent.struct_info = flag
        env, _ = target_env(ANALOGY_AMBIG, seed, dirs)
        out[label] = [run_episode(agent, env)[:2]]
    assert all(v[0][1] for v in out.values()), out
    return {k: [(s, int(ok)) for s, ok in v] for k, v in out.items()}


# ---------------------------------------------------------------------- #
# v1.9: multi-source synthesis
# ---------------------------------------------------------------------- #
WORLD_A = ["#####D#####",          # lever + door (no alarm): door iff lever
           "#.........#",
           "#.L..P..L.#",
           "#....A....#",
           "#...L.L...#",
           "#....L....#",
           "###########"]

WORLD_B = ["###########",          # alarm-candy + door (no lever): door iff not alarm
           "#..Q...L..#",
           "#.L.......D",
           "#....A..L.#",
           "#.L.....Q.#",
           "#....L....#",
           "###########"]

WORLD_C = ["##D########",          # both mechanisms, for the first time together
           "#..L....Q.#",
           "#.L....L..#",
           "#...QA....#",
           "#.L.....P.#",
           "#....L....#",
           "###########"]

CANDY = 5.0                         # the alarm is a lure: +5 when eaten (then it is gone)


def rule_lever(m):
    return "P" in m


def rule_no_alarm(m):
    return "Q" not in m


def world(layout, seed, dirs, lo, hi, rule, salt):
    pal = make_palette_in(seed + salt, lo, hi)
    return RawGridWorld(layout, pal, dirs, seed=seed, door_rule=rule, alarm_reward=CANDY,
                        candy=True), pal


def train_A(seed, episodes=12, **kw):
    pal, dirs = make_palette(seed), make_action_perm(seed)
    agent = ACOREEngine(n_actions=4, seed=seed, **kw)
    env = RawGridWorld(WORLD_A, pal, dirs, seed=seed, door_rule=rule_lever)
    log = [run_episode(agent, env) for _ in range(episodes)]
    return agent, pal, dirs, env, log


def train_B(agent, seed, dirs, episodes=10):
    env, pal = world(WORLD_B, seed, dirs, 100, 200, rule_no_alarm, 300)
    log = [run_episode(agent, env, TARGET_STEPS) for _ in range(episodes)]
    return env, pal, log


def rule_c(m):
    return "P" in m and "Q" not in m


_A_CACHE, _AB_CACHE = {}, {}


def trained_a(seed):
    if seed not in _A_CACHE:
        agent, pal_a, dirs, env_a, log_a = train_A(seed)
        _A_CACHE[seed] = (agent.compact(), pal_a, dirs)
    agent, pal_a, dirs = _A_CACHE[seed]
    return pickle.loads(pickle.dumps(agent)), pal_a, dirs


def trained_ab(seed, b_episodes=6):
    """Agent that lived in world A (lever) and then world B (alarm); copies."""
    key = (seed, b_episodes)
    if key not in _AB_CACHE:
        agent, pal_a, dirs = trained_a(seed)
        env_b, pal_b, log_b = train_B(agent, seed, dirs, b_episodes)
        _AB_CACHE[key] = (agent.compact(), pal_a, pal_b, dirs, env_b, log_b)
    agent, pal_a, pal_b, dirs, env_b, log_b = _AB_CACHE[key]
    return pickle.loads(pickle.dumps(agent)), pal_a, pal_b, dirs, env_b, log_b


def world_c(seed, dirs):
    return world(WORLD_C, seed, dirs, 200, 300, rule_c, 700)


def symbols_of(*pals):
    inv = {}
    for tag, pal in pals:
        inv.update({v: tag + k for k, v in pal.items()})
    return inv


def law_text(formula, inv):
    parts = []
    for e, pol in formula:
        sym = inv.get(dict(e[0]).get((0, 1)), "?")
        parts.append(sym if pol else "~" + sym)
    return "&".join(sorted(parts, key=lambda p: p.lstrip("~")))


def eaten(env):
    return "Q" in env.trace


def door_prediction(agent, env, seq):
    """Script a press sequence in a fresh episode, walk to the door and ask the
    agent's imagination what the bump will do, BEFORE doing it."""
    obs = env.reset()
    while True:                                   # let the agent find its body
        a = agent.notice_world(obs)
        if a is None:
            break
        nxt, r, d = env.step(a)
        agent.observe(obs, a, nxt, r, d)
        obs = nxt
    obs, _r, _d = press_sequence(agent, env, obs, seq)
    path = path_to_bump(env, "D")
    obs, _r, _d = forced_seq(agent, env, obs, path[:-1])
    g = agent._translate(tuple(map(tuple, obs)))
    pred = agent.predict(g, agent._locate_self(g), env.dirs.index(path[-1]),
                         frozenset(agent.wm), ())
    _nxt, _r, done = env.step(env.dirs.index(path[-1]))
    return (pred is not None and bool(pred[3])), bool(done)


COUNTERFACTUALS = ([], ["P"], ["Q"], ["Q", "P"], ["P", "Q"])


def counterfactual_score(agent, seed, dirs, seqs=COUNTERFACTUALS):
    blob = pickle.dumps(agent)
    out = {}
    for seq in seqs:
        env, _pal = world_c(seed, dirs)
        pred, real = door_prediction(pickle.loads(blob), env, seq)
        out["".join(seq) or "-"] = (pred, real)
    return out


def test_world_b_learns_its_own_door(seed=0):
    """In B the borrowed analogy (B door = A door, needs a lever B lacks) is
    pessimistic. Retests are per world, so the door is tried anyway; the
    contradiction kills the analogy and the evidence filed under A's door is
    moved to B's own door (A's law stays untouched)."""
    agent, pal_a, pal_b, dirs, env_b, log = trained_ab(seed)
    steps = [s for s, _ok, _h in log]
    opt = env_b.optimal_steps()
    assert log[-1][1] and steps[-1] <= opt + 1, f"B never solved: {steps}"
    assert agent.reattributed > 0, "no evidence moved after the analogy died"
    inv = symbols_of(("A:", pal_a), ("B:", pal_b))
    laws = {}
    for ctx, g in agent.latent_groups.items():
        door = inv.get(dict(ctx).get((0, 1)))
        _f, o_true, _o = next(h for h in g["hyps"] if h[0] == g["map"])
        text = law_text(g["map"], inv)
        laws[door] = text if o_true[2] == 100.0 else "NOT(" + text + ")"
    assert laws.get("A:D") == "A:P", laws
    assert laws.get("B:D") in ("~B:Q", "NOT(B:Q)"), laws
    return f"B episodes {steps} (opt {opt}); door laws {laws}; moved {agent.reattributed}"


def test_role_synthesis(seed=0):
    """Entering C, the two door roles (same profile, same outcomes, different
    worlds) are fused into ONE role whose law explains both worlds: P & ~Q."""
    agent, pal_a, pal_b, dirs, _env_b, _log = trained_ab(seed)
    env, pal_c = world_c(seed, dirs)
    run_episode(agent, env, 1)
    inv = symbols_of(("A:", pal_a), ("B:", pal_b), ("C:", pal_c))
    assert len(agent.synthetic_roles) == 1, agent.synthetic_roles
    (u, info), = agent.synthetic_roles.items()
    parts = sorted(inv.get(p) for p in info["parts"])
    law = law_text(info["law"], inv)
    assert parts == ["A:D", "B:D"], parts
    assert law == "A:P&~B:Q", law
    w, assign = agent.structure_report()[0]
    seen_as = {inv.get(t): ("UNIFIED" if r == u else inv.get(r)) for t, r in assign.items()}
    assert seen_as.get("C:D") == "UNIFIED" and seen_as.get("C:Q") == "B:Q", seen_as
    cf = counterfactual_score(trained_ab(seed)[0], seed, dirs)
    assert all(p == r for p, r in cf.values()), cf
    return (f"parts {parts} -> law 'door iff {law}'; C seen as {seen_as} ({w}); "
            f"5/5 counterfactuals {dict((k, int(p)) for k, (p, _r) in cf.items())}")


def test_synthesis_first_try(seed=0):
    """First episode in C: optimal, and the +5 lure (known from B) is refused
    because the unified law says it would lock the door."""
    out = []
    for seed in (seed,):
        agent, _pa, _pb, dirs, _eb, _log = trained_ab(seed)
        env, _pal = world_c(seed, dirs)
        steps, ok, hazards = run_episode(agent, env, TARGET_STEPS)
        opt = env.optimal_steps()
        assert ok and steps == opt and hazards == 0 and not eaten(env), \
            f"seed {seed}: {(steps, ok, hazards)} opt {opt} trace {env.trace}"
        out.append(steps)
    return f"first try in C: {out} (opt {opt}), lure never eaten"


def test_synthesis_ablation(seed=0):
    """Same world C, three other minds (the full one is tested above):
    no-synth = A+B but the two door roles are kept apart,
    A-only = lever world only, B-only = alarm world only (as its home world).
    Behaviour on the first try + the decisive counterfactuals
    (nothing pressed / lever only / lure then lever)."""
    res = {}
    for label in ("no-synth", "A-only", "B-only"):
        if label == "B-only":
            dirs = make_action_perm(seed)
            agent = ACOREEngine(n_actions=4, seed=seed)
            home = RawGridWorld(WORLD_B, make_palette(seed), dirs, seed=seed,
                                door_rule=rule_no_alarm, alarm_reward=CANDY, candy=True)
            for _ in range(8):
                run_episode(agent, home)
            agent = agent.compact()
        elif label == "A-only":
            agent, _pa, dirs = trained_a(seed)
        else:
            agent, _pa, _pb, dirs, _eb, _log = trained_ab(seed)
            agent.synthesis = False                   # the flag acts on entering C
        blob = pickle.dumps(agent)
        env, _pal = world_c(seed, dirs)
        steps, ok, _h = run_episode(pickle.loads(blob), env, TARGET_STEPS)
        cf = counterfactual_score(pickle.loads(blob), seed, dirs,
                                  ([], ["P"], ["Q", "P"]))
        wrong = [k for k, (p, r) in cf.items() if p != r]
        res[label] = (steps, int(ok), int(eaten(env)), wrong)
    assert not res["no-synth"][1] and not res["B-only"][1], res
    assert all(res[k][3] for k in res), res          # every partial mind is wrong somewhere
    return {k: f"{s} steps ok={o} lure={e} wrong={w}" for k, (s, o, e, w) in res.items()}


if __name__ == "__main__":
    gc.set_threshold(50_000, 20, 20)      # many short-lived tuples: collect less often
    t0 = time.perf_counter()
    tests = [
        test_body_discovery,
        test_one_shot_hazard_with_rotation_transfer,
        test_visible_carrying_state,
        test_same_step_link,
        test_hidden_state,
        test_delayed_link,
        test_delayed_experiment_design,
        test_latent_experiment_design,
        test_compound_and,
        test_compound_and_not,
        test_autonomous_compound_world,
        test_reversibility_classification,
        test_superstition_science,
        test_autonomous_full_world_and_transfer,
        test_analogical_transfer_first_try,
        test_analogy_ablation,
        test_ambiguous_analogy,
        test_structural_hypothesis_space,
        test_inverted_world_role_swap,
        test_alien_world_is_contained,
        test_structural_information_ablation,
        test_world_b_learns_its_own_door,
        test_role_synthesis,
        test_synthesis_first_try,
        test_synthesis_ablation,
    ]
    failed = 0
    for test in tests:
        try:
            print(f"[PASS] {test.__name__}  -> {test()}")
        except AssertionError as err:
            failed += 1
            print(f"[FAIL] {test.__name__}: {err}")
    elapsed = time.perf_counter() - t0
    print(f"\n{len(tests) - failed}/{len(tests)} passed in {elapsed * 1000:.1f} ms")
    raise SystemExit(1 if failed else 0)
