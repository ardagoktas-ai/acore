"""
Micro-universe physics for Acore Studio.

Re-implemented exactly as in my Acore test suite: the world only emits a grid
of integers; palette and motor meaning are shuffled per seed.
"""

import random
from collections import deque

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SYMBOLS = ["#", ".", "A", "a", "K", "D", "L", "F", "C", "S", "G", "X", "T", "B", "P", "Q", "R"]
TIMER_DELAY = 3
CANDY = 5.0


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

DECOY_LAB = ["#######",
             "#.....#",
             "#BAT..#",
             "###G###",
             "#.....#",
             "#######"]

WORLD_FULL = ["##########F",
              "#A..L....C#",
              "#........L#",
              "#.L...B.T.#",
              "#####G#####",
              "#........D#",
              "###########"]

COMPOUND_WORLD = ["##########F",
                  "#P...L...Q#",
                  "#....A....#",
                  "#..L...L..#",
                  "#R.......D#",
                  "###########"]

WORLD_A = ["#####D#####",
           "#.........#",
           "#.L..P..L.#",
           "#....A....#",
           "#...L.L...#",
           "#....L....#",
           "###########"]

WORLD_B = ["###########",
           "#..Q...L..#",
           "#.L.......D",
           "#....A..L.#",
           "#.L.....Q.#",
           "#....L....#",
           "###########"]

WORLD_C = ["##D########",
           "#..L....Q.#",
           "#.L....L..#",
           "#...QA....#",
           "#.L.....P.#",
           "#....L....#",
           "###########"]


def rule_p_not_q(m):
    return "P" in m and "Q" not in m


def rule_lever(m):
    return "P" in m


def rule_no_alarm(m):
    return "Q" not in m


# name -> (layout, door_rule, extra RawGridWorld keyword arguments)
WORLDS = {
    "open_room": (OPEN_ROOM, None, {}),
    "key_corridor": (KEY_CORRIDOR, None, {}),
    "chest_corridor": (CHEST_CORRIDOR, None, {}),
    "switch_room": (SWITCH_ROOM, None, {}),
    "timer_room": (TIMER_ROOM, None, {}),
    "decoy_lab": (DECOY_LAB, None, {}),
    "compound": (COMPOUND_WORLD, rule_p_not_q, {}),
    "world_full": (WORLD_FULL, None, {}),
    "world_a": (WORLD_A, rule_lever, {}),
    "world_b": (WORLD_B, rule_no_alarm, {"alarm_reward": CANDY, "candy": True}),
    "world_c": (WORLD_C, rule_p_not_q, {"alarm_reward": CANDY, "candy": True}),
}


def list_worlds():
    return list(WORLDS.keys())


def build_world(name, seed=0):
    """Return (env, palette, action_dirs) for a world name and seed."""
    if name not in WORLDS:
        raise ValueError(f"unknown world '{name}'; known: {', '.join(WORLDS)}")
    layout, rule, extra = WORLDS[name]
    pal = make_palette(seed)
    dirs = make_action_perm(seed)
    env = RawGridWorld(layout, pal, dirs, seed=seed, door_rule=rule, **extra)
    return env, pal, dirs
