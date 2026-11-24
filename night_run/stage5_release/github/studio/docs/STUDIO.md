# Acore Studio - research prototype

## What it is

Acore Studio is a local web app in which a person watches my Acore engine live in a
micro-universe and asks it questions in natural language. It is not a general chatbot and it
is not AGI. Every answer is generated from the engine's own internal model:

- `latent_groups` / `latent_hypotheses()` - hidden-cause version spaces (e.g. the door rule),
- `link_groups()` - same-step and delayed causal links (e.g. what opens the gate, and the delay),
- `structure_report()` - surviving structural mappings to known roles,
- `reversibility(obs, action)` - learned reversibility of each action,
- `stats()`, `self_values`, `motor_map` - progress, body and motor knowledge.

A rule-based language layer (regex + keyword patterns with synonyms and paraphrases) detects
the intent and fills templates that verbalise the engine's formulas with the symbol letters of
the world (e.g. "the door (D) opens if P was pressed and Q was not (MAP formula P&~Q)"). There
is no LLM and no network access. When the engine has not learned something, or a question is
outside its world, the answer says so plainly and carries `source: "none"`.

## How to run it

The engine `acore_v5_0` is imported from the folder in the environment variable `ACORE_PATH`;
if it is not set, the fallback is `../../stage1_acore` relative to `studio/`.

```
cd studio
python server.py --port 8765
```

Then open `http://127.0.0.1:8765/`. The default port is 8765.

Python API:

```python
from acore_studio import Studio
s = Studio(world="compound", seed=0)
s.run_episodes(12)
print(s.ask("what opens the door?"))
print(s.snapshot()["laws"])
```

`new_world(world, seed)` keeps the same engine, so transfer between worlds can be watched.
`reset_agent()` starts a fresh engine.

### JSON routes

| Method | Route | Body | Returns |
|---|---|---|---|
| GET | /api/worlds | - | `{worlds, current}` |
| POST | /api/new | `{world, seed}` | snapshot + grid |
| POST | /api/step | - | `{action, reward, done, grid}` |
| POST | /api/run | `{episodes}` | `{results, snapshot}` |
| POST | /api/ask | `{text}` | `{answer, source, confidence}` |
| GET | /api/snapshot | - | `{world, tick, stats, laws, grid}` |
| POST | /api/reset | - | snapshot + grid (fresh engine) |

### Worlds

open_room, key_corridor, chest_corridor, switch_room, timer_room, decoy_lab, compound (door
opens iff P pressed and Q not pressed), world_full, world_a, world_b, world_c. The physics is the
one of my test suite; the palette and action permutation come from the seed.

## Question grammar

Intents are checked in this order; the first match wins (case-insensitive).

| Intent | Example phrasings | Source |
|---|---|---|
| help | "help", "what can I ask", "how does this work" | help |
| door | anything mentioning "door": "what opens the door", "how do I get through the door", "why does the door open" | latent_groups |
| gate | anything mentioning "gate" | link_groups |
| reversible | "is moving left reversible", "can I undo action 2", "is that permanent" | reversibility |
| unsure | "what are you unsure about", "open hypotheses", "what don't you know" | latent_hypotheses |
| body | "where am I", "what is my body", "which colour am I", "what do my actions do" | self_values |
| learned | "what have you learned", "laws", "stats", "summary", "what do you know" | stats |
| anything else | - | none |

Confidence: `high` when one hypothesis survives, `medium` when rivals survive but none would
change behaviour (or a listing), `low` when rivals still matter or an answer is unknown,
`none` when nothing is known.

## Design decisions

- `step()` advances one step of a running episode and starts a new episode when the previous one
  ended or reached 150 steps. `run_episodes()` first closes any half-finished `step()` episode
  with `end_episode()`, so the engine never sees two episodes interleaved.
- `new_world()` closes the current episode and keeps the engine, which allows transfer between
  worlds. `reset_agent()` creates a fresh `ACOREEngine(n_actions=4, seed=seed)` with the seed the
  Studio was constructed with.
- The server serialises all engine access with one lock. Episode runs are capped at 200 per
  request so a single click cannot block the UI indefinitely.
- An extra route, `POST /api/reset`, exposes `reset_agent()` to the UI. It is not part of the
  required route list.
- Intent order puts "door" and "gate" before the generic intents, so "why does the door open"
  is never read as a body or stats question.
- Door answers come from the MAP formula of the hidden-cause group whose target is the door
  value. Confidence is `high` when one formula survives, `medium` when the rivals are
  policy-irrelevant, and `low` otherwise.

## Honest limitations

- The language layer is pattern matching, not understanding. Unusual phrasing may fall into
  the wrong intent or into `none`.
- Answers only cover the engine's own model of its micro-universe; nothing outside it.
- Reversibility is reported only for the four actions from the current position.
- Engine values are mapped back to symbols with the palettes of the worlds visited; if worlds
  with different seeds are mixed, a value may be named after the first palette that used it.
  Synthesised (unified) roles have no symbol of their own and show as "?".
- Symbol verbs ("pressed" / "touched") are chosen by the template from the symbol letter, not
  learned by the engine.
- Performance of the Studio has not been measured: [TODO: measure episode and ask() latency].
