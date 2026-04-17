# AGENTS.md — Instructions for Claude Code working on ScatterRad

This file is **the first thing you should read** in any session. It tells you how
to navigate the repo, what to read, what NOT to read, and how to stay
token-efficient.

> Recommended environment: activate the `scatterrad` conda env before building
> or running the repo, so PyTorch, pyradiomics, and kymatio are available.
>
> ```bash
> conda activate scatterrad
> ```

---

## Golden rules

1. **Do not read files you don't need.** Every unnecessary `view` or `grep`
   wastes tokens. Follow the "what to read" table below.
2. **Work on one module at a time.** Complete the current phase in
   `docs/IMPLEMENTATION_PLAN.md` before touching anything else.
3. **Don't re-derive decisions.** Architectural choices are locked in
   `docs/decisions/`. If you think one is wrong, flag it — don't silently
   override.
4. **Write tests alongside code.** Every new module lands with tests. The
   existing config/paths tests are the style reference.
5. **Keep modules self-contained.** Public API at the top of each file. No
   circular imports. One-way dependencies only (see ARCHITECTURE.md).
6. **Ask before expanding scope.** V1 is tightly bounded. Anything in
   `V2_ROADMAP.md` is off-limits unless the user explicitly asks.

---

## What to read, for what task

| Task                                   | Read these (in order)                                                       | Skip these |
| -------------------------------------- | --------------------------------------------------------------------------- | ---------- |
| Starting any new session               | `AGENTS.md`, `IMPLEMENTATION_PLAN.md`, `ARCHITECTURE.md`                    | everything else |
| Implementing a module                  | The module's spec in `docs/modules/`, plus `SCHEMAS.md` if JSON is involved | other module specs |
| Touching JSON I/O                      | `SCHEMAS.md`, the config parser (`src/scatterrad/config/`)                  | non-related modules |
| Modifying preprocessing                | `docs/modules/02_preprocessing.md`, `docs/decisions/DEC004_*`               | training/eval modules |
| Modifying scatter model                | `docs/modules/04_scatter_track.md`, `docs/decisions/DEC002_*`, `DEC003_*`   | radiomics module |
| Modifying radiomics model              | `docs/modules/03_radiomics_track.md`                                        | scatter module |
| Writing tests                          | The closest existing test file in `tests/` as style reference               | — |
| Debugging a failing test               | The failing test file and the single module under test                      | unrelated modules |

### What you almost never need to read in full

- `src/scatterrad/config/*.py` — once written, treat as stable; reference via
  module-level docstrings only.
- `src/scatterrad/paths.py` — stable; just `from scatterrad.paths import
  ScatterRadPaths`.
- Other modules' internal code — use their public API exposed in
  `__init__.py`.

---

## How to read efficiently

- **Targeted views.** Prefer `view path line_range=[a,b]` when you know the
  section. Reading a whole 500-line file to find one function is waste.
- **Grep before view.** If you don't know where a symbol lives, `bash grep -n`
  or `bash rg` first, then view the specific lines.
- **Trust the specs.** The module spec docs are written to be complete. If the
  spec says "public API is function X with signature Y", don't re-read the
  source to confirm.
- **Skip tests unless debugging them.** Tests are long. Read their names
  (via `grep "^def test_"`) before reading bodies.

---

## When implementing a new module

Follow this procedure, in order:

1. Read `docs/modules/<module>.md` in full. It's self-contained.
2. Read `SCHEMAS.md` only if the module touches JSON I/O.
3. Read the public API of any module listed under "Depends on" (via
   `__init__.py` only, not full source).
4. Write the module stub with types and docstrings. No bodies yet.
5. Write the tests.
6. Fill in bodies until tests pass.
7. Update `IMPLEMENTATION_PLAN.md` to mark the phase done.
8. Commit. Each phase = one commit.

---

## Style constraints

- Python 3.10+, full type hints, `from __future__ import annotations` at top.
- No bare `Exception` catches. Each module defines its own error class.
- No `print` in library code — use `logging.getLogger(__name__)`.
- Dataclasses over dicts for structured config (frozen=True where possible).
- Docstrings: one-line summary, blank line, details. Google or NumPy style.
- Tests use pytest, fixtures over setup/teardown, parametrize where useful.
- Line length: 100.
- Formatters: `ruff format`, `ruff check --fix`.

---

## Token-saving tactics

- When modifying a file you've already seen this session: use `str_replace`
  without re-viewing unless content might have changed.
- Never dump whole files into your thinking. Summarize what you need.
- Prefer small, incremental edits over rewrites.
- If a task feels like it needs 10+ file reads, stop and ask the user whether
  to narrow scope.

---

## If you get stuck

Do not flail. Stop and produce a short note:

- What you're trying to do (1 sentence)
- What you've tried (bullet list)
- What specifically is unclear (1 question)

The user would rather answer one clarifying question than debug a sprawling
half-finished PR.
