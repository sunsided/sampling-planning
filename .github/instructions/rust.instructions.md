---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`; property tests in `tests/it/properties.rs`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Planner loops are short drivers that call
  documented step functions. If a block needs a "what" comment, extract it into a function
  named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe`: the crate root has `#![forbid(unsafe_code)]`.
- The crate is `#![no_std]` with `alloc`. Flag any `std` use outside tests, benches and
  examples. Float math goes through `libm`, so results stay identical across platforms.
- Randomness comes only from a caller-provided `rand_core::Rng`. A seeded run must be fully
  reproducible: no hidden RNGs, no iteration over hash maps.
- `KdTree` must return exactly what `LinearScan` returns, ties broken toward the earliest
  inserted node. Planners must give identical runs with either structure; tests compare them.
- Validity is three-valued; planners accept only `Validity::Valid` and treat `Unknown` like
  `Invalid`.
- Public API changes go with a matching update to the README "Features" list and
  `docs/design.md`.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
