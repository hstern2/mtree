# mtree

Molecular minimum spanning tree.

CLI for minimum-spanning-tree plots of one or more `.smi` files, built on
[TMAP](https://tmap.gdb.tools/) (Probst & Reymond) for the layout and
matplotlib for rendering. Each input file is drawn in its own color; the
legend is labeled with the filename (without the `.smi` extension).

## Install

In-project:

```
uv sync
uv run mtree ...
```

Globally as a `uv` tool (must pin Python 3.10 — `tmap-viz` has no 3.11 wheel,
and `uv tool install` ignores the project's `requires-python`):

```
uv tool install -p 3.10 .
```

## Usage

Two subcommands: `build` computes the MST layout (the expensive step:
fingerprints, LSH forest, TMAP layout) and saves it to an `.npz`; `plot`
renders a PNG from that `.npz`. Splitting them lets you iterate on the
visualization without re-running the layout.

```
mtree build actives.smi decoys.smi library.smi -o tree.npz
mtree plot tree.npz -o tree.png
```

Each input file becomes one color/legend entry (labeled by filename stem).
Groups are plotted in the order given on the command line, so later-listed
files render on top — list larger sets first so smaller sets aren't
overdrawn.

`build` options: `-o/--output`, `-k/--k` (layout kNN).
`plot` options: `-o/--output`, `-d/--diameter`, `-a/--alpha`, `-m/--marker`,
`-c/--color`, `--edge-width`, `--dpi`, `--dark/--light`, `--title`.

Per-dataset overrides use `LABEL=VALUE` (e.g. `-c actives=red -m decoys=x`);
`ALL=VALUE` sets the default. Colors named via `-c` are excluded from the
auto-palette so other datasets won't be assigned the same color. Run
`mtree colors` to list named matplotlib colors with terminal swatches.

See `uv run mtree -h`, `uv run mtree build -h`, `uv run mtree plot -h`.
