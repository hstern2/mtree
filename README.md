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
uv tool install --python 3.10 .
```

## Usage

```
mtree actives.smi decoys.smi library.smi -o tree.png
```

Options: `-o/--output`, `-k/--k` (layout kNN), `--node-size`, `--edge-width`,
`--dpi`, `--dark/--light`, `--title`. See `uv run mtree --help`.
