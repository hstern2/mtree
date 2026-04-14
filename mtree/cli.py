from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from rdkit import Chem
from rdkit import RDLogger
from mhfp.encoder import MHFPEncoder
import _tmap as tm
import typer


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

RDLogger.DisableLog("rdApp.*")


def _read_smi(path: Path) -> list[str]:
    smiles: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        mol = Chem.MolFromSmiles(token)
        if mol is None:
            continue
        smiles.append(Chem.MolToSmiles(mol))
    return smiles


@app.command(help="Minimum-spanning-tree plot of one or more .smi files.")
def main(
    smi_files: List[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="One or more .smi files; each becomes one color/legend entry.",
    ),
    output: Path = typer.Option(
        Path("mtree.png"), "-o", "--output", help="Output PNG path."
    ),
    k: int = typer.Option(20, "-k", "--k", help="kNN used by the TMAP layout."),
    node_size: float = typer.Option(6.0, "--node-size", help="Scatter marker size."),
    edge_width: float = typer.Option(0.4, "--edge-width", help="MST edge linewidth."),
    dpi: int = typer.Option(200, "--dpi", help="Output DPI."),
    dark: bool = typer.Option(False, "--dark/--light", help="Dark (faerun-style) background."),
    title: Optional[str] = typer.Option(None, "--title", help="Optional figure title."),
):
    all_smiles: list[str] = []
    labels: list[str] = []
    for f in smi_files:
        smis = _read_smi(f)
        if not smis:
            typer.echo(f"warning: no valid SMILES in {f}", err=True)
            continue
        all_smiles.extend(smis)
        labels.extend([f.stem] * len(smis))

    if len(all_smiles) < 2:
        typer.echo("error: need at least 2 valid SMILES across all inputs", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {len(all_smiles)} molecules from {len(smi_files)} file(s).")

    enc = MHFPEncoder(1024)
    fps = [tm.VectorUint(enc.encode(s)) for s in all_smiles]

    lf = tm.LSHForest(1024, 64)
    lf.batch_add(fps)
    lf.index()

    cfg = tm.LayoutConfiguration()
    cfg.k = k
    cfg.sl_scaling_type = tm.RelativeToAvgLength

    typer.echo("Computing TMAP layout...")
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    x = np.asarray(list(x))
    y = np.asarray(list(y))
    s = np.asarray(list(s), dtype=int)
    t = np.asarray(list(t), dtype=int)

    if dark:
        plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.03, 0.03, 0.80, 0.94])
    lax = fig.add_axes([0.85, 0.03, 0.13, 0.94])
    lax.axis("off")
    frame_color = "#cccccc" if not dark else "#555555"
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    segs = np.stack(
        [np.column_stack([x[s], y[s]]), np.column_stack([x[t], y[t]])],
        axis=1,
    )
    edge_color = (0.7, 0.7, 0.7) if dark else (0.6, 0.6, 0.6)
    ax.add_collection(
        LineCollection(
            segs, colors=[edge_color], linewidths=edge_width, alpha=0.6, zorder=1
        )
    )

    unique = sorted(dict.fromkeys(labels))
    n = len(unique)
    if n <= 10:
        palette = [colormaps["tab10"](i) for i in range(n)]
    elif n <= 20:
        palette = [colormaps["tab20"](i) for i in range(n)]
    else:
        hsv = colormaps["hsv"]
        palette = [hsv(i / n) for i in range(n)]
    colors = {lab: palette[i] for i, lab in enumerate(unique)}
    labels_arr = np.array(labels)
    for lab in unique:
        idx = np.where(labels_arr == lab)[0]
        ax.scatter(
            x[idx],
            y[idx],
            s=node_size ** 2,
            color=[colors[lab]],
            label=lab,
            edgecolors="none",
            zorder=2,
        )

    ax.set_aspect("auto")
    if title:
        ax.set_title(title)
    handles, labels_ = ax.get_legend_handles_labels()
    lax.legend(
        handles,
        labels_,
        loc="upper left",
        frameon=True,
        markerscale=1.5,
    )

    pad = 0.02 * max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    fig.savefig(output, dpi=dpi)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
