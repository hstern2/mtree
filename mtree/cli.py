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


@app.command("build", help="Compute the MST layout from one or more .smi files and save it to an .npz.")
def build(
    smi_files: List[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="One or more .smi files; each becomes one color/legend entry.",
    ),
    output: Path = typer.Option(
        Path("mtree.npz"), "-o", "--output", help="Output .npz layout path."
    ),
    k: int = typer.Option(20, "-k", "--k", help="kNN used by the TMAP layout."),
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

    unique_labels = list(dict.fromkeys(labels))
    np.savez(
        output,
        x=x,
        y=y,
        s=s,
        t=t,
        labels=np.asarray(labels),
        unique_labels=np.asarray(unique_labels),
    )
    typer.echo(f"Wrote {output}")


@app.command("plot", help="Render an MST plot from a saved layout (.npz).")
def plot(
    layout: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Layout .npz produced by `mtree build`.",
    ),
    output: Path = typer.Option(
        Path("mtree.png"), "-o", "--output", help="Output PNG path."
    ),
    diameter: List[str] = typer.Option(
        [],
        "-d",
        "--diameter",
        help="Scatter marker diameter in typographic points (1 point = 1/72 inch). Pass a bare number or ALL=VALUE to set the default, or LABEL=VALUE to override one dataset. Repeatable. Default 5.",
    ),
    alpha: List[str] = typer.Option(
        [],
        "-a",
        "--alpha",
        help="Scatter marker opacity (0 = transparent, 1 = opaque). Same form as --diameter. Default 0.7.",
    ),
    marker: List[str] = typer.Option(
        [],
        "-m",
        "--marker",
        help="Marker symbol. Pass ALL=SYMBOL to set the default, or LABEL=SYMBOL to override one dataset. Repeatable. Default 'o'.",
    ),
    edge_width: float = typer.Option(0.4, "--edge-width", help="MST edge linewidth."),
    dpi: int = typer.Option(200, "--dpi", help="Output DPI."),
    dark: bool = typer.Option(False, "--dark/--light", help="Dark (faerun-style) background."),
    title: Optional[str] = typer.Option(None, "--title", help="Optional figure title."),
):
    def _parse(flag, specs, default, cast, allow_bare):
        d = default
        overrides: dict = {}
        for spec in specs:
            if "=" in spec:
                lab, val = spec.split("=", 1)
                lab = lab.strip()
                try:
                    parsed = cast(val.strip())
                except ValueError:
                    typer.echo(f"error: {flag} value for {lab!r} is invalid: {val!r}", err=True)
                    raise typer.Exit(code=1)
                if lab == "ALL":
                    d = parsed
                else:
                    overrides[lab] = parsed
            elif allow_bare:
                try:
                    d = cast(spec)
                except ValueError:
                    typer.echo(f"error: {flag} expects VALUE, ALL=VALUE, or LABEL=VALUE, got {spec!r}", err=True)
                    raise typer.Exit(code=1)
            else:
                typer.echo(f"error: {flag} expects ALL=VALUE or LABEL=VALUE, got {spec!r}", err=True)
                raise typer.Exit(code=1)
        return d, overrides

    default_diameter, diameters = _parse("--diameter", diameter, 5.0, float, allow_bare=True)
    default_alpha, alphas = _parse("--alpha", alpha, 0.7, float, allow_bare=True)
    default_marker, markers = _parse("--marker", marker, "o", str, allow_bare=False)

    data = np.load(layout, allow_pickle=False)
    x = data["x"]
    y = data["y"]
    s = data["s"]
    t = data["t"]
    labels_arr = np.asarray([str(v) for v in data["labels"]])
    unique = [str(v) for v in data["unique_labels"]]

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

    n = len(unique)
    if n <= 10:
        palette = [colormaps["tab10"](i) for i in range(n)]
    elif n <= 20:
        palette = [colormaps["tab20"](i) for i in range(n)]
    else:
        hsv = colormaps["hsv"]
        palette = [hsv(i / n) for i in range(n)]
    colors = {lab: palette[i] for i, lab in enumerate(unique)}
    for lab in unique:
        idx = np.where(labels_arr == lab)[0]
        dia = diameters.get(lab, default_diameter)
        ax.scatter(
            x[idx],
            y[idx],
            s=dia ** 2,
            color=[colors[lab]],
            label=lab,
            marker=markers.get(lab, default_marker),
            edgecolors="none",
            alpha=alphas.get(lab, default_alpha),
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
