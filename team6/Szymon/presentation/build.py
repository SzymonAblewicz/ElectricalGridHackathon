"""Generate every number and figure the pipeline document prints.

    python build.py                 main document
    python build.py --full-dump     also emit the COMPLETE matrices, blocked

All of it is the real WP2033_all-island run: 754 buses, 980 branches.  Nothing
is a toy, nothing is a stand-in, and no value is typed by hand anywhere in the
.tex - every entry and every check comes out of here.

The pipeline itself is imported from ``ptdf_all``, not reimplemented, so these
matrices are the same objects the results/ CSVs were built from.

One mechanical limit shapes the layout.  TeX cannot typeset a 980-column
array - it runs out of columns long before it runs out of paper.  So:

  * the main body shows real, labelled windows of every matrix, with the
    surrounding structure given as a heatmap of the whole thing;
  * ``--full-dump`` writes every entry of every matrix, split into column
    blocks narrow enough for TeX to set.  That appendix runs to hundreds of
    pages, which is why it is off by default.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SZYMON = os.path.dirname(HERE)
KIT = os.path.join(SZYMON, "participant-kit new")
RESULTS = os.path.join(KIT, "results")
GEN = os.path.join(HERE, "generated")
FIGS = os.path.join(HERE, "figures")

sys.path.insert(0, SZYMON)
sys.path.insert(0, KIT)

import plotstyle                                    # noqa: E402
from ptdf_all import build, injections              # noqa: E402  the real pipeline

DPI = 200
BLOCK = 12          # columns per block in the full dump - TeX-safe
WIN = 10            # rows/cols in a main-body window


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #

def esc(s):
    s = str(s)
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def num(v, dp=4):
    if v == 0:
        return "0"
    if abs(v) < 10.0 ** -(dp + 1):
        return "0"
    return f"{v:.{dp}f}"


def array_tex(M, rows, cols, dp=4, corner="", scale=None, rot=90):
    """A labelled matrix as a *tabular*, not an array.

    Array cells are math mode, and \\rotatebox inside them is fragile - it
    breaks on the station names.  A tabular keeps the labels in text mode and
    the numbers in \\( \\) where they belong, which typesets reliably at any
    width.  ``scale`` divides every entry; the caller states it in the caption.
    """
    A = np.asarray(M, dtype=float)
    if scale:
        A = A / scale
    # No \rotatebox: it breaks scanning inside tabular cells here, and the page
    # is landscape, so horizontal headers fit.
    head = " & ".join(r"{\tiny %s}" % esc(str(c)[:13]) for c in cols)
    body = "\\\\\n".join(
        r"{\tiny %s} & " % esc(str(r)[:16]) + " & ".join(r"$%s$" % num(v, dp)
                                                        for v in row)
        for r, row in zip(rows, A))
    return (r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}"
            + "\n" + r"\begin{tabular}{l|" + "r" * A.shape[1] + "}" + "\n"
            + r"{\tiny %s} & " % esc(corner) + head + r"\\ \hline" + "\n"
            + body + "\n" + r"\end{tabular}")


def emit(name, body):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as f:
        # These are \input fragments with no preamble of their own.  The magic
        # comment points an editor at the parent, so opening one and hitting
        # build does not try to typeset it alone and fail on the missing
        # \begin{document}.
        f.write("% !TeX root = ../pipeline.tex\n")
        f.write(body.rstrip() + "\n")


def mac(name, value):
    return r"\newcommand{\%s}{%s}" % (name, value)


def heat(M, name, title, sub, xlabel, ylabel, diverging=True, barlabel=""):
    plotstyle.use()
    fig, ax = plt.subplots(figsize=(13.4, 5.4))
    fig.subplots_adjust(top=0.74)
    A = np.asarray(M, dtype=float)
    if diverging:
        lim = float(np.percentile(np.abs(A), 99)) or 1e-9
        im = ax.imshow(A, cmap=plotstyle.diverging_cmap.reversed(),
                       aspect="auto", vmin=-lim, vmax=lim, interpolation="nearest")
    else:
        im = ax.imshow(np.abs(A), cmap=plotstyle.sequential_cmap, aspect="auto",
                       vmin=0, vmax=float(np.percentile(np.abs(A), 99)) or 1e-9,
                       interpolation="nearest")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    bar.set_label(barlabel, fontsize=9, color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    fig.suptitle(title, fontsize=13, x=0.02, ha="left", y=0.97,
                 fontweight="semibold")
    fig.text(0.02, 0.90, sub, fontsize=8.6, color=plotstyle.INK_SOFT,
             ha="left", va="top", linespacing=1.5)
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", dpi=DPI, facecolor=plotstyle.SURFACE)
    plt.close(fig)
    print("  ->", os.path.relpath(path, HERE))


# --------------------------------------------------------------------------- #

def full_dump(M, rows, cols, stem, dp=4, scale=None, label=""):
    """Every entry, split into TeX-safe column blocks."""
    A = np.asarray(M, dtype=float)
    if scale:
        A = A / scale
    n = A.shape[1]
    parts = []
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        parts.append(
            r"\subsection*{%s \quad columns %d--%d of %d}" % (label, s + 1, e, n)
            + "\n" + r"\[" + array_tex(A[:, s:e], rows, cols[s:e], dp, rot=90)
            + r"\]" + "\n")
    emit(f"dump_{stem}.tex", "\n".join(parts))
    pages = len(parts) * max(1, len(rows) // 45)
    print(f"  dump_{stem}.tex  {A.shape[0]}x{n}  "
          f"{len(parts)} blocks  ~{pages} pages")
    return pages


def main(argv):
    want_dump = "--full-dump" in argv
    os.makedirs(GEN, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    import pypsa
    import gridkit
    gridkit.quiet()
    n = pypsa.Network(os.path.join(KIT, "networks", "WP2033_all-island.nc"))
    br, K, b, phi, PTDF, rank = build(n)
    buses, edges = list(n.buses.index), list(br.index)
    nb, ne = len(buses), len(edges)
    L = (K * b) @ K.T
    Lp = np.linalg.pinv(L, hermitian=True)

    # ---- Stage 4, exactly as the real run does it -----------------------
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)
    n.lpf(n.snapshots)
    load = gridkit.line_loading(n)
    snap = load.max(axis=1).idxmax()
    p = injections(n, [snap]).reindex(buses).fillna(0.0).iloc[:, 0].to_numpy()
    p_eff = p + K @ (b * phi)
    f = PTDF @ p_eff - b * phi
    sgn = np.sign(f); sgn[sgn == 0] = 1.0
    sens = PTDF * sgn[:, None]

    truth = pd.concat([n.lines_t.p0.loc[[snap]],
                       n.transformers_t.p0.loc[[snap]]], axis=1)[edges]
    resid = float(np.abs(f - truth.to_numpy()[0]).max())

    station = n.buses["psse_name"].astype(str)
    wind = n.generators[n.generators["carrier"] == "wind"]
    wind_buses = sorted(set(wind["bus"]))
    cong = br[br["hours_at_rating"] > 0].sort_values(
        "hours_at_rating", ascending=False) if "hours_at_rating" in br.columns \
        else br.iloc[:0]
    if not len(cong):
        hrs = (load >= 0.999).sum()
        cong = br.loc[[e for e in edges if hrs.get(e, 0) > 0]]

    # ---- windows: real buses, real branches, real values ----------------
    # Anchored on the worst circuit so the window is the part of the grid the
    # rest of this document is actually about.
    worst = cong.index[0] if len(cong) else edges[0]
    seed = [br.at[worst, "bus0"], br.at[worst, "bus1"]]
    nbr = [u for u in buses
           if any(K[buses.index(u), edges.index(e)] != 0
                  for e in edges if e in cong.index)]
    win_b = list(dict.fromkeys(seed + nbr))[:WIN]
    bi = [buses.index(u) for u in win_b]
    win_e = list(cong.index[:WIN]) if len(cong) else edges[:WIN]
    ei = [edges.index(e) for e in win_e]
    lab_b = [f"{station.get(u, u)[:12]}" for u in win_b]
    lab_e = [e for e in win_e]
    win_w = wind_buses[:WIN]
    wi = [buses.index(u) for u in win_w]
    lab_w = [f"{station.get(u, u)[:12]}" for u in win_w]

    emit("w_K.tex", array_tex(K[np.ix_(bi, ei)], lab_b, lab_e, 0, "bus / branch"))
    emit("w_L.tex", array_tex(L[np.ix_(bi, bi)], lab_b, lab_b, 2, "", scale=1e3))
    emit("w_Lp.tex", array_tex(Lp[np.ix_(bi, bi)], lab_b, lab_b, 3, "", scale=1e-4))
    emit("w_PTDF.tex", array_tex(PTDF[np.ix_(ei, wi)], lab_e, lab_w, 4,
                                 "branch / wind bus"))

    # A window mixing wind and non-wind buses, so the selection about to be
    # made is visible as a choice rather than appearing from nowhere.  Wind
    # columns carry a star.
    windset = set(wind_buses)
    mix_w = [u for u in buses if u in windset][:5]
    mix_o = [u for u in buses if u not in windset][:5]
    mix = [u for pair in zip(mix_w, mix_o) for u in pair]
    mi = [buses.index(u) for u in mix]
    lab_m = [f"{station.get(u, u)[:11]}{'*' if u in windset else ''}" for u in mix]
    emit("w_PTDF_mixed.tex", array_tex(PTDF[np.ix_(ei, mi)], lab_e, lab_m, 4,
                                       "branch / bus"))
    emit("w_sens.tex", array_tex(sens[np.ix_(ei, wi)], lab_e, lab_w, 4,
                                 "branch / wind bus"))
    emit("w_B.tex", r"\operatorname{diag}\big(" +
         ",\\; ".join(f"{b[j]:.1f}" for j in ei) + r"\big)")
    emit("w_f.tex", array_tex(np.c_[f[ei], sgn[ei], load.max()[win_e].to_numpy() * 100],
                              lab_e, ["f (MW)", "sign", "peak load %"], 2,
                              "branch"))

    # ---- heatmaps of the whole thing ------------------------------------
    heat(K, "m_K.pdf", f"$K$ — the full {nb} $\\times$ {ne} incidence matrix",
         f"Exactly two non-zeros per column: $+1$ at bus0, $-1$ at bus1. "
         f"{int((K != 0).sum()):,} non-zeros, {100 * (K != 0).mean():.2f}\\% dense.",
         "branch $e$", "bus $i$", True, "$+1$ / $-1$")
    heat(L, "m_L.pdf", f"$L = K B K^{{T}}$ — the full {nb} $\\times$ {nb} Laplacian",
         f"{int((np.abs(L) > 1e-9).sum()):,} non-zeros "
         f"({100 * (np.abs(L) > 1e-9).mean():.2f}% dense). The structure is the "
         "grid's own topology.", "bus $j$", "bus $i$", True, "MW/rad")
    heat(Lp, "m_Lp.pdf", f"$L^{{+}}$ — the full {nb} $\\times$ {nb} pseudoinverse",
         f"{100 * (np.abs(Lp) > 1e-12).mean():.1f}\\% dense. Inversion destroys "
         "sparsity: every bus is now coupled to every other.",
         "bus $j$", "bus $i$", False, "rad/MW")
    heat(PTDF, "m_PTDF.pdf",
         f"PTDF $= B K^{{T}} L^{{+}}$ — all {ne} $\\times$ {nb} = {ne * nb:,} entries",
         "Every branch against every bus. No numeric form of this is readable, "
         "so it is shown whole as an image.", "bus $n$", "branch $e$", True,
         "MW per MW")
    heat(PTDF[:, [buses.index(u) for u in wind_buses]], "m_PTDFwind.pdf",
         f"PTDF$\\,\\cdot\\,S$ — the same matrix, {len(wind_buses)} wind columns kept",
         f"A column selection, nothing more: {ne} $\\times$ "
         f"{len(wind_buses)}. The other "
         f"{nb - len(wind_buses)} buses were needed to compute these values "
         "and are not curtailable.",
         "wind node $n$", "branch $e$", True, "MW per MW")
    heat(sens[:, [buses.index(u) for u in wind_buses]],
         "m_sens.pdf",
         f"Sensitivity — {ne} $\\times$ {len(wind_buses)}, signed along real flow",
         "PTDF re-pointed by $\\operatorname{sign}(f_e)$. Positive means "
         "curtailing there unloads the branch.",
         "wind node $n$", "branch $e$", True, "MW per MW")

    # ---- macros ---------------------------------------------------------
    macros = [
        mac("gnb", f"{nb:,}"), mac("gne", f"{ne:,}"),
        mac("gnlines", f"{int((br['kind'] == 'Line').sum()):,}"),
        mac("gntx", f"{int((br['kind'] == 'Transformer').sum()):,}"),
        mac("gnwind", f"{len(wind_buses)}"),
        mac("gnwindgen", f"{len(wind)}"),
        mac("gnPtdfEntries", f"{ne * nb:,}"), mac("gnLpEntries", f"{nb * nb:,}"),
        mac("grank", f"{rank}"), mac("gnullity", f"{nb - rank}"),
        mac("growsum", f"{np.abs(PTDF.sum(axis=1)).max():.1e}"),
        mac("gLone", f"{np.abs(L @ np.ones(nb)).max():.1e}"),
        mac("gpsum", f"{abs(p.sum()):.1e}"), mac("gpeffsum", f"{abs(p_eff.sum()):.1e}"),
        mac("gresid", f"{resid:.1e}"),
        mac("gKdense", f"{100 * (K != 0).mean():.2f}"),
        mac("gLdense", f"{100 * (np.abs(L) > 1e-9).mean():.2f}"),
        mac("gLpdense", f"{100 * (np.abs(Lp) > 1e-12).mean():.1f}"),
        mac("gnshift", f"{int((np.abs(phi) > 1e-9).sum())}"),
        mac("gshiftdeg", f"{np.degrees(np.abs(phi)).max():.0f}"),
        mac("gsnap", esc(str(snap))),
        mac("gworst", esc(str(worst))),
        mac("gncong", f"{len(cong)}"),
        mac("gncongtx", f"{int((cong['kind'] == 'Transformer').sum())}"),
        mac("gbmin", f"{b.min():.1f}"), mac("gbmax", f"{b.max():.1f}"),
        mac("gptdfmin", f"{PTDF.min():.4f}"), mac("gptdfmax", f"{PTDF.max():.4f}"),
        mac("gwinb", f"{len(win_b)}"), mac("gwine", f"{len(win_e)}"),
        mac("gnnonwind", f"{nb - len(wind_buses):,}"),
        mac("gnwindentries", f"{ne * len(wind_buses):,}"),
        mac("gcommute", f"{np.abs((PTDF * sgn[:, None])[:, [buses.index(u) for u in wind_buses]] - (PTDF[:, [buses.index(u) for u in wind_buses]] * sgn[:, None])).max():.1e}"),
    ]
    emit("macros.tex", "\n".join(macros))

    print(f"network  {nb} buses, {ne} branches "
          f"({int((br['kind'] == 'Line').sum())} lines + "
          f"{int((br['kind'] == 'Transformer').sum())} transformers)")
    print(f"rank(L)  {rank}/{nb} -> nullity {nb - rank}")
    print(f"checks   PTDF row sums {np.abs(PTDF.sum(axis=1)).max():.1e} | "
          f"1'p {abs(p.sum()):.1e} | 1'p_eff {abs(p_eff.sum()):.1e}")
    print(f"         flows vs n.lpf() at {snap}: {resid:.1e} MW")
    print(f"density  K {100 * (K != 0).mean():.2f}%  "
          f"L {100 * (np.abs(L) > 1e-9).mean():.2f}%  "
          f"L+ {100 * (np.abs(Lp) > 1e-12).mean():.1f}%")

    if want_dump:
        print("\nfull dump (every entry, blocked for TeX):")
        tot = 0
        tot += full_dump(K, [station.get(u, u)[:12] for u in buses], edges,
                         "K", 0, label="$K$")
        tot += full_dump(L, [station.get(u, u)[:12] for u in buses],
                         [station.get(u, u)[:12] for u in buses], "L", 2,
                         scale=1e3, label="$L$ ($\\times 10^{3}$)")
        tot += full_dump(Lp, [station.get(u, u)[:12] for u in buses],
                         [station.get(u, u)[:12] for u in buses], "Lp", 3,
                         scale=1e-4, label="$L^{+}$ ($\\times 10^{-4}$)")
        tot += full_dump(PTDF, edges,
                         [station.get(u, u)[:12] for u in buses], "PTDF", 4,
                         label="PTDF")
        tot += full_dump(sens[:, [buses.index(u) for u in wind_buses]], edges,
                         [station.get(u, u)[:12] for u in wind_buses], "sens", 4,
                         label="Sensitivity")
        emit("dump_flag.tex", r"\newcommand{\fulldump}{}")
        print(f"  TOTAL ~{tot} pages")
    else:
        emit("dump_flag.tex", "% full dump not generated; run build.py --full-dump")
        for s in ("K", "L", "Lp", "PTDF", "sens"):
            pth = os.path.join(GEN, f"dump_{s}.tex")
            if not os.path.exists(pth):
                emit(f"dump_{s}.tex", "% not generated")

    print("\nNow: latexmk -pdf pipeline.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
