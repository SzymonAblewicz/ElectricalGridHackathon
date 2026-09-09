# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""The k smallest Laplacian eigenvalues of one PyPSA case, and their eigenvectors.

This is the first half of spectral clustering: the eigenvectors belonging to the
smallest eigenvalues of a graph Laplacian embed each bus in R^k, and buses that
sit close together there are hard to separate without cutting a lot of capacity.
`cluster.py` reads what this writes and runs k-means on that embedding.

The graph comes from `get_weighted_graph.py` rather than being rebuilt here, so
there is one definition of a corridor and one definition of its weight. The
weighting is always the capacity one, never `reciprocal()`: a Laplacian
eigenvector reads its weights as *affinity*, where a big number means "these two
belong together". That is the opposite of what `betweenness.py` needs, where the
weight has to read as a distance - hence the reciprocal there and not here.
Feeding 1/s_nom to this script would cluster the grid inside out, treating the
3,846 MVA backbone as the weakest link in the network, so there is no switch for
it.

Which Laplacian matters more than it looks. Weighted degree on these cases spans
33 to 7,910 MVA across buses, and 0.02 MW to 7,910 MVA once generators are nodes
too. The combinatorial L = D - A takes that skew literally: with `EMBEDDING =
"unnorm"` and generators on, its six smallest eigenvectors each isolate one
near-dangling generator and k-means returns clusters of size 1456, 1, 1, 1, 1, 1.
The normalised L_sym divides that skew out and returns 377, 285, 249, 223, 175,
152 on the same graph. So "sym" is the default, and "unnorm" is kept mostly
because watching it fail is a good way to see what normalisation is for.

    sym     L_sym, then row-normalised in cluster.py    (Ng-Jordan-Weiss)
    rw      L_sym, then scaled by D^-1/2 in cluster.py  (Shi-Malik)
    unnorm  L = D - A, used as-is                       (degenerates here)

"sym" and "rw" share this solve: the random-walk eigenvectors are exactly
D^-1/2 U_sym, so what separates the two is a rescaling applied at clustering
time, not a second decomposition.

On the solver: `eigsh(M, k, which="SM")` is the obvious call and the wrong one.
Lanczos converges from the ends of the spectrum inwards, and "smallest
magnitude" asks it for the interior, so it grinds. Shift-invert instead - factor
(M - sigma*I) once and iterate on its inverse, whose largest eigenvalues are the
ones nearest sigma. On a 751-bus case at k=8 that is 0.006s against 0.022s for
"SM" and 0.026s for "SA", and the margin widens with size. SIGMA sits just below
zero rather than at it because a connected graph puts exactly one eigenvalue at
0, and factorising an exactly singular matrix is not a thing sparse LU will do.

Set the constants below and run the file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)

import get_weighted_graph as gwg  # noqa: E402


# ---- what to run ---- #
#
# Same cases as get_weighted_graph.py: directory names under
# grid_TF_Wind/data/pypsa, of the shape
# TYTFS2024_{WP,SV}{2024,2033}_V35_{transmission,full}.
#
# K is the number of clusters wanted. It does not have to be right first time -
# the run reports the eigengap, which is the spectrum's own opinion on how many
# clusters the graph has, so a first pass at any K tells you what to set it to.

CASE = "TYTFS2024_WP2024_V35_transmission"
K = 6                # clusters wanted
GENERATORS = True    # include the generators as nodes of their own
EMBEDDING = "sym"    # "sym" | "rw" | "unnorm"


# Eigenpairs computed beyond K, so the gap *after* the K-th one is visible and
# the eigengap heuristic has something to say about whether K was a good choice.
EXTRA = 5

# Just below zero: near enough that shift-invert converges on the bottom of the
# spectrum, far enough that the factorisation is not asked to invert the exact
# zero eigenvalue every connected graph has.
SIGMA = -1e-5

EMBEDDINGS = ("sym", "rw", "unnorm")


# ---- the algorithm ---- #


def spectrum(M: sparse.csr_array, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k algebraically smallest eigenpairs of a sparse symmetric matrix.

    Shift-invert, for the reason in the module docstring. CSC because that is
    what the sparse LU behind it wants; handing it CSR only makes scipy convert.

    Eigenvalues come back ordered by distance from SIGMA, which is nearly but
    not exactly ascending, so they are sorted here. Column signs are arbitrary
    and stay that way: flipping one reflects the whole point cloud, and k-means
    is invariant under that.
    """
    values, vectors = eigsh(M.tocsc(), k=k, sigma=SIGMA, which="LM")
    order = np.argsort(values)
    return values[order], vectors[:, order]


def eigengap(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Consecutive gaps, and the cluster count the largest of them suggests.

    A gap between the i-th and (i+1)-th eigenvalue argues for i clusters: the
    first i eigenvectors are cheap to separate along and the next one is not.

    The gap after the first eigenvalue is skipped when picking the largest. On a
    connected graph lambda_1 is 0 and lambda_2 is the algebraic connectivity, so
    that gap says the graph is in one piece - which would win the argmax on most
    graphs while only ever suggesting k=1.
    """
    gaps = np.diff(values)
    suggested = int(np.argmax(gaps[1:])) + 2 if len(gaps) > 1 else 1
    return gaps, suggested


# ---- output ---- #


def plot(values: np.ndarray, suggested: int, path: Path, title: str) -> None:
    """The computed eigenvalues, with the gap the heuristic picked marked."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    index = np.arange(1, len(values) + 1)

    ax.plot(index, values, "-", color="#b0b7c3", linewidth=1, zorder=1)
    ax.scatter(index, values, s=34, c="#2c6fbb", zorder=3, linewidths=0)
    ax.axvspan(
        suggested, suggested + 1, color="#d1495b", alpha=0.13, zorder=0,
        label=f"largest gap after {suggested} — suggests k = {suggested}",
    )
    ax.axvline(K, color="#3f8f4f", linestyle="--", linewidth=1.2, zorder=2,
               label=f"K = {K} (configured)")

    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- entry point ---- #


def main() -> None:
    if EMBEDDING not in EMBEDDINGS:
        raise ValueError(f"EMBEDDING must be one of {EMBEDDINGS}, not {EMBEDDING!r}")

    case_dir = gwg.PYPSA_DIR / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available: {', '.join(sorted(p.name for p in gwg.PYPSA_DIR.iterdir() if p.is_dir()))}")

    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir) if GENERATORS else None
    A, node_index = gwg.adjacency(buses, branches, generators)
    D, L, L_sym = gwg.matrices(A)

    # "sym" and "rw" differ only in a rescaling cluster.py applies afterwards,
    # so both decompose the same matrix here.
    M = L if EMBEDDING == "unnorm" else L_sym

    n = A.shape[0]
    want = K + EXTRA
    # eigsh can return at most n-1 eigenpairs, and asking for a large share of
    # them is slower than a dense solve would be - but these cases run to a few
    # thousand nodes and K is a handful, so this only trips on a toy case.
    if want >= n:
        raise ValueError(
            f"K + EXTRA = {want} eigenpairs wanted from a {n}-node graph; "
            f"lower K to at most {n - EXTRA - 1}")

    values, vectors = spectrum(M, want)
    gaps, suggested = eigengap(values)
    components = csgraph.connected_components(A, directed=False)[0]

    # Both halves of the spectral pipeline write to one stage folder: the
    # eigenvectors are an intermediate that only cluster.py consumes, so they
    # belong beside the clusters they produce rather than in a stage of their own.
    out = gwg.out_dir(CASE, GENERATORS, "clustering")
    tag = "_gen" if GENERATORS else ""
    stem = f"spectrum_{EMBEDDING}{tag}"
    path = out / f"{stem}.npz"
    # The embedding tag and the degrees travel with the vectors, so cluster.py
    # applies the transform this decomposition was actually made for rather than
    # whichever one its own constant happens to name.
    np.savez(
        path,
        eigenvalues=values,
        eigenvectors=vectors,
        degrees=D.diagonal(),
        embedding=np.array(EMBEDDING),
        case=np.array(CASE),
        generators=np.array(GENERATORS),
    )
    plot(values, suggested, out / f"{stem}.pdf",
         f"{CASE}{' +generators' if GENERATORS else ''} — {EMBEDDING}")

    listing = "\n".join(
        f"    {i + 1:>2}. {v:>13.6g}"
        + (f"   gap {gaps[i]:>11.6g}" if i < len(gaps) else "")
        + ("  <- largest" if i + 1 == suggested else "")
        for i, v in enumerate(values)
    )
    # A zero eigenvalue per component is structural, not numerical: it says the
    # graph is in that many pieces. Worth printing beside the spectrum, because
    # if it is not 1 then the leading eigenvectors are component indicators and
    # clustering recovers the components before it says anything about the grid
    # inside them.
    matrix = "L (combinatorial)" if EMBEDDING == "unnorm" else "L_sym (normalised)"
    print(
        f"{CASE}{' +generators' if GENERATORS else ''}\n"
        f"  nodes      {n} in {components} component"
        f"{'s' if components != 1 else ''}\n"
        f"  edges      {A.nnz // 2}\n"
        f"  matrix     {matrix} for embedding {EMBEDDING!r}\n"
        f"  solved     {want} smallest eigenpairs, shift-invert at sigma={SIGMA:g}\n"
        f"  spectrum   K = {K} configured, eigengap suggests k = {suggested}\n"
        f"{listing}\n"
        f"             (the gap after the 1st is structural on a connected\n"
        f"              graph, so it is excluded from the suggestion)\n"
        f"  wrote      {path}"
    )


if __name__ == "__main__":
    main()
