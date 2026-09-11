"""Generate every number the constraint-group / cascade slide prints.

    python build_protocol_slide.py

Reads the CSVs protocol.py, cascade_protocol.py and wind_usage.py already
wrote under ``participant-kit new/results/N-1 contingencies/protocol/`` and
emits ``generated/protocol_macros.tex``.  Nothing on the slide is typed by
hand - same discipline as ``build.py`` for the rest of the document.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SZYMON = os.path.dirname(HERE)
KIT = os.path.join(SZYMON, "participant-kit new")
PROT = os.path.join(KIT, "results", "N-1 contingencies", "protocol")
GEN = os.path.join(HERE, "generated")

sys.path.insert(0, SZYMON)
import protocol  # noqa: E402  the real THRESHOLD/TOL/MARGIN constants


def emit(name, body):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as f:
        f.write("% !TeX root = ../pipeline.tex\n")
        f.write(body.rstrip() + "\n")


def mac(name, value):
    return r"\newcommand{\%s}{%s}" % (name, value)


def fnum(x, dp=0):
    return f"{x:,.{dp}f}"


def main():
    S = pd.read_csv(os.path.join(PROT, "summary.csv")).iloc[0]
    C = pd.read_csv(os.path.join(PROT, "cascade.csv"))
    tr = pd.read_csv(os.path.join(PROT, "lookup", "trips.csv"))
    W = pd.read_csv(os.path.join(PROT, "wind_usage.csv"))

    lines = []
    A = lines.append

    # ---- the sweep, and the rule --------------------------------------
    A(mac("pN", fnum(S.events)))
    A(mac("pOut", fnum(S.outages)))
    A(mac("pTau", f"{protocol.THRESHOLD:.2f}"))
    A(mac("pTol", f"{(protocol.TOL - 1) * 100:.1f}"))

    # ---- today's method (shift-factor threshold) -----------------------
    A(mac("pEirCleared", f"{S.eir_cleared_pct:.1f}"))
    A(mac("pEirBrokeEvents", fnum(S.eir_events_broke_a_line)))
    A(mac("pEirBrokeLines", fnum(S.eir_lines_broken)))
    A(mac("pEirMedNodes", fnum(S.eir_median_nodes)))
    A(mac("pEirCommonMW", fnum(S.eir_total_MW_common)))

    # ---- net sensitivity, no veto (why the ranking alone is not enough) -
    A(mac("pOvlBrokeEvents", fnum(S.ovl_events_broke_a_line)))
    A(mac("pOvlBrokeLines", fnum(S.ovl_lines_broken)))

    # ---- net sensitivity + veto ----------------------------------------
    A(mac("pVetoCleared", f"{S.veto_cleared_pct:.1f}"))
    A(mac("pVetoBrokeEvents", fnum(S.veto_events_broke_a_line)))
    A(mac("pVetoBrokeLines", fnum(S.veto_lines_broken)))
    A(mac("pVetoMedNodes", fnum(S.veto_median_nodes)))
    A(mac("pVetoCommonMW", fnum(S.veto_total_MW_common)))

    saved_MW = S.eir_total_MW_common - S.veto_total_MW_common
    saved_pct = 100 * saved_MW / S.eir_total_MW_common
    A(mac("pSavedMW", fnum(saved_MW)))
    A(mac("pSavedPct", f"{saved_pct:.1f}"))

    # ---- LP optimum, for scale -----------------------------------------
    A(mac("pLpCleared", f"{S.lp_cleared_pct:.1f}"))
    A(mac("pLpFeasible", f"{S.lp_feasible_pct:.1f}"))

    # ---- precomputed lookup: N-1 constraint groups ---------------------
    A(mac("pTrips", fnum(len(tr))))
    A(mac("pTripsOutages", fnum(tr.outage.nunique())))
    A(mac("pTripsCircuits", fnum(tr.circuit.nunique())))
    A(mac("pMedGroupOverlap", fnum(tr.group_size.median())))
    A(mac("pMedGroupEir", fnum(tr.eirgrid_group_size.median())))
    vetoed = int(tr.vetoed_from_eirgrid.sum())
    eirmem = int(tr.eirgrid_group_size.sum())
    A(mac("pVetoedFromEir", fnum(vetoed)))
    A(mac("pVetoedFromEirPct", f"{100 * vetoed / eirmem:.1f}"))

    # ---- finding 1: curtailment is concentrated on a handful of nodes ---
    # Checked first, honestly: is any node cut in ZERO of the pN events?  No
    # - every node is asked at least once (min 6 events, of 20,203).  The real
    # finding is concentration, not absence, and it is reported as measured.
    A(mac("pWindN", fnum(len(W))))
    A(mac("pWindFleetMW", fnum(W.p_nom_MW.sum())))
    A(mac("pMinCutEvents", fnum(int(W.cut_events.min()))))

    rare = W[W.cut_events < 0.01 * S.events]
    A(mac("pRareN", fnum(len(rare))))
    A(mac("pRarePct", f"{100 * len(rare) / len(W):.1f}"))
    A(mac("pRareMW", fnum(rare.p_nom_MW.sum())))
    A(mac("pRareFleetPct", f"{100 * rare.p_nom_MW.sum() / W.p_nom_MW.sum():.1f}"))

    A(mac("pMedEventsPct", f"{100 * W.cut_events.median() / S.events:.2f}"))
    A(mac("pMedKeptPct", f"{100 * (1 - (W.cut_total_MW / W.avail_total_MW).median()):.2f}"))

    ranked = W.sort_values("cut_total_MW", ascending=False)
    cum = 100 * ranked.cut_total_MW.cumsum() / ranked.cut_total_MW.sum()
    A(mac("pTopFivePct", f"{cum.iloc[4]:.0f}"))
    A(mac("pTopTenPct", f"{cum.iloc[9]:.0f}"))
    A(mac("pTopTwentyPct", f"{cum.iloc[19]:.0f}"))

    A(mac("pTotalCutMW", fnum(W.cut_total_MW.sum())))
    A(mac("pTotalOfferedMW", fnum(W.avail_total_MW.sum())))
    A(mac("pOverallCutPct", f"{100 * W.cut_total_MW.sum() / W.avail_total_MW.sum():.2f}"))

    # ---- finding 2: stopping the cascade --------------------------------
    A(mac("pCascN", fnum(len(C))))
    for arm, tag in (("nothing", "Nothing"), ("eir", "CEir"), ("veto", "CVeto"),
                     ("lp", "CLp")):
        A(mac("p" + tag + "Lost", fnum(int(C[arm + "_lost"].sum()))))
        A(mac("p" + tag + "Spread", fnum(int((C[arm + "_lost"] > 0).sum()))))
        A(mac("p" + tag + "Isl", fnum(int(C[arm + "_islanded"].sum()))))

    emit("protocol_macros.tex", "\n".join(lines))
    print("wrote generated/protocol_macros.tex (%d macros)" % len(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
