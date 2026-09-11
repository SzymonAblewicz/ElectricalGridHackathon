"""Corrected greedy dispatch-down arms.

protocol.arm_greedy has two defects, and both of them charge the net-ranked
(overlap) arm for something the per-circuit arm never pays:

1.  THE STEP WAS SIZED ON ONE CIRCUIT.

        r0   = rel[e0, i]
        want = (|cur[e0]| - lim[e0]) / r0

    e0 is the single worst circuit.  The per-circuit arm picks
    i = argmax rel[e0], so r0 is the largest entry there is and want is the
    right size.  The net-ranked arm picks i for its *breadth*, so rel[e0, i]
    can be near zero - want blows up, min(room, want) collapses onto room, and
    the node is dispatched to zero to fix a circuit it barely touches.  The
    ranking was not being tested; the mismatch between the ranking and the step
    was.

    Corrected: step to the first circuit that reaches its rating,

        want = min over e still over of  need_e / rel[e, i]   (rel > 0)

    At that point the violated set changes, so the score changes, so the node
    should be re-chosen.  That is what a greedy is.  It is the same rule for
    both arms, and for the per-circuit arm it very nearly reduces to the old
    one, so the control is barely moved.

2.  A VETOED NODE WAS DELETED FOR GOOD.

        if room <= EPS:
            left[i] = 0.0

    room is the headroom cap *at the current flows*.  Cutting somewhere else
    moves those flows.  A node blocked at iteration 3 is often the right node
    at iteration 9, and this threw it away.  Only the veto arms pay it, and it
    is why the greedy failed to clear events the LP proves are clearable.

    Corrected: the block is for this pass only and is lifted the moment any
    cut lands, because that is the moment the headroom it was measured against
    stopped being true.

RANKING RULES
-------------
    worst   rel on the worst violated circuit alone.  This is a per-circuit
            constraint group, dispatched in merit order.  The control.
    sum     sum of relief over every violated circuit.  The original overlap
            score.  Its scale grows with the number of violated circuits, so
            the 0.05 membership threshold does not mean the same thing for it
            as for `worst` - which is a second reason it behaved oddly.
    need    relief averaged over the violated circuits, weighted by how much
            relief each still needs.  Same units as `worst`, so the threshold
            is comparable, and it answers the question actually being asked:
            per MW cut here, how much of the *outstanding* overload goes away.
    minmax  maximise the worst-case fractional progress, min_e rel[e,i]/need_e.
            The bottleneck view - picks the node that no circuit is starved by.
"""

import numpy as np

from protocol import TOL, THRESHOLD, EPS, relief, cap   # noqa: F401


def score_nodes(rel, viol, need, e0, rank):
    """Rank every wind node for this iteration.  One row per rule."""
    if rank == "worst":
        return rel[e0].copy()
    R = rel[viol]
    if rank == "sum":
        return R.sum(axis=0)
    if rank == "need":
        return (need[:, None] * R).sum(axis=0) / need.sum()
    if rank == "minmax":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.min(R / need[:, None], axis=0) * need.mean()
    raise ValueError(rank)


def arm(V, f, over, avail, s, live, veto=False, rank="worst",
        step="first", iters=20000, thr=None):
    """One node at a time, highest score first, until nothing is over.

    Identical code for every arm.  `rank` and `veto` are the only things that
    change between the control and the method, which is the whole point.
    """
    thr = THRESHOLD if thr is None else thr
    cur, left = f.copy(), avail.copy()
    used = np.zeros(len(avail))
    lim = s * TOL
    blocked = np.zeros(len(avail), bool)

    for _ in range(iters):
        viol = [e for e in over if abs(cur[e]) > lim[e]]
        if not viol:
            break
        need = np.array([abs(cur[e]) - lim[e] for e in viol])
        e0 = viol[int(np.argmax(need))]
        rel = relief(V, cur)
        sc = score_nodes(rel, viol, need, e0, rank)
        sc = np.where((left > EPS) & ~blocked, sc, -np.inf)
        i = int(np.argmax(sc))
        if not np.isfinite(sc[i]) or sc[i] <= thr:
            break

        room = left[i]
        if veto:
            # already-over branches may not get worse; everything else gets
            # its rating.  Both are one statement about lim.
            lm = np.where(np.abs(cur) > lim, np.abs(cur), lim)
            room = min(room, cap(V, cur, lm, live, i))
        if room <= EPS:
            blocked[i] = True      # this pass only - see docstring
            continue

        ri = rel[viol, i]
        if step == "first":
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(ri > 1e-9, need / ri, np.inf)
            want = float(np.min(w))
        else:                       # the old rule, kept so it can be shown
            r0 = rel[e0, i]
            want = (abs(cur[e0]) - lim[e0]) / r0 if r0 > 1e-9 else room
        if not np.isfinite(want):
            want = room
        st = float(min(room, max(want, 1e-4)))
        cur = cur + st * V[:, i]
        left[i] -= st
        used[i] += st
        blocked[:] = False          # flows moved, every veto is stale
    return cur, used


def arm_stale(V, V0, f, over, avail, s, live, rank="worst", thr=None,
              roster_only=False, iters=20000):
    """A per-circuit group drawn from the INTACT network, applied after a trip.

    This is the other control, and it is the operationally honest one.  A
    constraint group is a precomputed list.  It is drawn once, from a study
    case, and then applied in real time whenever that circuit is congested.
    But the congestion being relieved here is *post-contingency*: one branch is
    out, and the shift factors are not the ones the list was drawn from.

    So membership - and, unless ``roster_only``, the ranking and the step size
    too - come from ``V0``, the intact-network sensitivities, because that is
    what a precomputed list contains.  The flows themselves are updated with
    the true post-outage ``V``, because physics does not consult the list.  The
    loop keeps going until the *real* flows are inside their ratings, which is
    what an operator watching real telemetry would do.

    The waste this exposes is not a fairness choice and not an ordering choice.
    It is nodes that looked effective in the study case and are not effective
    now.

    ``roster_only`` keeps the stale membership but ranks and sizes on the true
    post-outage sensitivities - so the difference between the two runs is the
    roster alone.
    """
    thr = THRESHOLD if thr is None else thr
    cur, left = f.copy(), avail.copy()
    used = np.zeros(len(avail))
    lim = s * TOL
    # the operator's belief about direction is the study case's, so the group
    # is drawn on |V0| against the pre-trip flow sense, which is all a static
    # list can encode
    rel0_all = relief(V0, f)

    for _ in range(iters):
        viol = [e for e in over if abs(cur[e]) > lim[e]]
        if not viol:
            break
        need = np.array([abs(cur[e]) - lim[e] for e in viol])
        e0 = viol[int(np.argmax(need))]
        rel_true = relief(V, cur)
        member = rel0_all[e0] >= thr          # the precomputed list
        rel_use = rel_true if roster_only else rel0_all
        sc = np.where(member & (left > EPS), rel_use[e0], -np.inf)
        i = int(np.argmax(sc))
        if not np.isfinite(sc[i]) or sc[i] <= thr:
            break
        ri = rel_use[e0, i]
        want = need[int(np.argmax(need))] / ri if ri > 1e-9 else left[i]
        st = float(min(left[i], max(want, 1e-4)))
        cur = cur + st * V[:, i]              # true physics
        left[i] -= st
        used[i] += st
    return cur, used


def arm_group(V, f, over, avail, s, live, net=False, veto=False, thr=None,
              iters=400):
    """Constraint groups dispatched as UNITS - which is what a group is.

    A constraint group is not a ranked list an operator walks down one node at
    a time.  It is a *set with participation factors*, dispatched together: a
    single instruction scales the whole group,

        x_i = lam * w_i,        w_i = effectiveness_i * available_i

    and lam rises until the constraint is inside its rating.  Everything above
    (protocol.arm_greedy, arms2.arm) is a node-at-a-time greedy with perfect
    re-solution at every step, which is why it lands within a fraction of a
    percent of the LP and leaves the group definition nothing to do.  A real
    group cannot re-solve.  This is the comparison the method was actually
    proposed for.

    net=False   one group per congested circuit, factors from that circuit's
                own effectiveness, circuits taken worst-first.  Relieving
                circuit 2 after circuit 1 cuts every node that belongs to both
                a SECOND time, because the second group does not know the
                first one already cut them.  That double-cut is the waste.

    net=True    ONE group spanning every congested circuit, factors from the
                need-weighted net effectiveness, dispatched once.  A node that
                helps three circuits is cut once, for all three.

    The participation rule (effectiveness x output) is identical in both arms,
    so the equity posture is identical and the difference is the group
    definition and nothing else.
    """
    thr = THRESHOLD if thr is None else thr
    cur, left = f.copy(), avail.copy()
    used = np.zeros(len(avail))
    lim = s * TOL

    def dispatch(w, rel_vec, need):
        """Scale the group by one lam until `need` MW of relief is delivered."""
        nonlocal cur, left, used
        tot = float((rel_vec * w).sum())
        if tot <= 1e-12:
            return False
        lam = need / tot
        x = np.minimum(lam * w, left)
        if veto:
            lm = np.where(np.abs(cur) > lim, np.abs(cur), lim)
            # one cap for the group: the largest common scaling that leaves
            # every live branch inside lim.  A group is dispatched together,
            # so the veto has to bind on the group, not on a node.
            with np.errstate(divide="ignore", invalid="ignore"):
                d = V @ x
                up = np.where(d > 1e-12, (lm - cur) / d, np.inf)
                dn = np.where(d < -1e-12, (-lm - cur) / d, np.inf)
            r = np.minimum(up, dn)
            r[~live] = np.inf
            x = x * max(0.0, min(1.0, float(np.nanmin(r))))
        if x.sum() <= EPS:
            return False
        cur = cur + V @ x
        left = np.maximum(left - x, 0.0)
        used = used + x
        return True

    if net:
        for _ in range(iters):
            viol = [e for e in over if abs(cur[e]) > lim[e]]
            if not viol:
                break
            need = np.array([abs(cur[e]) - lim[e] for e in viol])
            rel = relief(V, cur)
            R = rel[viol]
            nr = (need[:, None] * R).sum(axis=0) / need.sum()   # net effectiveness
            grp = (nr >= thr) & (left > EPS)
            if not grp.any():
                break
            w = np.where(grp, nr * left, 0.0)
            if not dispatch(w, nr, float(need.max())):
                break
    else:
        for e in sorted(over, key=lambda x: -(abs(f[x]) - s[x])):
            for _ in range(iters):
                nd = abs(cur[e]) - lim[e]
                if nd <= 1e-6:
                    break
                rel = relief(V, cur)[e]
                grp = (rel >= thr) & (left > EPS)
                if not grp.any():
                    break
                w = np.where(grp, rel * left, 0.0)
                if not dispatch(w, rel, nd):
                    break
    return cur, used
