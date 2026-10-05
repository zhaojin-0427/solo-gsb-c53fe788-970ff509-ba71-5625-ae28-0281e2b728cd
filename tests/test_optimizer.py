"""Tests for the exact global optimizer (branch & bound, never greedy)."""

import itertools

import numpy as np

from app.assembler import Edge, evaluate_subset, optimize
from app.geometry import fit_rigid, rotation_matrix
from app.relax import lp_relaxation_feasible


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mask():
    m = np.zeros((120, 120), bool)
    m[40:80, 10:50] = True
    return m


def _triangle_masks():
    m = np.zeros((120, 120), bool)
    m[40:80, 10:50] = True
    m2 = np.zeros((120, 120), bool)
    m2[40:80, 0:40] = True
    m3 = np.zeros((120, 120), bool)
    m3[40:80, 0:40] = True
    return {"A": m, "B": m2, "C": m3}


def _edge(eid, a, b, rel, src_pts, score, pose_limit, overlap_limit=10**9):
    Rr = rotation_matrix(rel[0])
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = (src_pts - rel[1]) @ Rr
    fit = fit_rigid(dst_pts, src_pts)
    return Edge(eid, a, b, float(score), float(pose_limit), int(overlap_limit),
                src_pts, dst_pts, fit.theta, fit.t, fit.max_residual, fit.rms)


def _stub_edge(eid, score=1.0):
    return Edge(eid, "x", "y", score, 1.0, 0,
                np.zeros((2, 2)), np.zeros((2, 2)),
                0.0, np.zeros(2), 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Real geometry: loop closure forces an edge out
# --------------------------------------------------------------------------- #
def test_loop_inconsistent_edge_is_rejected():
    masks = _triangle_masks()
    pts = np.array([[10.0, y] for y in range(42, 78, 6)])
    ab = _edge("ab", "A", "B", (0.0, np.array([-50.0, 0.0])), pts, 10, 4)
    bc = _edge("bc", "B", "C", (0.0, np.array([-50.0, 0.0])),
               pts + np.array([-50.0, 0.0]), 10, 4)
    # A-C with a 15px loop mismatch; joint LS leaves every edge at 5px > 4
    ac = _edge("ac", "A", "C", (0.0, np.array([-115.0, 0.0])), pts, 10, 4)

    # The full loop is infeasible ...
    ok_full, metrics, _, _ = evaluate_subset([ab, bc, ac], masks, {})
    assert not ok_full
    assert all(m.loop_error > 4.0 for m in metrics.values())

    # ... while each two-edge chain (a tree, no closed loop) is feasible, so
    # the exact solver keeps a score-optimal two-edge combination instead of
    # ever accepting the inconsistent three-edge loop.
    chosen, cache = optimize([ab, bc, ac], masks)
    assert len(chosen) == 2
    ok, _, _, _ = evaluate_subset(chosen, masks, cache)
    assert ok


# --------------------------------------------------------------------------- #
# Search semantics with an injected feasibility oracle
# --------------------------------------------------------------------------- #
def test_optimizer_is_exact_not_greedy():
    # Scores: hi is the best single edge; combining the two lower edges
    # beats it, but any set containing hi is infeasible together with
    # either lower edge. Greedy-by-score returns just hi (score 10); the
    # exact solver returns mid+lo (score 12).
    hi = _stub_edge("hi", score=10)
    mid = _stub_edge("mid", score=6)
    lo = _stub_edge("lo", score=6)

    def feasible(subset):
        ids = {e.id for e in subset}
        if "hi" in ids and (ids & {"mid", "lo"}):
            return False
        return True

    chosen, _ = optimize([hi, mid, lo], {}, feasible=feasible)
    assert sorted(e.id for e in chosen) == ["lo", "mid"]


def test_optimizer_tie_break_lexicographic_id_list():
    e1, e2, e3 = _stub_edge("z"), _stub_edge("a"), _stub_edge("m")
    # all singletons feasible; pairs infeasible -> exactly one chosen
    chosen, _ = optimize([e1, e2, e3], {},
                         feasible=lambda s: len(s) <= 1)
    assert [e.id for e in chosen] == ["a"]


def test_optimizer_tie_break_on_sorted_id_tuple():
    a, b, c = _stub_edge("a"), _stub_edge("b"), _stub_edge("c")
    # maximum feasible size is 2; only {a,b} is infeasible, so the two
    # optimal equal-score sets are {a,c} and {b,c} -> {a,c} wins on
    # lexicographic order of the sorted seam-id list
    chosen, _ = optimize(
        [a, b, c], {},
        feasible=lambda s: len(s) <= 2
        and {e.id for e in s} != {"a", "b"})
    assert sorted(e.id for e in chosen) == ["a", "c"]


def test_optimizer_full_set_shortcut_and_empty():
    masks = _triangle_masks()
    pts = np.array([[10.0, y] for y in range(42, 78, 6)])
    ab = _edge("ab", "A", "B", (0.0, np.array([-50.0, 0.0])), pts, 10, 5)
    chosen, _ = optimize([ab], masks)
    assert [e.id for e in chosen] == ["ab"]
    assert optimize([], masks)[0] == []


# --------------------------------------------------------------------------- #
# LP relaxation soundness, including multi-component (forest) subsets
# --------------------------------------------------------------------------- #
def _consistent_edge(eid, a, b, theta, t, pts, rng, score=1.0, limit=6.0):
    """Edge whose pairs follow a known global placement (closed loops OK)."""
    R = rotation_matrix(theta)
    src_pts = pts + rng.normal(scale=0.05, size=pts.shape)
    dst_pts = src_pts @ R.T + np.asarray(t)
    dst_pts += rng.normal(scale=0.05, size=pts.shape)
    fit = fit_rigid(dst_pts, src_pts)
    return Edge(eid, a, b, score, float(limit), 10**9,
                src_pts, dst_pts, fit.theta, fit.t,
                fit.max_residual, fit.rms)


def test_relaxation_sound_on_random_multi_component_subsets():
    """Exact-feasible => LP-feasible, even when the subset is a forest.

    Random instances mix consistent loops with independent seams so the
    subset graph has multiple connected components with *different* free-
    variable counts.  The per-component LP may only prune exactly infeasible
    sets; pruning a feasible one would silently remove the global optimum.
    """
    rng = np.random.default_rng(2026)
    masks = {c: _mask() for c in "ABCDEF"}
    pts = np.array([[20.0, y] for y in range(42, 78, 7)])

    from app.geometry import compose
    ab_theta, ab_t = 0.05, np.array([40.0, 2.0])
    bc_theta, bc_t = -0.04, np.array([38.0, -3.0])
    ac_theta, ac_t = compose(ab_theta, ab_t, bc_theta, bc_t)
    edges = [
        # consistent closed loop A-B-C-A (ac == ab * bc by construction)
        _consistent_edge("ab", "A", "B", ab_theta, ab_t, pts, rng),
        _consistent_edge("bc", "B", "C", bc_theta, bc_t, pts, rng),
        _consistent_edge("ac", "A", "C", ac_theta, ac_t, pts, rng),
        # a separate seam D-E and link E-F (a tree on its own)
        _consistent_edge("de", "D", "E", 0.1, np.array([42.0, 5.0]), pts, rng),
        _consistent_edge("ef", "E", "F", -0.08, np.array([40.0, -4.0]), pts, rng),
    ]
    # Every listed edge is consistent by construction, so every subset is
    # exactly pose-feasible (overlap limits disabled); the LP must agree for
    # all 2**5 subsets, in particular the multi-component cyclic ones.
    for r in range(1, len(edges) + 1):
        for sub in itertools.combinations(edges, r):
            sub = list(sub)
            ok_exact, _, _, _ = evaluate_subset(sub, masks, {})
            assert ok_exact
            assert lp_relaxation_feasible(sub)

    # And the explicit failure from the bug report: cyclic 3-cluster + seam.
    crashing = [edges[0], edges[1], edges[2], edges[3]]
    assert lp_relaxation_feasible(crashing)

    # Sanity: a genuinely contradictory loop is still pruned infeasible.
    bad = _edge("bad", "A", "C", (0.0, np.array([20.0, 30.0])), pts, 1, 1.0)
    assert not lp_relaxation_feasible([edges[0], edges[1], bad])

