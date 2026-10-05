"""Linear-programming relaxation of the seam feasibility test.

The exact question is: *does there exist a single set of global fragment
poses for which every selected seam's point residuals, loop-closure error
and (handled separately, non-linear) mask overlap stay within limits?*

Pose corrections are linearised around the spanning-tree propagated poses
(where every tree edge has zero residual).  Variables per non-anchor
fragment: ``(dx, dy, dtheta)``.  Every Euclidean residual
``|r + J @ delta| <= L`` is outer-approximated with a 12-gon
(``n_d . (r + J delta) <= L`` for 12 evenly spaced unit directions),
turning feasibility into an LP.

Because this is an *outer* approximation (linearised geometry, polygonal
norm), ``LP infeasible`` guarantees the true problem is infeasible as
well, so pruning on it can never discard a valid seam set.  LP-feasible
nodes still get the exact Gauss-Newton + rasterised-overlap check.

Subset search may visit a seam set whose graph is still a *forest* (e.g.
a closed-loop component plus a seam not yet linked to it).  Components
never share a fragment, hence their correction variables are independent:
we build one LP per connected component and AND the verdicts.  This is
exactly equivalent to one LP over a unified variable space with a
block-diagonal constraint matrix, but keeps every row a fixed-width
``3 * free_fragments`` vector.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from .geometry import rotation_matrix
from .posegraph import forest_poses

_NDIR = 12
_DIRS = np.column_stack([
    np.cos(np.linspace(0.0, 2.0 * np.pi, _NDIR, endpoint=False)),
    np.sin(np.linspace(0.0, 2.0 * np.pi, _NDIR, endpoint=False)),
])  # (12, 2)
J90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _component_rows(nodes, pose, tree_ids, subset):
    """Constraint rows of the relaxation for one connected component.

    Returns ``(A_rows, b_rows)`` with every row in ``A_rows`` of width
    ``nv = 3 * (len(nodes) - 1)``; the component anchor (smallest id) is
    held fixed, as in the exact solver.
    """
    anchor = nodes[0]
    free = [n for n in nodes if n != anchor]
    col = {n: 3 * i for i, n in enumerate(free)}
    nv = 3 * len(free)

    A_rows: List[np.ndarray] = []
    b_rows: List[float] = []

    def add_cone(constant: np.ndarray, coefs: List[Tuple[str, np.ndarray]],
                 limit: float):
        """``|constant + sum coefs_fid @ delta_fid|_2 <= limit`` (relaxed)."""
        for d in range(_NDIR):
            nxy = _DIRS[d]
            row = np.zeros(nv)
            for fid, M in coefs:
                c = col.get(fid)
                if c is not None:
                    # M is the 2x3 Jacobian block of this residual
                    row[c:c + 3] += nxy @ M
            A_rows.append(row)
            b_rows.append(limit - float(nxy @ constant))

    comp_set = set(nodes)
    for e in subset:
        if e.a not in comp_set:
            continue
        theta_a, t_a = pose[e.a]
        theta_b, t_b = pose[e.b]
        Ra, Rb = rotation_matrix(theta_a), rotation_matrix(theta_b)

        # ---- point-pair residual in world frame -------------------- #
        gap = e.b_pts @ Rb.T + t_b - e.a_pts @ Ra.T - t_a
        # Jacobian blocks (2x3): w.r.t. correction of b and of a.
        for k in range(len(e.a_pts)):
            coefs = [
                (e.b, np.column_stack([np.eye(2), J90 @ (Rb @ e.b_pts[k])])),
                (e.a, np.column_stack([-np.eye(2), -J90 @ (Ra @ e.a_pts[k])])),
            ]
            add_cone(gap[k], coefs, e.pose_limit)

        # ---- loop closure: implied relative motion vs seam's fit ---- #
        if e.id not in tree_ids:
            Rr = Ra.T @ Rb
            # The seam's pairwise fit predicts b points as
            #   p_a^fit = R(local) p_b + t(local)
            # implied under tree poses: Rr p_b + Ra^T(t_b-t_a).
            rel_t = Ra.T @ (t_b - t_a)
            loop_gap_local = (
                e.b_pts @ Rr.T + rel_t
                - (e.b_pts @ rotation_matrix(e.local_theta).T + e.local_t)
            )  # (k, 2), in fragment-a local frame
            # Derivative of Ra^T (Rb pb + tb - ta) w.r.t. corrections:
            #   d tb : Ra^T ; d ta : -Ra^T ;
            #   d theta_b: Ra^T J90 Rb pb ; d theta_a: -J90 Rr pb
            for k in range(len(e.a_pts)):
                Mb = np.zeros((2, 3))
                Mb[:, :2] = Ra.T
                Mb[:, 2] = Ra.T @ (J90 @ (Rb @ e.b_pts[k]))
                Ma = np.zeros((2, 3))
                Ma[:, :2] = -Ra.T
                Ma[:, 2] = -J90 @ (Rr @ e.b_pts[k])
                add_cone(loop_gap_local[k], [(e.b, Mb), (e.a, Ma)],
                         e.pose_limit)
    return A_rows, b_rows


def lp_relaxation_feasible(subset) -> bool:
    """Conservative necessary-feasibility LP over one seam subset.

    The subset may span several still-disconnected components; their pose
    variables are disjoint, so each component is solved independently and
    the whole subset is relaxed-feasible iff every component is.
    """
    if not subset:
        return True

    for nodes, pose, tree_ids in forest_poses(subset):
        A_rows, b_rows = _component_rows(nodes, pose, tree_ids, subset)
        if not A_rows:
            # A component with edges always produces point-pair rows; the
            # only row-free case is a lone node, trivially pose-feasible.
            continue
        res = linprog(np.zeros(len(A_rows[0])),
                      A_ub=np.asarray(A_rows), b_ub=np.asarray(b_rows),
                      bounds=(None, None), method="highs")
        if res.status not in (0, 3):  # 0 optimal / 3 unbounded -> feasible
            return False
    return True
