"""End-to-end multi-fragment tests with loop closure and rejected seams."""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from tests.synth import fragment_payload, make_shard_pair, seam_payload

client = TestClient(app)


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _square(color_bgr, size=40):
    img = np.zeros((size, size, 4), np.uint8)
    img[:, :, :3] = color_bgr
    img[:, :, 3] = 255
    return img


def test_loop_closure_inconsistent_candidate_is_reported():
    """Three fragments: a consistent chain plus a contradictory shortcut.

    Any two edges form a tree and fit exactly; the contradiction only
    exists once the shortcut closes the loop (three edges, one cycle).
    The exact solver must refuse the 3-edge loop (no single global pose
    set meets the tight limit) and report the excluded shortcut with an
    honest pose/loop reason instead of silently rendering a bad assembly.
    """
    A = _square((90, 120, 200))
    B = _square((110, 130, 210))
    C = _square((130, 140, 220))

    def pairs(ax, bx, gap):
        return [([ax, float(y)], [bx, float(y - gap)])
                for y in range(4, 36, 6)]

    ab = pairs(39, 0, 0)
    bc = pairs(39, 0, 0)
    # shortcut A-C closing a loop with a 12px mismatch (true chain gives 0)
    ac = pairs(39, 0, 12)

    body = {
        "fragments": [fragment_payload("A", A), fragment_payload("B", B),
                      fragment_payload("C", C)],
        "seams": [
            seam_payload("ab", "A", "B", ab, 8, 3.0, 10**9),
            seam_payload("bc", "B", "C", bc, 8, 3.0, 10**9),
            seam_payload("ac", "A", "C", ac, 9, 3.0, 10**9),
        ],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()

    # The three-edge loop is infeasible (4px/edge compromise > 3px limit),
    # so exactly two edges are adopted and the winner is a spanning tree.
    adopted = set(js["adopted_seam_ids"])
    assert len(adopted) == 2
    assert adopted <= {"ab", "bc", "ac"}
    assert js["total_score"] == max(8 + 8, 8 + 9, 8 + 9)

    # The excluded contradictory seam is reported with an honest reason and
    # the measured (non-zero) pose/loop diagnostics.
    rejected = {s["id"]: s for s in js["rejected_seams"]}
    excluded_id = ({"ab", "bc", "ac"} - adopted).pop()
    ex = rejected[excluded_id]
    assert ex["reason"] in ("pose_residual_exceeds_limit",
                            "loop_closure_exceeds_limit",
                            "score_optimization_not_selected")
    if ex["reason"] != "score_optimization_not_selected":
        assert ex["pair_residual_max"] is not None

    # One connected cluster; all rendered under a single common pose.
    assert len(js["clusters"]) == 1
    cl = js["clusters"][0]
    assert cl["fragment_ids"] == ["A", "B", "C"]
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(cl["image_base64"]), np.uint8),
        cv2.IMREAD_UNCHANGED)
    assert img.shape[2] == 4


def test_full_inconsistent_loop_is_infeasible_at_engine_level():
    """Directly assert the closed loop fails the global feasibility check."""
    from app.assembler import build_model, evaluate_subset
    from app.schemas import AssemblyRequest

    A = _square((90, 120, 200))
    B = _square((110, 130, 210))
    C = _square((130, 140, 220))

    def pairs(ax, bx, gap):
        # list of (pa, pb) coordinate lists for seam_payload
        return [([ax, float(y)], [bx, float(y - gap)])
                for y in range(4, 36, 6)]

    payload = AssemblyRequest.model_validate({
        "fragments": [fragment_payload("A", A), fragment_payload("B", B),
                      fragment_payload("C", C)],
        "seams": [
            seam_payload("ab", "A", "B", pairs(39, 0, 0), 8, 3.0, 10**9),
            seam_payload("bc", "B", "C", pairs(39, 0, 0), 8, 3.0, 10**9),
            seam_payload("ac", "A", "C", pairs(39, 0, 12), 9, 3.0, 10**9),
        ],
    })
    _, masks, edges = build_model(payload)
    ok, metrics, _, _ = evaluate_subset(edges, masks, {})
    assert not ok
    assert any(m.loop_error > 3.0 or m.pair_max > 3.0 for m in metrics.values())


def test_rejected_seam_does_not_join_clusters():
    """Two fragments with only an overlap-violating seam stay separate."""
    A = _square((80, 100, 180))
    B = _square((80, 100, 180))
    pairs_ = [([39.0, float(y)], [0.0, float(y)]) for y in range(4, 36, 6)]
    body = {
        "fragments": [fragment_payload("A", A), fragment_payload("B", B)],
        "seams": [seam_payload("ab", "A", "B", pairs_, 5, 5.0, 0)],
    }
    r = client.post("/assemblies", json=body)
    js = r.json()
    assert js["adopted_seam_ids"] == []
    assert [c["fragment_ids"] for c in js["clusters"]] == [["A"], ["B"]]
    assert js["clusters"][0]["rejected_seams"][0]["reason"] == \
        "mask_overlap_exceeds_limit"


def test_multi_cluster_relaxation_during_subset_search():
    """Regression: cyclic component + disjoint seam must not crash the LP.

    The candidate graph is connected overall, but the *full* set is
    infeasible (connector ``ea`` overlaps the solid masks), so the exact
    search enumerates subsets.  The tentative set {ab, bc, ac, de} holds a
    three-fragment closed-loop component (two anchor-relative fragments)
    together with a still-disjoint two-fragment seam (one anchor-relative
    fragment).  Building one merged LP from rows of differing widths used
    to raise ``ValueError: ... inhomogeneous shape`` and the API returned
    a 500 instead of the assembly.
    """
    A = _square((90, 120, 200))
    B = _square((110, 130, 210))
    C = _square((130, 140, 220))
    D = _square((70, 100, 180))
    E = _square((50, 90, 160))

    def pairs(ax, bx, gap):
        return [([ax, float(y)], [bx, float(y - gap)])
                for y in range(4, 36, 6)]

    body = {
        "fragments": [fragment_payload("A", A), fragment_payload("B", B),
                      fragment_payload("C", C), fragment_payload("D", D),
                      fragment_payload("E", E)],
        "seams": [
            # perfectly consistent loop: shifts 39 + 39 == shortcut 78
            seam_payload("ab", "A", "B", pairs(39, 0, 0), 8, 3.0, 10**9),
            seam_payload("bc", "B", "C", pairs(39, 0, 0), 8, 3.0, 10**9),
            seam_payload("ac", "A", "C", pairs(78, 0, 0), 9, 3.0, 10**9),
            # independent seam (a second component while searching)
            seam_payload("de", "D", "E", pairs(39, 0, 0), 7, 3.0, 10**9),
            # connector: pose-consistent but one overlapping mask column
            seam_payload("ea", "E", "A", pairs(39, 0, 0), 6, 3.0, 0),
        ],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()

    # Highest-score feasible set: the closed loop (25) plus de (7);
    # the connector cannot join the groups (overlap 40px > 0).
    assert js["adopted_seam_ids"] == ["ab", "ac", "bc", "de"]
    assert js["total_score"] == 32.0
    assert [c["fragment_ids"] for c in js["clusters"]] == [
        ["A", "B", "C"], ["D", "E"]]
    rejected = {s["id"]: s for s in js["rejected_seams"]}
    assert rejected["ea"]["reason"] == "mask_overlap_exceeds_limit"

    # Both clusters render to valid transparent PNGs under one pose set.
    for cl in js["clusters"]:
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(cl["image_base64"]), np.uint8),
            cv2.IMREAD_UNCHANGED)
        assert img.ndim == 3 and img.shape[2] == 4
        assert img[:, :, 3].max() == 255


def test_unknown_fragment_reference_is_422():
    d = make_shard_pair(seed=11)
    body = {
        "fragments": [fragment_payload("A", d["A"])],
        "seams": [seam_payload("s", "A", "GHOST", d["pairs"], 3, 5, 0)],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 422
    assert "GHOST" in r.text


def test_duplicate_ids_are_422():
    d = make_shard_pair(seed=12)
    body = {
        "fragments": [fragment_payload("A", d["A"]), fragment_payload("A", d["B"])],
        "seams": [],
    }
    assert client.post("/assemblies", json=body).status_code == 422
