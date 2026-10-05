"""Generate a real, ready-to-send /assemblies request and run it.

Usage:
    python examples/make_example.py            # writes examples/request.json
    python examples/make_example.py --call     # also POSTs to a running server

The request contains five transparent-PNG potsherds and five candidate
seams:

  * shard-A -- shard-B -- shard-C with a shortcut A-C closing a perfectly
    consistent loop (all three seams fit one common set of poses);
  * shard-D -- shard-E, an independent seam that only joins the loop group
    through the candidate graph;
  * shard-E -- shard-A, a connector whose solid masks overlap by a whole
    pixel column under the only pose satisfying its point pairs, so the
    *full* candidate set is infeasible and the exact search has to choose
    a subset.

The expected answer keeps the closed loop plus the independent seam (two
clusters) and reports the connector as ``mask_overlap_exceeds_limit``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import cv2
import numpy as np


def png_b64(img_bgra: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgra)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_tile(color_bgr, size: int = 40) -> np.ndarray:
    """One solid, slightly rounded transparent tile in its own image frame."""
    img = np.zeros((size, size, 4), np.uint8)
    img[:, :, :3] = color_bgr
    img[:, :, 3] = 255
    cv2.circle(img, (0, 0), 6, (0, 0, 0, 0), -1)
    cv2.circle(img, (size - 1, size - 1), 6, (0, 0, 0, 0), -1)
    return img


def seam_pairs(ax: float, bx: float, gap: float = 0.0):
    """Boundary matches: A's right edge at x=ax, B's left edge at x=bx.

    With ``gap`` zero the fitted pose places B exactly ``ax - bx`` pixels
    to the right of A (a vertical column of solid pixels overlaps).
    """
    return [{"a": [float(ax), float(y)], "b": [float(bx), float(y - gap)]}
            for y in range(4, 36, 6)]


def build_request():
    colors = {
        "shard-A": (140, 110, 70),
        "shard-B": (150, 120, 80),
        "shard-C": (160, 130, 90),
        "shard-D": (90, 120, 150),
        "shard-E": (100, 130, 160),
    }
    fragments = [{"id": fid, "image_base64": png_b64(make_tile(col))}
                 for fid, col in colors.items()]

    return {
        "fragments": fragments,
        "seams": [
            {
                # consistent closed loop: A +39px = B, B +39px = C,
                # so the shortcut A-C must shift C by 78px.
                "id": "seam-AB", "fragment_a": "shard-A", "fragment_b": "shard-B",
                "score": 12.5, "pose_error_limit": 3.0, "overlap_limit": 200,
                "pairs": seam_pairs(39, 0),
            },
            {
                "id": "seam-BC", "fragment_a": "shard-B", "fragment_b": "shard-C",
                "score": 12.5, "pose_error_limit": 3.0, "overlap_limit": 200,
                "pairs": seam_pairs(39, 0),
            },
            {
                "id": "seam-AC", "fragment_a": "shard-A", "fragment_b": "shard-C",
                "score": 10.0, "pose_error_limit": 3.0, "overlap_limit": 200,
                "pairs": seam_pairs(78, 0),
            },
            # independent seam (its own component while the search runs)
            {
                "id": "seam-DE", "fragment_a": "shard-D", "fragment_b": "shard-E",
                "score": 8.0, "pose_error_limit": 3.0, "overlap_limit": 200,
                "pairs": seam_pairs(39, 0),
            },
            # connector: pose-consistent, but its 1px-wide overlap column
            # (40 solid px) violates overlap_limit = 0.
            {
                "id": "seam-EA", "fragment_a": "shard-E", "fragment_b": "shard-A",
                "score": 6.0, "pose_error_limit": 3.0, "overlap_limit": 0,
                "pairs": seam_pairs(39, 0),
            },
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true",
                    help="POST the request to a running service")
    ap.add_argument("--url", default="http://127.0.0.1:8000/assemblies")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    req = build_request()
    req_path = os.path.join(here, "request.json")
    with open(req_path, "w") as f:
        json.dump(req, f)
    print("wrote", req_path)

    if args.call:
        import urllib.request
        data = json.dumps(req).encode()
        http_req = urllib.request.Request(
            args.url, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http_req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        out_path = os.path.join(here, "response.json")
        with open(out_path, "w") as f:
            json.dump(body, f, indent=2)
        # also write the rendered cluster PNG for inspection
        for cl in body["clusters"]:
            raw = base64.b64decode(cl["image_base64"])
            p = os.path.join(here, f"cluster_{cl['cluster_id']}.png")
            with open(p, "wb") as f:
                f.write(raw)
        print("wrote", out_path, "and cluster PNGs")


if __name__ == "__main__":
    main()
