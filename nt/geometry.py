"""Quad geometry in image space (x right, y down). Corner order is LT, LB, RB, RT.

In that order the shoelace area is negative, because the walk goes down the left edge first.
"""
from __future__ import annotations

import math
from typing import Sequence

Pt = Sequence[float]


def signed_area(pts: Sequence[Pt]) -> float:
    s = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        s += x1 * y2 - x2 * y1
    return s / 2


def is_convex(pts: Sequence[Pt]) -> bool:
    """True for a strictly convex, non-self-intersecting quad."""
    signs = []
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        cx, cy = pts[(i + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross == 0:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def top_edge_angle_deg(pts: Sequence[Pt]) -> float:
    """Angle of the LT->RT edge against +x; positive means the right side is lower in the image."""
    return math.degrees(math.atan2(pts[3][1] - pts[0][1], pts[3][0] - pts[0][0]))


def edge_lengths(pts: Sequence[Pt]) -> list[float]:
    """[left, bottom, right, top] edge lengths."""
    return [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def aspect_ratio(pts: Sequence[Pt]) -> float:
    """Mean horizontal edge length over mean vertical edge length."""
    left, bottom, right, top = edge_lengths(pts)
    vertical = (left + right) / 2
    return float("inf") if vertical == 0 else (bottom + top) / 2 / vertical


def bbox(pts: Sequence[Pt]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def canonical_permutation(pts: Sequence[Pt]) -> tuple[list[int], bool]:
    """Indices that reorder a convex quad into LT, LB, RB, RT, and whether that differs from the input.

    Picks the winding that matches the spec, then the cyclic start whose LT->RT edge is closest to horizontal
    with LT left of RT. This holds up to about 45 degrees of roll; beyond that the labels are ambiguous.
    Non-convex quads are returned unchanged so the validator can flag them.
    """
    identity = [0, 1, 2, 3]
    if not is_convex(pts):
        return identity, False
    order = identity if signed_area(pts) < 0 else identity[::-1]
    best_score, best = None, identity
    for k in range(4):
        cand = order[k:] + order[:k]
        q = [pts[i] for i in cand]
        score = abs(top_edge_angle_deg(q)) + (0.0 if q[3][0] > q[0][0] else 1000.0)
        if best_score is None or score < best_score:
            best_score, best = score, cand
    return best, best != identity
