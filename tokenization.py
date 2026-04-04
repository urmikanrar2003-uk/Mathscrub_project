"""
Token Construction — Exact Sequence from MathScrub Fig. 2
==========================================================

Pipeline (left-to-right, top row then bottom row):

  TOP ROW:
    Step 1 — Binarization
    Step 2 — Connected Components
    Step 3 — Delaunay Triangulation

  BOTTOM ROW (right to left in the image, executed left to right here):
    Step 4 — Nested Suppression
    Step 5 — Grouping by Union-Find
    Step 6 — Edge Filtering

  OUTPUT:
    → Tokens (semantically coherent groups of components)

Requirements:
    pip install opencv-python numpy scipy matplotlib python-dotenv tqdm
"""

import os
import io
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay
from PIL import Image


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — BINARIZATION
#  Convert input image to binary (black ink on white background)
#  using Otsu's adaptive thresholding
# ═══════════════════════════════════════════════════════════════

def step1_binarize(image_bgr: np.ndarray) -> np.ndarray:
    """
    Input : BGR image (numpy array)
    Output: Binary image — ink pixels = 255, background = 0
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu automatically finds optimal threshold
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — CONNECTED COMPONENTS
#  Label each ink blob as a separate connected component.
#  Extract centroid, area, bounding box, horizontal interval.
# ═══════════════════════════════════════════════════════════════

def step2_connected_components(binary: np.ndarray, min_area: int = 20):
    """
    Input : Binary image
    Output: List of component dicts, label map

    Each component dict contains:
        id         — unique index
        centroid   — (cx, cy) float
        area       — pixel count
        bbox       — (x, y, w, h)
        h_interval — (x_left, x_right) horizontal projection
    """
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = []
    for i in range(1, num_labels):          # label 0 = background, skip
        x, y, w, h, area = stats[i]
        if area < min_area:                 # filter tiny noise
            continue
        cx, cy = centroids[i]
        components.append({
            "id":         len(components),
            "centroid":   (float(cx), float(cy)),
            "area":       int(area),
            "bbox":       (int(x), int(y), int(w), int(h)),
            "h_interval": (int(x), int(x + w)),   # H(ci) in paper
        })

    return components, label_map


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — DELAUNAY TRIANGULATION
#  Build triangulation over component centroids.
#  Produces a set of candidate edges between neighboring
#  components — these will be filtered in later steps.
# ═══════════════════════════════════════════════════════════════

def step3_delaunay_triangulation(components):
    """
    Input : List of component dicts
    Output: Set of candidate edges as (i, j) index pairs
            where i, j are indices into the components list

    Uses scipy.spatial.Delaunay on centroid coordinates.
    Each triangle in the triangulation contributes 3 edges.
    Duplicate edges are removed (stored as min,max pairs).
    """
    if len(components) < 3:
        # Cannot triangulate fewer than 3 points
        # Return all possible pairs as candidate edges
        edges = set()
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                edges.add((i, j))
        return edges

    centroids = np.array([c["centroid"] for c in components])
    tri = Delaunay(centroids)

    candidate_edges = set()
    for simplex in tri.simplices:           # each simplex = one triangle
        for k in range(3):
            i = simplex[k]
            j = simplex[(k + 1) % 3]
            candidate_edges.add((min(i, j), max(i, j)))

    return candidate_edges


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — NESTED SUPPRESSION
#  Before filtering edges, suppress components that are fully
#  nested (contained) inside a larger component's bounding box.
#  This prevents small symbols inside large deletion strokes
#  from incorrectly anchoring the triangulation.
#
#  A component ci is suppressed if its bbox is fully contained
#  within another component cj's bbox AND cj is significantly
#  larger (area ratio below threshold).
# ═══════════════════════════════════════════════════════════════

def step4_nested_suppression(components, containment_ratio: float = 0.9):
    """
    Input : List of component dicts
            containment_ratio — how much of ci must be inside cj
                                to suppress ci (default 0.9 = 90%)
    Output: Filtered list of components (suppressed ones removed)
            suppressed_ids — set of component ids that were removed

    Logic:
        For each pair (ci, cj), check if ci's bbox is contained
        within cj's bbox. If the overlap between their bboxes
        covers >= containment_ratio of ci's bbox area, suppress ci.
    """
    suppressed_ids = set()

    for i, ci in enumerate(components):
        xi, yi, wi, hi = ci["bbox"]
        ai = ci["area"]

        for j, cj in enumerate(components):
            if i == j:
                continue
            if cj["id"] in suppressed_ids:
                continue

            xj, yj, wj, hj = cj["bbox"]

            # Compute intersection of bboxes
            ix1 = max(xi, xj);  iy1 = max(yi, yj)
            ix2 = min(xi+wi, xj+wj);  iy2 = min(yi+hi, yj+hj)

            if ix2 <= ix1 or iy2 <= iy1:
                continue                    # no overlap at all

            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            ci_bbox_area      = wi * hi

            if ci_bbox_area == 0:
                continue

            # If ci is mostly inside cj AND cj is much larger → suppress ci
            coverage = intersection_area / ci_bbox_area
            if coverage >= containment_ratio and cj["area"] > ai * 2:
                suppressed_ids.add(ci["id"])
                break

    active_components = [c for c in components if c["id"] not in suppressed_ids]
    return active_components, suppressed_ids


# ═══════════════════════════════════════════════════════════════
#  STEP 5 — GROUPING BY UNION-FIND
#  After edges pass geometric filtering, use Union-Find to
#  transitively merge components that are connected through
#  the retained edges into coherent tokens.
# ═══════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def step5_union_find_grouping(components, retained_edges):
    """
    Input : components — active component list (after nested suppression)
            retained_edges — (i, j) pairs that passed edge filtering
    Output: tokens — list of lists, each inner list = component ids
                     forming one semantically coherent token

    Transitive merging: if A-B retained and B-C retained,
    then A, B, C all belong to the same token.
    """
    uf = UnionFind(len(components))

    for i, j in retained_edges:
        uf.union(i, j)

    # Collect groups by root
    groups = {}
    for idx in range(len(components)):
        root = uf.find(idx)
        groups.setdefault(root, []).append(components[idx]["id"])

    tokens = list(groups.values())
    return tokens


# ═══════════════════════════════════════════════════════════════
#  STEP 6 — EDGE FILTERING
#  Apply 4 geometric constraints from Equation (3) in the paper.
#  Only edges passing ALL four constraints are retained.
#
#  Constraints:
#    d_ij   < τ_d      — distance not too large
#    |θ_ij| < τ_θ      — edge is roughly horizontal
#    o_ij   > τ_h      — components overlap horizontally
#    r_ij   > τ_a      — components are similar in size
# ═══════════════════════════════════════════════════════════════

def _euclidean_distance(ci, cj):
    """ d_ij = sqrt((xi-xj)^2 + (yi-yj)^2) """
    xi, yi = ci["centroid"]
    xj, yj = cj["centroid"]
    return float(np.sqrt((xi - xj)**2 + (yi - yj)**2))

def _inclination_angle(ci, cj):
    """ |θ_ij| = |arctan((yi-yj)/(xi-xj))| in degrees """
    xi, yi = ci["centroid"]
    xj, yj = cj["centroid"]
    if abs(xi - xj) < 1e-6:
        return 90.0
    return float(abs(np.degrees(np.arctan2(yi - yj, xi - xj))))

def _horizontal_overlap(ci, cj):
    """ o_ij = |H(ci) ∩ H(cj)| / min(|H(ci)|, |H(cj)|) """
    lo_i, hi_i = ci["h_interval"]
    lo_j, hi_j = cj["h_interval"]
    intersection = max(0, min(hi_i, hi_j) - max(lo_i, lo_j))
    denom = min(hi_i - lo_i, hi_j - lo_j)
    return 0.0 if denom == 0 else intersection / denom

def _area_ratio(ci, cj):
    """ r_ij = min(A(ci), A(cj)) / max(A(ci), A(cj)) """
    ai, aj = ci["area"], cj["area"]
    return 0.0 if max(ai, aj) == 0 else min(ai, aj) / max(ai, aj)


def step6_edge_filtering(
    components,
    candidate_edges,
    tau_d:     float = 150.0,   # max centroid distance (pixels)
    tau_theta: float = 30.0,    # max angle from horizontal (degrees)
    tau_h:     float = 0.3,     # min horizontal overlap ratio
    tau_a:     float = 0.1,     # min area ratio
):
    """
    Input : components — active component list
            candidate_edges — all edges from Delaunay triangulation
            τ_d, τ_θ, τ_h, τ_a — geometric thresholds (Eq. 3)
    Output: retained_edges — edges passing all 4 constraints
            rejected_edges — edges failing at least one constraint
            constraint_log — per-edge constraint values (for debugging)
    """
    retained_edges = []
    rejected_edges = []
    constraint_log = []

    # Build index lookup for fast access
    comp_by_idx = {idx: c for idx, c in enumerate(components)}

    for i, j in candidate_edges:
        ci = comp_by_idx.get(i)
        cj = comp_by_idx.get(j)
        if ci is None or cj is None:
            continue

        d     = _euclidean_distance(ci, cj)
        theta = _inclination_angle(ci, cj)
        o     = _horizontal_overlap(ci, cj)
        r     = _area_ratio(ci, cj)

        passed = (
            d     < tau_d     and   # Constraint 1: proximity
            theta < tau_theta and   # Constraint 2: horizontal alignment
            o     > tau_h     and   # Constraint 3: horizontal overlap
            r     > tau_a           # Constraint 4: similar size
        )

        if passed:
            retained_edges.append((i, j))
        else:
            rejected_edges.append((i, j))

        constraint_log.append({
            "edge": (i, j),
            "d": round(d, 2),
            "theta": round(theta, 2),
            "o": round(o, 3),
            "r": round(r, 3),
            "passed": passed,
            "fail_reasons": {
                "distance":  d     >= tau_d,
                "angle":     theta >= tau_theta,
                "overlap":   o     <= tau_h,
                "area":      r     <= tau_a,
            }
        })

    return retained_edges, rejected_edges, constraint_log


# ═══════════════════════════════════════════════════════════════
#  FULL TOKEN CONSTRUCTION PIPELINE
#  Orchestrates all 6 steps in the exact order from Fig. 2
# ═══════════════════════════════════════════════════════════════

def token_construction(
    image_bgr:          np.ndarray,
    # Step 2 params
    min_area:           int   = 20,
    # Step 4 params
    containment_ratio:  float = 0.9,
    # Step 6 params
    tau_d:              float = 150.0,
    tau_theta:          float = 30.0,
    tau_h:              float = 0.3,
    tau_a:              float = 0.1,
    # Debug
    verbose:            bool  = False,
):
    """
    Full token construction pipeline following Fig. 2 sequence:

        Step 1  Binarization
        Step 2  Connected Components
        Step 3  Delaunay Triangulation
        Step 4  Nested Suppression
        Step 5  Grouping by Union-Find
        Step 6  Edge Filtering

    Returns:
        tokens          — list of component-id lists (one list per token)
        components      — all components after nested suppression
        all_components  — all components before nested suppression
        retained_edges  — edges that passed filtering
        rejected_edges  — edges that were filtered out
        binary          — binarized image
        label_map       — connected component label map
        debug           — dict with intermediate results
    """

    # ── Step 1: Binarization ──────────────────────────────────
    binary = step1_binarize(image_bgr)
    if verbose:
        print(f"[Step 1] Binarization complete. Image shape: {binary.shape}")

    # ── Step 2: Connected Components ─────────────────────────
    all_components, label_map = step2_connected_components(binary, min_area=min_area)
    if verbose:
        print(f"[Step 2] Connected components: {len(all_components)} found "
              f"(min_area={min_area})")

    # ── Step 3: Delaunay Triangulation ───────────────────────
    candidate_edges = step3_delaunay_triangulation(all_components)
    if verbose:
        print(f"[Step 3] Delaunay triangulation: {len(candidate_edges)} candidate edges")

    # ── Step 4: Nested Suppression ───────────────────────────
    components, suppressed_ids = step4_nested_suppression(
        all_components, containment_ratio=containment_ratio
    )
    if verbose:
        print(f"[Step 4] Nested suppression: {len(suppressed_ids)} components suppressed, "
              f"{len(components)} remaining")

    # Rebuild candidate edges on the filtered component set
    # (remove any edges involving suppressed components)
    active_ids = {c["id"] for c in components}
    id_to_new_idx = {c["id"]: new_idx for new_idx, c in enumerate(components)}

    filtered_candidates = set()
    for i, j in candidate_edges:
        old_id_i = all_components[i]["id"] if i < len(all_components) else None
        old_id_j = all_components[j]["id"] if j < len(all_components) else None
        if old_id_i in active_ids and old_id_j in active_ids:
            ni = id_to_new_idx[old_id_i]
            nj = id_to_new_idx[old_id_j]
            filtered_candidates.add((min(ni, nj), max(ni, nj)))

    # ── Step 5 (depends on Step 6 output) ────────────────────
    # Note: In the figure, Nested Suppression → Union-Find → Edge Filtering
    # is the visual layout but the logical dependency is:
    # Edge Filtering must happen BEFORE Union-Find grouping
    # (you need to know which edges to group).
    # Step 5 (Union-Find) is called after Step 6 (Edge Filtering).

    # ── Step 6: Edge Filtering ───────────────────────────────
    retained_edges, rejected_edges, constraint_log = step6_edge_filtering(
        components,
        filtered_candidates,
        tau_d=tau_d,
        tau_theta=tau_theta,
        tau_h=tau_h,
        tau_a=tau_a,
    )
    if verbose:
        print(f"[Step 6] Edge filtering: {len(retained_edges)} retained, "
              f"{len(rejected_edges)} rejected")

    # ── Step 5: Grouping by Union-Find ───────────────────────
    tokens = step5_union_find_grouping(components, retained_edges)
    if verbose:
        print(f"[Step 5] Union-Find grouping: {len(tokens)} tokens formed")

    debug = {
        "candidate_edges":  list(candidate_edges),
        "suppressed_ids":   list(suppressed_ids),
        "constraint_log":   constraint_log,
    }

    return (
        tokens,
        components,
        all_components,
        retained_edges,
        rejected_edges,
        binary,
        label_map,
        debug,
    )


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATION
#  Draw each step's result on the image for inspection
# ═══════════════════════════════════════════════════════════════

COLORS = plt.cm.tab20.colors

def visualize_all_steps(
    image_bgr,
    binary,
    all_components,
    components,
    tokens,
    retained_edges,
    rejected_edges,
    save_path: str = None,
):
    """
    Side-by-side visualization of all 6 steps matching the paper figure layout.
    Top row   : Binarization | Connected Components | Delaunay Triangulation
    Bottom row: Nested Suppression | Grouping by Union-Find | Edge Filtering
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Token Construction Pipeline — MathScrub Fig. 2", fontsize=14, y=1.01)

    # ── Top row ──────────────────────────────────────────────

    # [0,0] Step 1: Binarization
    axes[0, 0].imshow(binary, cmap="gray")
    axes[0, 0].set_title("Step 1 — Binarization", fontweight="bold")
    axes[0, 0].axis("off")

    # [0,1] Step 2: Connected Components (each component a different color)
    comp_vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    for c in all_components:
        x, y, w, h = c["bbox"]
        color = tuple(int(v * 255) for v in COLORS[c["id"] % len(COLORS)][:3])
        cv2.rectangle(comp_vis, (x, y), (x+w, y+h), color, 1)
    axes[0, 1].imshow(comp_vis)
    axes[0, 1].set_title(f"Step 2 — Connected Components ({len(all_components)})",
                          fontweight="bold")
    axes[0, 1].axis("off")

    # [0,2] Step 3: Delaunay Triangulation (all candidate edges)
    tri_vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    centroids_all = np.array([c["centroid"] for c in all_components])
    if len(all_components) >= 3:
        tri = Delaunay(centroids_all)
        for simplex in tri.simplices:
            pts = centroids_all[simplex].astype(int)
            cv2.polylines(tri_vis, [pts], isClosed=True, color=(100, 100, 255), thickness=1)
    for c in all_components:
        cx, cy = int(c["centroid"][0]), int(c["centroid"][1])
        cv2.circle(tri_vis, (cx, cy), 3, (255, 50, 50), -1)
    axes[0, 2].imshow(tri_vis)
    axes[0, 2].set_title("Step 3 — Delaunay Triangulation", fontweight="bold")
    axes[0, 2].axis("off")

    # ── Bottom row ────────────────────────────────────────────

    # [1,0] Step 4: Nested Suppression (show suppressed in red, active in green)
    supp_vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    active_ids = {c["id"] for c in components}
    for c in all_components:
        x, y, w, h = c["bbox"]
        if c["id"] in active_ids:
            cv2.rectangle(supp_vis, (x, y), (x+w, y+h), (50, 200, 50), 1)
        else:
            cv2.rectangle(supp_vis, (x, y), (x+w, y+h), (255, 50, 50), 2)
    axes[1, 0].imshow(supp_vis)
    suppressed_n = len(all_components) - len(components)
    axes[1, 0].set_title(
        f"Step 4 — Nested Suppression\n(green=active, red=suppressed  {suppressed_n} removed)",
        fontweight="bold"
    )
    axes[1, 0].axis("off")

    # [1,1] Step 5: Grouping by Union-Find (tokens in same color)
    token_vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    comp_to_token = {cid: t_idx for t_idx, token in enumerate(tokens) for cid in token}
    comp_lookup   = {c["id"]: c for c in components}
    for c in components:
        t_idx  = comp_to_token.get(c["id"], 0)
        color  = tuple(int(v * 255) for v in COLORS[t_idx % len(COLORS)][:3])
        x, y, w, h = c["bbox"]
        cv2.rectangle(token_vis, (x, y), (x+w, y+h), color, 2)
        cx, cy = int(c["centroid"][0]), int(c["centroid"][1])
        cv2.circle(token_vis, (cx, cy), 3, color, -1)
    # Draw retained edges
    for i, j in retained_edges:
        if i < len(components) and j < len(components):
            pt1 = tuple(np.array(components[i]["centroid"]).astype(int))
            pt2 = tuple(np.array(components[j]["centroid"]).astype(int))
            t_idx = comp_to_token.get(components[i]["id"], 0)
            color = tuple(int(v * 255) for v in COLORS[t_idx % len(COLORS)][:3])
            cv2.line(token_vis, pt1, pt2, color, 1, cv2.LINE_AA)
    axes[1, 1].imshow(token_vis)
    axes[1, 1].set_title(
        f"Step 5 — Grouping by Union-Find\n({len(tokens)} tokens, same color = same token)",
        fontweight="bold"
    )
    axes[1, 1].axis("off")

    # [1,2] Step 6: Edge Filtering (green=retained, red=rejected)
    edge_vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    for i, j in rejected_edges:
        if i < len(components) and j < len(components):
            pt1 = tuple(np.array(components[i]["centroid"]).astype(int))
            pt2 = tuple(np.array(components[j]["centroid"]).astype(int))
            cv2.line(edge_vis, pt1, pt2, (220, 80, 80), 1, cv2.LINE_AA)
    for i, j in retained_edges:
        if i < len(components) and j < len(components):
            pt1 = tuple(np.array(components[i]["centroid"]).astype(int))
            pt2 = tuple(np.array(components[j]["centroid"]).astype(int))
            cv2.line(edge_vis, pt1, pt2, (50, 220, 50), 2, cv2.LINE_AA)
    for c in components:
        cx, cy = int(c["centroid"][0]), int(c["centroid"][1])
        cv2.circle(edge_vis, (cx, cy), 3, (255, 200, 0), -1)
    axes[1, 2].imshow(edge_vis)
    axes[1, 2].set_title(
        f"Step 6 — Edge Filtering\n(green=retained {len(retained_edges)}, "
        f"red=rejected {len(rejected_edges)})",
        fontweight="bold"
    )
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved → {save_path}")
    else:
        plt.show()

    plt.close()


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION: PIL IMAGE ENTRY POINT (for HuggingFace streaming)
# ═══════════════════════════════════════════════════════════════

def process_pil_image(
    pil_image:          Image.Image,
    sample_id:          int,
    output_dir:         Path = None,
    params:             dict = None,
    save_vis:           bool = False,
    save_crops:         bool = False,
    verbose:            bool = False,
):
    """
    Entry point for HuggingFace streamed PIL images.
    Converts PIL → BGR then runs full token_construction pipeline.
    """
    if params is None:
        params = {}

    # Convert PIL to BGR
    if pil_image.mode not in ("RGB",):
        pil_image = pil_image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Run full pipeline
    (tokens, components, all_components,
     retained_edges, rejected_edges,
     binary, label_map, debug) = token_construction(
        image_bgr,
        min_area          = params.get("min_area",          20),
        containment_ratio = params.get("containment_ratio", 0.9),
        tau_d             = params.get("tau_d",             150.0),
        tau_theta         = params.get("tau_theta",          30.0),
        tau_h             = params.get("tau_h",               0.3),
        tau_a             = params.get("tau_a",               0.1),
        verbose           = verbose,
    )

    meta = {
        "sample_id":          sample_id,
        "image_shape":        list(image_bgr.shape),
        "num_all_components": len(all_components),
        "num_suppressed":     len(all_components) - len(components),
        "num_components":     len(components),
        "num_tokens":         len(tokens),
        "num_retained_edges": len(retained_edges),
        "num_rejected_edges": len(rejected_edges),
        "tokens":             tokens,
        "components": [
            {
                "id":         c["id"],
                "centroid":   list(c["centroid"]),
                "area":       c["area"],
                "bbox":       list(c["bbox"]),
                "h_interval": list(c["h_interval"]),
            }
            for c in components
        ],
    }

    if output_dir is not None:
        sample_dir = Path(output_dir) / f"sample_{sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        with open(sample_dir / "tokens.json", "w") as f:
            json.dump(meta, f, indent=2)

        if save_vis:
            visualize_all_steps(
                image_bgr, binary, all_components, components,
                tokens, retained_edges, rejected_edges,
                save_path=str(sample_dir / "pipeline_steps.png"),
            )

        if save_crops:
            crops_dir = sample_dir / "crops"
            crops_dir.mkdir(exist_ok=True)
            comp_lookup = {c["id"]: c for c in components}
            H, W = image_bgr.shape[:2]
            for t_idx, token in enumerate(tokens):
                bboxes = [comp_lookup[cid]["bbox"] for cid in token]
                x1 = max(0, min(b[0] for b in bboxes) - 5)
                y1 = max(0, min(b[1] for b in bboxes) - 5)
                x2 = min(W, max(b[0] + b[2] for b in bboxes) + 5)
                y2 = min(H, max(b[1] + b[3] for b in bboxes) + 5)
                cv2.imwrite(str(crops_dir / f"token_{t_idx:04d}.png"),
                            image_bgr[y1:y2, x1:x2])

    return meta


# ═══════════════════════════════════════════════════════════════
#  QUICK TEST — run on a single image file
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python token_construction.py <image_path> [output_dir]")
        print("Example: python token_construction.py sample.png ./output")
        sys.exit(1)

    image_path = sys.argv[1]
    out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./tc_output")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load: {image_path}")
        sys.exit(1)

    print(f"Running token construction on: {image_path}")

    (tokens, components, all_components,
     retained, rejected, binary, label_map, debug) = token_construction(
        img, verbose=True
    )

    print(f"\nResults:")
    print(f"  All components     : {len(all_components)}")
    print(f"  After suppression  : {len(components)}")
    print(f"  Retained edges     : {len(retained)}")
    print(f"  Rejected edges     : {len(rejected)}")
    print(f"  Tokens formed      : {len(tokens)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    visualize_all_steps(
        img, binary, all_components, components,
        tokens, retained, rejected,
        save_path=str(out_dir / "pipeline_steps.png"),
    )
    print(f"\nDone! Visualization → {out_dir / 'pipeline_steps.png'}")