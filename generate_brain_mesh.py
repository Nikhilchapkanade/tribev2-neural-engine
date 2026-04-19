"""
generate_brain_mesh.py
Uses nilearn's fsaverage cortical surface to generate a proper neuroscience-grade
brain mesh (.glb) with per-vertex Yeo 7 network region IDs encoded in vertex colors.

This is how the Rewire creator did it — real neuroscience meshes, not Sketchfab downloads.
"""
import numpy as np
import nibabel as nib
from nilearn import datasets
import trimesh
import json
import os

# ── Yeo 7 Network region mapping based on MNI coordinates ────────────────────
# fsaverage vertices are in MNI space:
#   X: left(-) → right(+)
#   Y: posterior(-) → anterior(+) 
#   Z: inferior(-) → superior(+)

YEO7_REGIONS = {
    0: {"name": "visual",           "label": "Visual Cortex",       "edtech_label": "Visual Processing"},
    1: {"name": "somatomotor",      "label": "Somatomotor",         "edtech_label": "Motor Response"},
    2: {"name": "dorsal_attention",  "label": "Dorsal Attention",    "edtech_label": "Sustained Focus"},
    3: {"name": "ventral_attention", "label": "Ventral Attention",   "edtech_label": "Alertness / Surprise"},
    4: {"name": "limbic",           "label": "Limbic",              "edtech_label": "Emotional Engagement"},
    5: {"name": "frontoparietal",   "label": "Frontoparietal",      "edtech_label": "Working Memory"},
    6: {"name": "default_mode",     "label": "Default Mode",        "edtech_label": "Cognitive Drift"},
}

# Encode region ID into vertex color: R channel = regionID/7, G=same, B=0
# This lets the dashboard shader decode which network each vertex belongs to
def region_to_color(region_id):
    """Encode region ID as a distinguishable vertex color."""
    colors = [
        [0, 0, 200, 255],      # 0: Visual — blue
        [0, 200, 0, 255],      # 1: Somatomotor — green  
        [200, 0, 0, 255],      # 2: Dorsal Attention — red
        [200, 200, 0, 255],    # 3: Ventral Attention — yellow
        [200, 0, 200, 255],    # 4: Limbic — magenta
        [0, 200, 200, 255],    # 5: Frontoparietal — cyan
        [150, 100, 50, 255],   # 6: Default Mode — brown
    ]
    return colors[region_id]


def classify_vertex(x, y, z):
    """
    Classify a vertex to one of 7 Yeo networks based on its MNI coordinates.
    This is an approximation — real TRIBE v2 uses voxel-level atlas mapping.
    """
    # Visual Cortex: posterior occipital
    if y < -65:
        return 0  # visual
    
    # Somatomotor: dorsal central strip
    if z > 55 and abs(y) < 25:
        return 1  # somatomotor
    
    # Dorsal Attention: superior parietal
    if z > 35 and y < -20 and y > -65:
        return 2  # dorsal_attention
    
    # Ventral Attention: lateral temporal/inferior frontal
    if abs(x) > 40 and z < 15 and z > -20:
        return 3  # ventral_attention
    
    # Limbic: medial temporal / orbitofrontal
    if z < -5 and y > -20:
        return 4  # limbic
    
    # Frontoparietal: lateral prefrontal
    if y > 20 and abs(x) > 25 and z > 10:
        return 5  # frontoparietal
    
    # Default Mode: medial prefrontal + PCC
    if abs(x) < 20 and (y > 30 or (y < -40 and z > 20)):
        return 6  # default_mode
    
    # Fallback: assign based on position quadrant
    if y > 0:
        return 6 if abs(x) < 25 else 5  # medial=DMN, lateral=frontoparietal
    else:
        return 2 if z > 20 else 0  # high=dorsal attention, low=visual


def main():
    print("Fetching fsaverage cortical surface from nilearn...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    
    # Load left and right hemisphere geometry (GIfTI format)
    print("Loading hemisphere meshes...")
    
    gii_l = nib.load(fsaverage['pial_left'])
    coords_l = gii_l.darrays[0].data  # vertex coordinates
    faces_l = gii_l.darrays[1].data   # face indices
    
    gii_r = nib.load(fsaverage['pial_right'])
    coords_r = gii_r.darrays[0].data
    faces_r = gii_r.darrays[1].data
    
    print(f"Left hemisphere: {len(coords_l)} vertices, {len(faces_l)} faces")
    print(f"Right hemisphere: {len(coords_r)} vertices, {len(faces_r)} faces")
    
    # Combine both hemispheres into one mesh
    n_left = len(coords_l)
    coords = np.vstack([coords_l, coords_r])
    faces = np.vstack([faces_l, faces_r + n_left])
    
    print(f"Combined brain: {len(coords)} vertices, {len(faces)} faces")
    
    # Classify each vertex to a Yeo 7 network region
    print("Classifying vertices to Yeo 7 networks...")
    vertex_regions = np.zeros(len(coords), dtype=int)
    vertex_colors = np.zeros((len(coords), 4), dtype=np.uint8)
    
    region_counts = {i: 0 for i in range(7)}
    
    for i in range(len(coords)):
        x, y, z = coords[i]
        region = classify_vertex(x, y, z)
        vertex_regions[i] = region
        vertex_colors[i] = region_to_color(region)
        region_counts[region] += 1
    
    print("Region distribution:")
    for rid, count in region_counts.items():
        pct = count / len(coords) * 100
        print(f"  {YEO7_REGIONS[rid]['name']:25s}: {count:6d} vertices ({pct:.1f}%)")
    
    # Create trimesh and export as GLB
    print("Creating trimesh and exporting as GLB...")
    mesh = trimesh.Trimesh(
        vertices=coords,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False  # Don't modify the mesh
    )
    
    # Export to GLB (binary glTF)
    output_path = os.path.join("dashboard", "public", "brain.glb")
    mesh.export(output_path, file_type='glb')
    print(f"Exported brain mesh to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # Also export the region mapping as JSON for the dashboard
    region_map = {}
    for i in range(len(coords)):
        region_map[str(i)] = int(vertex_regions[i])
    
    map_path = os.path.join("dashboard", "public", "vertex_regions.json")
    with open(map_path, "w") as f:
        json.dump({
            "total_vertices": len(coords),
            "n_left_hemisphere": n_left,
            "regions": YEO7_REGIONS,
            # Don't export individual vertex mappings (too large)
            # The vertex colors in the .glb already encode this
            "region_counts": {str(k): v for k, v in region_counts.items()},
        }, f, indent=2)
    print(f"Exported region metadata to {map_path}")


if __name__ == "__main__":
    main()
