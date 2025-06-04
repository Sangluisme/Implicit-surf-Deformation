import numpy as np
from datasets.pointshape import DeformPointCloud
import trimesh
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_corres_color(dptc_x: DeformPointCloud, frequency=1):
    coords = dptc_x.verts
    min_c = np.min(coords, axis=0)
    max_c = np.max(coords, axis=0)
    norm_coords = (coords - min_c) / np.where(max_c - min_c != 0, max_c - min_c, 1e-8)
    colors = np.sin(frequency * norm_coords)
    return colors


def save_corres_color(dptc_x: DeformPointCloud, dptc_y: DeformPointCloud, colors:np.ndarray, path: str, prefix: str):
    
    # colors = get_corres_color(dptc_x, 5)
    colors = colors[:dptc_x.verts.shape[0],:]
    
    filename = os.path.join(path, prefix +'_ptc_x.ply')
    ptc_x = trimesh.points.PointCloud(dptc_x.verts.squeeze(), colors=colors.squeeze())
    ptc_x.export(filename)
    
    ptc_y = trimesh.points.PointCloud(dptc_y.verts.squeeze(), colors=colors.squeeze())
    filename = os.path.join(path, prefix + '_ptc_y.ply')
    ptc_y.export(filename)