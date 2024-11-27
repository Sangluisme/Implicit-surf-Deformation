import trimesh
import numpy as np
import jax.numpy as jnp
from numpy import linalg as LA
from plyfile import PlyData
from flax import serialization

def load_mesh(filename):
    ext = filename.split('.')[-1]
    if ext == 'ply' or ext == 'off':
        mesh = trimesh.load(filename)
    
    else:
        raise NotImplementedError
    
    # for debug
    if isinstance(mesh, trimesh.Trimesh):
        # print("This is a mesh.")
        pass
        
    elif isinstance(mesh, trimesh.PointCloud):
        # print("This is a point cloud.")
        pass
    else:
        raise NotImplementedError
    
    points = mesh.vertices
    
    points, center, scale = normalize_pc(points)
    
    mesh.vertices = points
    
    normals = jnp.array(mesh.vertex_normals)
    
    return points, normals, center, mesh, scale

def load_mesh_with_normals(filename):
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, 11), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["nx"]
        vertices[:, 4] = plydata["vertex"].data["ny"]
        vertices[:, 5] = plydata["vertex"].data["nz"]
    points = vertices[:,:3]
    normals = vertices[:3:6]
    
    points, center, scale = normalize_pc(points)
    
    return points, normals, center, scale
    

def normalize_pc(point_set):
    pnts = jnp.array(point_set[:,:3])
  
    shape_scale = np.max([np.max(pnts[:,0])-np.min(pnts[:,0]),np.max(pnts[:,1])-np.min(pnts[:,1]),np.max(pnts[:,2])-np.min(pnts[:,2])])
    shape_center = [(np.max(pnts[:,0])+np.min(pnts[:,0]))/2, (np.max(pnts[:,1])+np.min(pnts[:,1]))/2, (np.max(pnts[:,2])+np.min(pnts[:,2]))/2]
    pnts = pnts - jnp.array(shape_center)
   
    # pnts = pnts / shape_scale
    shape_scale = 1

    point_set[:,:3] = pnts

    return jnp.array(point_set), jnp.array(shape_center), shape_scale


def points_k_mean(points, k=50):
    centeriod, labels = trimesh.points.k_means(points, k)
    return centeriod, labels


def save_pointcloud(points, filename="", color=None):
    
    if color is not None:
        ptc = trimesh.points.PointCloud(points.squeeze(), colors=color.squeeze())
    else:
        ptc = trimesh.points.PointCloud(points.squeeze())
    ptc.export(filename)
    
    
def save_mesh(verts, faces, color=None, filename=""):
    if color is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=color)
    else:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)
    
    return mesh


def save_ply(mesh, filename=""):
    mesh.export(filename + '.ply')
    return mesh

def get_bounding_box(points):
    points = points.squeeze()
    object_bbox_min = np.array([np.min(points[...,0]), np.min(points[...,1]), np.min(points[...,2])]) - 0.05
    object_bbox_max = np.array([np.max(points[...,0]), np.max(points[...,1]), np.max(points[...,2])]) + 0.05
    return object_bbox_min -0.2, object_bbox_max + 0.2



def save_deform_pointcloud(filename, dptc):
    bytes_data = serialization.to_bytes(dptc)
    with open(filename+'.flax', 'wb') as f:
        f.write(bytes_data)
        

def load_deform_pointcloud(filename, dptc):
    with open(filename +'.flax', 'rb') as f:
        bytes_data = f.read()
    loaded_instance = serialization.from_bytes(dptc, bytes_data)
    return loaded_instance


    