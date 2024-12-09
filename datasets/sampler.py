from scipy.spatial import cKDTree
from functools import partial
import numpy as np
import jax
from typing import Optional
import jax.random as jrnd
import jax.numpy as jnp
import flax
import trimesh
from dataclasses import field

@partial(jax.jit, static_argnames=("batch_size","num_points"))
def get_indices(key, batch_size, num_points):
    indices = jrnd.choice(
            key, num_points,
            (batch_size,), replace=True if batch_size > num_points else False)
    return indices


def sample_points_with_normal(mesh, batch_size):
    points, face_index = trimesh.sample.sample_surface(mesh, batch_size)
    normals = []
    if hasattr(mesh, 'face_normals') and len(mesh.face_normals) > 0:
        normals = mesh.face_normals[face_index]
    return jnp.array(points), jnp.array(normals)

    

@partial(jax.jit, static_argnames='n_samples')
def sample_array(key, n_samples, array, n_array):
    sample_indices = jrnd.choice(
        key, len(array),
        (n_samples,),
        replace=False if n_samples < len(array) else True
    )
    sample= array[sample_indices]

    sample_n = []
    if len(n_array) > 0:
        sample_n = n_array[sample_indices]
    return sample, sample_n, sample_indices

def compute_local_sigma(points, k=50):
    if k >= len(points):
        # raise ValueError(f"Cannot find {k=} neighbours with {points.shape=}")
        print(f"WARNING: Cannot find {k=} neighbours with {points.shape=}. Set k={points.shape}.")
        k=points.shape[0]

    if k == 0:
        return np.zeros(len(points))
    sigmas = []
    ptree = cKDTree(points)

    for points_batch in np.array_split(points, 100, axis=0):
        distances = ptree.query(points_batch, k + 1)
        sigmas.append(distances[0][:, -1])
    return np.concatenate(sigmas)


def generate_local_samples(key, points, local_sigma):
    if len(local_sigma) > 0: 
        num_points, dims = points.shape
        noise = jrnd.normal(key, (num_points, dims))
        query_samples = points +  noise * local_sigma[:,None]
    else:
        num_points, dims = points.shape
        noise = jrnd.normal(key, (num_points, dims))
        query_samples = points +  noise
    return jnp.reshape(query_samples, (-1, dims))
    
    
@partial(jax.jit, static_argnames=("n_points", "n_dims"))
def generate_global_samples(key, lower, upper, n_points, n_dims):
    return jrnd.uniform(
        key,
        shape=(n_points, n_dims),
        minval=jnp.array(lower),
        maxval=jnp.array(upper),
    )

def get_bounding_box(points):
    points = points.squeeze()
    object_bbox_min = np.array([np.min(points[...,0]), np.min(points[...,1]), np.min(points[...,2])]) - 0.05
    object_bbox_max = np.array([np.max(points[...,0]), np.max(points[...,1]), np.max(points[...,2])]) + 0.05
    return object_bbox_min -0.2, object_bbox_max + 0.2


def points_k_mean(points, k=50):
    centeriod, labels = trimesh.points.k_means(points, k)
    return centeriod, labels


