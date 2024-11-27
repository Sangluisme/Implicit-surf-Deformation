import numpy as np
from dataclasses import dataclass, field
import jax.numpy as jnp
import trimesh
import jax.random as jrnd
import flax


@flax.struct.dataclass
class PointShape:
    verts: jnp.ndarray
    normals : jnp.ndarray = field(default_factory=jnp.ndarray)
    mesh: trimesh.Trimesh = field(default_factory=trimesh.Trimesh)
    name: str = ""
    center: np.ndarray = np.zeros(3)
    scale: float = 0.0

        
@flax.struct.dataclass
class DeformPointCloud:
    verts: jnp.ndarray
    points: jnp.ndarray 
    local_sigma: jnp.ndarray
    verts_normals: jnp.ndarray = field(default_factory=jnp.ndarray)
    points_normals: jnp.ndarray = field(default_factory=jnp.ndarray)
    lower: jnp.ndarray = -jnp.ones(3)
    upper: jnp.ndarray = jnp.ones(3)
    