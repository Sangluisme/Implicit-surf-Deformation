import jax.numpy as jnp
import flax
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from dataclasses import dataclass
from typing import Tuple
from jax import lax
import jax
import os
import functools
import time
import plyfile
from utils.mesh_utils import save_mesh


_color_blue = jnp.array([121, 247, 176])
_color_red = jnp.array([204, 0, 0])
_color_gray = jnp.array([96,96,96])

def grid_slice(x, step=64**3):
    num_points, dim = x.shape
    x_arr = []
    for N in range(0, num_points, step):
        if N + step < num_points:
            x_arr.append(lax.slice(x, (N,0), (N + step,dim)))
        else:
            x_arr.append(lax.slice(x, (N,0), (num_points,dim)))
    return x_arr
#   return [lax.slice(x, (N,), (N + step,)) for N in range(0, x.shape[0], step)]

def iter_grid(lower, upper, n, batch_size):
    n_dims = len(lower)
    axis_batch_size = max(1, int(batch_size / n ** (n_dims - 1)))

    axis_values = [jnp.linspace(lower[i], upper[i], n) for i in range(len(lower))]
    for i in range(0, n, axis_batch_size):
        grid_values = jnp.meshgrid(axis_values[0][i:i+axis_batch_size], *axis_values[1:], indexing='ij')
        yield jnp.stack(grid_values, axis=-1)
        

@flax.struct.dataclass
class PlotManager:
    directory: str=" "
    lower_bound: np.array = np.array([-1.0, -1.0, -1.0])
    upper_bound: np.array = np.array([1.0, 1.0, 1.0])
    f_batch_size: int = 64 ** 3
    resolution: int=128
    levelset: float = 0.0
    checkpoint_intervel: int=1000
    vertex_size: int=1000
    prefix: str=" "
    

    def __call__(self, upper, lower, vertex_size, prefix):
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "vertex_size", vertex_size)
        object.__setattr__(self, "prefix", prefix)
    
    def should_save(self, epoch):
        return True if (epoch % self.checkpoint_intervel == 0) and (epoch > 0) else False
    
    def extract_mesh(self, f, time_step, connected=True) -> trimesh.Trimesh:
      
        outputs_numpy = []
        grid = get_grid([self.lower_bound, self.upper_bound], self.resolution)
        xx,yy,zz = grid['grid_points']

        grid_points = jnp.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        
        print('start tracing voxels....')
        start_time = time.time()

        voxel_slice = grid_slice(grid_points, step=self.f_batch_size)
        print('slicing finished. Total {0} slice to tract....'.format(len(voxel_slice)))
        for inputs in voxel_slice:
            t =jnp.tile(time_step, (inputs.shape[0],1))
            outputs_numpy.append(f(inputs, t=t))
        
        tracing_time = time.time()-start_time
        print('collapse time: {0}s'.format(tracing_time))
        
        outputs_numpy = np.concatenate(outputs_numpy).reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2])

        try:
            vertices, faces, normals, _ = marching_cubes(
                volume=outputs_numpy,
                level=self.levelset,
                spacing=(
                        grid['xyz'][0][2] - grid['xyz'][0][1],
                    grid['xyz'][0][2] - grid['xyz'][0][1],
                    grid['xyz'][0][2] - grid['xyz'][0][1]), 
                method="lewiner")
            
            vertices = vertices + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
            mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
            if connected:
                    connected_comp = mesh.split(only_watertight=False)
                    max_area = 0
                    max_comp = None
                    for comp in connected_comp:
                        if comp.area > max_area:
                            max_area = comp.area
                            max_comp = comp
                    mesh = max_comp
        except ValueError:
            print('marching cubes: no 0-level set found')
            vertices, faces, normals = np.array([]).reshape((0, 3)), [], None
            mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
        return mesh
    

    def visualize_velocity(self, input_x, f, time_step=None):
        points_slice = grid_slice(input_x, step=self.f_batch_size)
        
        output = []
        for p in points_slice:
            if time_step is not None:
                t =jnp.tile(time_step, (p.shape[0],1))
                velocity = f(p, t)
            else:
                velocity = f(p)

            points = p + velocity
            output.append(points)

        return jnp.concatenate(output)
    

    def get_color(self, input_x):
        color1 = jnp.repeat(_color_red[jnp.newaxis,:], self.vertex_size, axis=0)
        color2 = jnp.repeat(_color_gray[jnp.newaxis,:], input_x.shape[-2]-self.vertex_size, axis=0)

        color = jnp.concatenate((color1, color2))

        return color


    def save_input(self, model_input, filename):
        input_x = model_input['input_x']
        color1 = jnp.repeat(_color_red[jnp.newaxis,:], self.vertex_size, axis=0)
        color2 = jnp.repeat(_color_gray[jnp.newaxis,:], input_x.shape[-2]-self.vertex_size, axis=0)

        color = jnp.concatenate((color1, color2))

        self.save_points_ply(input_x, color, filename + 'input_x')

        if 'input_t' in model_input.keys():
            input_y = model_input['input_y']
            color1 = jnp.repeat(_color_red[jnp.newaxis,:], self.vertex_size, axis=0)
            color2 = jnp.repeat(_color_gray[jnp.newaxis,:], input_y.shape[-2]-self.vertex_size, axis=0)

            color = jnp.concatenate((color1, color2))

            self.save_points_ply(input_x, color, filename + 'input_y')
    

    def save_ply(self, mesh: trimesh.Trimesh, output_file):
        filename = os.path.join(self.directory, output_file + '.ply')
        mesh.export(filename, 'ply')
        
    def save_mesh(self, verts, faces, color=None, filename=""):
        filename = os.path.join(self.directory, filename + '.ply')
        save_mesh(verts=verts, faces=faces, color=color, filename=filename)



    def save_points_ply(self, points, normals=None, color=None, output_file=""):
        filename = os.path.join(self.directory, output_file + '.ply')
        
               
        if normals is not None:
            self.save_ptc_with_normal(points, normals, color, filename)
        else:
        
            if color is not None:
                ptc = trimesh.points.PointCloud(vertices=points.squeeze(),  colors=color.squeeze())
            else:
                ptc = trimesh.points.PointCloud(vertices=points.squeeze())
    
                
            ptc.export(filename)
    
    
    def save_ptc_with_normal(self, points, normals, color=None, output_file=""):
        num_verts = points.shape[0]

        dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        
        verts = np.hstack((points, normals))
        
        verts_tuple = np.zeros(
        (num_verts,),
        dtype=dtypes
        )

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(verts[i, :])
        
        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        # el_normal = plyfile.PlyElement.describe(vertex, "vertex")
        ply_data = plyfile.PlyData([el_verts])
        
        ply_data.write(output_file)
        
    
    def get_grid(self):
        grid = get_grid([self.lower_bound, self.upper_bound], self.resolution)
        return grid


def get_grid(bounding_box, resolution):
    
    input_min = bounding_box[0]
    input_max = bounding_box[1]
    bounding_box = input_max - input_min
    
    eps = max(bounding_box) / 2 * 0.1
    
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = jnp.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = jnp.max(x) - jnp.min(x)
        y = jnp.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = jnp.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = jnp.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = jnp.max(y) - jnp.min(y)
        x = jnp.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = jnp.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = jnp.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = jnp.max(z) - jnp.min(z)
        x = jnp.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = jnp.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = jnp.meshgrid(x, y, z)
    # grid_points = torch.tensor(np.vstack([xx.T.ravel(), yy.T.ravel(), zz.T.ravel()]).T, dtype=torch.float).cuda()
    
    return {"grid_points":(xx,yy,zz),
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "shortest_axis_index":shortest_axis,
            }