import os, re
import numpy as np
import jax.numpy as jnp
import sys
from itertools import product
from glob import glob
import trimesh
sys.path.append('./Implicit-surf-Deformation')
from datasets.pointshape import DeformPointCloud
from natsort import natsorted
import jax.random as jrnd
import jax
from flax import serialization
from functools import partial


def sort_list(l):
    return natsorted(l)


class FlaxSequenceShape:
    
    def __init__(self, data_root, num_shape, subindex=None, batch_size=5000, load=False):
        
        assert os.path.isdir(data_root), f'Invaid data root.'
        
        self.flax_folder = os.path.join(data_root,'train')
        self.data_root = data_root
        
        self.flax_files = [f for f in os.listdir(self.flax_folder) if f.endswith('.flax')]
        self.mesh_paths = sort_list(self.flax_files)
        
        # external test mesh
        self.external = os.path.join(data_root,'mesh')
        self.external_files = [f for f in os.listdir(self.external) if f.endswith('.ply')]
        self.external_files = sort_list(self.external_files)
        
        
        self.flax_dptcs = []
        
        self.num_shape = num_shape
        self.batch_size = batch_size
        self.combinations = self.generate_index_pair()
        
        self.load = load

        self.person_meshes = []
        self.scene_meshes = []
        
        
        if subindex is not None:
            selected = []
            combination = []
            for i in subindex:
                selected.append(self.mesh_paths[i])
                combination.append(self.combinations[i])
            
            self.mesh_paths = selected
            self.combinations = combination
        
        
        if self.load:
            self.load = False
            dptc_list = self.generate_pesudo_dptc(20000)
            for i in range(len(self.mesh_paths)):
                dptc_list = self.getitem(i, dptc_list)
                self.flax_dptcs.append(dptc_list)
            
            self.load = True
        
        
                       
        
    def __len__(self):
        return len(self.mesh_paths)
    
    
    def generate_index_pair(self):
        combination = [[i, i + 1] for i in range(len(self.mesh_paths))]
        return combination

    
    
    def getitem(self, index, dptc_list):
        if not self.load:
            filename = os.path.join(self.flax_folder, self.mesh_paths[index])
            with open(filename,'rb') as f:
                bytes_data = f.read()
                
            loaded_instance = serialization.from_bytes(dptc_list, bytes_data)
            
            dptc_x, dptc_y = loaded_instance
        else:
            dptc_list = self.flax_dptcs[index]
            dptc_x, dptc_y = dptc_list
        

        return dptc_x, dptc_y
    
    
    def get_index_pair(self, index):
        return self.combination[index]
    
    
    def generate_pesudo_dptc(self, point_num):
        dptc_x = DeformPointCloud(verts=jnp.zeros((5000,3)),
                verts_normals=jnp.zeros((5000,3)),
                points=jnp.zeros((point_num,3)),
                points_normals=jnp.zeros((point_num,3)), 
                local_sigma=jnp.zeros((point_num + 5000,3)),
                upper=jnp.ones((3)),
                lower=-jnp.ones((3))
        )  

        dptc_list = [dptc_x, dptc_x]
    
        return dptc_list


class FlaxPairShape:
    def __init__(self, data_root, num_shape, subindex=None,  batch_size=5000, load=False):
        
        assert os.path.isdir(data_root), f'Invaid data root.'
        
        self.flax_folder = os.path.join(data_root,'train')
        self.data_root = data_root
        
        self.test_folder = os.path.join(data_root, 'ptc')
        self.test_files = [f for f in os.listdir(self.test_folder) if f.endswith('.ply')]
        self.test_paths = sort_list(self.test_files)
        
        self.flax_files = [f for f in os.listdir(self.flax_folder) if f.endswith('.flax')]
        self.mesh_paths = sort_list(self.flax_files)
        
        # external test mesh
        self.external = os.path.join(data_root,'mesh')
        self.external_files = [f for f in os.listdir(self.external) if f.endswith('.ply')]
        self.external_files = sort_list(self.external_files)
    
    
        self.flax_dptcs = []
        
        self.num_shape = num_shape
        self.batch_size = batch_size
        self.combinations = self.generate_index_pair()
        
        self.load = load

        self.person_meshes = []
        self.scene_meshes = []
        
        if subindex is not None:
            selected = []
            combination = []
            for i in subindex:
                selected.append(self.mesh_paths[i])
                combination.append(self.combinations[i])
            
            self.mesh_paths = selected
            self.combinations = combination
        
        if self.load:
            self.load = False
            dptc_list = self.generate_pesudo_dptc(20000)
            for i in range(len(self.mesh_paths)):
                dptc_list = self.getitem(i, dptc_list)
                self.flax_dptcs.append(dptc_list)
            
            self.load = True
        
    def __len__(self):
        return len(self.mesh_paths)
    
    def generate_index_pair(self):
        shape_num = len(self.external_files)
        map_list = {}
        combination = []
        
        for index in range(shape_num):
            map_list.update({self.external_files[index][:-4]: index})
        
        for mesh_name in self.mesh_paths:
            mesh_x = mesh_name.split('-')[0]
            mesh_y = mesh_name.split('-')[1][:-5]
            
            index1 = map_list[mesh_x]
            index2 = map_list[mesh_y]
            combination.append([index1, index2])
            
        return combination
        
        
    def getitem(self, index, dptc_list):
        if not self.load:
            filename = os.path.join(self.flax_folder, self.mesh_paths[index])
            with open(filename,'rb') as f:
                bytes_data = f.read()
                
            loaded_instance = serialization.from_bytes(dptc_list, bytes_data)
            
            dptc_x, dptc_y = loaded_instance
        else:
            dptc_list = self.flax_dptcs[index]
            dptc_x, dptc_y = dptc_list
        
        return dptc_x, dptc_y
    
    
    def get_index_pair(self, index):
        return self.combination[index]
    
    
    def getitem_meshes(self, prefix):
        mesh = trimesh.load(os.path.join(self.external, prefix +'.ply'))
        return mesh

    
    def generate_pesudo_dptc(self, point_num):
        dptc_x = DeformPointCloud(verts=jnp.zeros((5000,3)),
                verts_normals=jnp.zeros((5000,3)),
                points=jnp.zeros((point_num,3)),
                points_normals=jnp.zeros((point_num,3)), 
                local_sigma=jnp.zeros((point_num + 5000,3)),
                upper=jnp.ones((3)),
                lower=-jnp.ones((3)),
        )  

        dptc_list = [dptc_x, dptc_x]

        return dptc_list


if __name__ == "__main__":
    
    data_root = './data/faust_r'
    num_shape = 20
    subindex = [1,2,3]
    
    flax_seq = FlaxPairShape(data_root=data_root, num_shape=num_shape, subindex=subindex)
    
    dptc_list = flax_seq.generate_pesudo_dptc(20000)
    dptc_x, dptc_y = flax_seq.getitem(0, dptc_list)