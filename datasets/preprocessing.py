import os, re
import numpy as np
import jax.numpy as jnp
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import trimesh
from tqdm import tqdm
import utils.mesh_utils as mesh_utils
from datasets.pointshape import PointShape, DeformPointCloud
import utils.general as utils
from flax import serialization
import pickle
from natsort import natsorted
import datasets.sampler as sampler
import argparse

def sort_list(l):
    return natsorted(l)



class Dress4DShapeGenerate:
    def __init__(self, data_root, seq_num, save_dir, point_num=20000):
        
        assert os.path.isdir(data_root), f'Invaid data root:{data_root}.'
        
        self.data_root = data_root
        self.save_dir = save_dir
        self.point_num=point_num
        
        utils.mkdir_ifnotexists(self.save_dir)
        
        self.save_dir = os.path.join(self.save_dir, 'Take'+str(seq_num))
        utils.mkdir_ifnotexists(self.save_dir)
        
        self.smpl_save_folder = os.path.join(self.save_dir, 'smpl')
        utils.mkdir_ifnotexists(self.smpl_save_folder)
        
        self.mesh_save_folder = os.path.join(self.save_dir, 'mesh')
        utils.mkdir_ifnotexists(self.mesh_save_folder)
        
        self.smpl_folder = os.path.join(data_root, 'Take'+str(seq_num), 'SMPL')
        self.smpl_paths = [f for f in os.listdir(self.smpl_folder) if f.endswith('.ply')]
        self.smpl_paths = sort_list(self.smpl_paths)
        
        self.mesh_folder = os.path.join(data_root, 'Take'+str(seq_num), 'Meshes_pkl')
        self.mesh_paths = [f for f in os.listdir(self.mesh_folder) if f.startswith('mesh')]
        self.mesh_paths = sort_list(self.mesh_paths)
        
        self.ptc_folder = os.path.join(self.save_dir, 'ptc')
        utils.mkdir_ifnotexists(self.ptc_folder)
        
        self.flax_folder = os.path.join(self.save_dir, 'flax_ptc')
        utils.mkdir_ifnotexists(self.flax_folder)
        
        self.train_folder = os.path.join(self.save_dir, 'train')
        utils.mkdir_ifnotexists(self.train_folder)
        
        
        self._size = len(self.mesh_paths)
    
    def __len__(self):
        return self._size
    
    def process(self):
        
        for index in tqdm(range(self._size)):
            self.process_mesh(index)

        print("create point cloud....")
        self.create_pointcloud()
        
        print('create treaining pair....')
        self.create_training_pair()
        
        
        
        
    def process_mesh(self, index):
        mesh_path = os.path.join(self.mesh_folder, self.mesh_paths[index])
        with open(mesh_path, 'rb') as file:
            # Load the pickle object
            data = pickle.load(file)
        
        points = data['vertices']
        shape_scale = np.max([np.max(points[:,0])-np.min(points[:,0]),np.max(points[:,1])-np.min(points[:,1]),np.max(points[:,2])-np.min(points[:,2])])
        shape_center = [(np.max(points[:,0])+np.min(points[:,0]))/2, (np.max(points[:,1])+np.min(points[:,1]))/2, (np.max(points[:,2])+np.min(points[:,2]))/2]
        points = points - jnp.array(shape_center)
        points = points / shape_scale
                
        mesh = trimesh.Trimesh(vertices=points, faces=data['faces'], vertex_normals=data['normals'], vertex_colors=data['colors'])
        
        prefix = self.mesh_paths[index][:-3]
        filename = os.path.join(self.mesh_save_folder, prefix.split('-')[-1] + 'ply')
        mesh.export(filename)
        
        smpl_path = os.path.join(self.smpl_folder, self.smpl_paths[index])
        ptc = trimesh.load(smpl_path)
        points = ptc.vertices
        shape_scale = np.max([np.max(points[:,0])-np.min(points[:,0]),np.max(points[:,1])-np.min(points[:,1]),np.max(points[:,2])-np.min(points[:,2])])
        shape_center = [(np.max(points[:,0])+np.min(points[:,0]))/2, (np.max(points[:,1])+np.min(points[:,1]))/2, (np.max(points[:,2])+np.min(points[:,2]))/2]
        points = points - jnp.array(shape_center)
        points = points / shape_scale
        ptc.vertices = points
        
        prefix = self.smpl_paths[index][:-3]
        filename = os.path.join(self.smpl_save_folder, prefix.split('-')[1].split('_')[0] + '.ply')
        
        ptc.export(filename)

        
    def create_pointcloud(self):
        mesh_paths = [f for f in os.listdir(self.smpl_save_folder) if f.endswith('.ply')]
        mesh_paths = sort_list(mesh_paths)
        for mesh_name in mesh_paths:
            prefix = mesh_name.split('.')[0]
            mesh_files =os.path.join(self.smpl_save_folder, mesh_name)
            save_file = os.path.join(self.flax_folder, prefix)
            points, mesh = create_flax_pointcloud(meshfile=mesh_files, savefile=save_file, point_num=self.point_num, normalize=True)
            filename = os.path.join(self.ptc_folder, prefix + '.ply')
            mesh_utils.save_pointcloud(points, filename)
        print('done.')
            
    
    def create_training_pair(self, step_length=5):
        dptc_list = generate_pesudo_dptc(self.point_num)
        dptc_x, dptc_y= dptc_list
        
        flax_paths = [f for f in os.listdir(self.flax_folder) if f.endswith('.flax')]
        flax_paths = sort_list(flax_paths)
        
        # here we take step length=5
        indices = np.arange(0, len(flax_paths)-step_length, step_length)
        for i in indices:
            dptc_x_file  = os.path.join(self.flax_folder, flax_paths[i])
            with open(dptc_x_file,'rb') as f:
                bytes_data = f.read()
                
            dptc_x = serialization.from_bytes(dptc_x, bytes_data)
            
            dptc_y_file  = os.path.join(self.flax_folder, flax_paths[i+step_length])
            with open(dptc_y_file,'rb') as f:
                bytes_data = f.read()
                
            dptc_y = serialization.from_bytes(dptc_y, bytes_data)
            
            dptc_list = [dptc_x, dptc_y]
            
            bytes_data = serialization.to_bytes(dptc_list)

            filename = os.path.join(self.train_folder, flax_paths[i].split('.')[-2] + '-' +flax_paths[i+5].split('.')[-2])
            
            with open(filename + '.flax', 'wb') as f:
                f.write(bytes_data)
            
            print('save {0}'.format(filename))


class FlaxPairGenerate:
    def __init__(self, data_root, corr_folder, save_dir, point_num=20000):
        
        assert os.path.isdir(data_root), f'Invaid data root:{data_root}.'
        
        self.point_num = point_num
        self.corr_folder = corr_folder
        self.data_root = os.path.join(data_root,'off')
        self.save_dir = save_dir
        
        utils.mkdir_ifnotexists(self.save_dir)
        
        self.corres_paths = [f for f in os.listdir(self.corr_folder) if f.endswith('.npy')]
        self.corres_paths = sort_list(self.corres_paths)
        
        self.mesh_paths = [f for f in os.listdir(self.data_root) if f.endswith('.off')]
        self.mesh_paths = sort_list(self.mesh_paths)
        
        #create dataset
        self.mesh_folder = os.path.join(self.save_dir, 'mesh')
        utils.mkdir_ifnotexists(self.mesh_folder)
        
        self.ptc_folder = os.path.join(self.save_dir, 'ptc')
        utils.mkdir_ifnotexists(self.ptc_folder)
        
        self.flax_folder = os.path.join(self.save_dir, 'flax_ptc')
        utils.mkdir_ifnotexists(self.flax_folder)
        
        self.train_folder = os.path.join(self.save_dir, 'train')
        utils.mkdir_ifnotexists(self.train_folder)

    def __len__(self):
        return len(self.corres_paths)
        

    def process(self):
        for index in range(len(self.corres_paths)):
            corr_xy = np.load(os.path.join(self.corr_folder, self.corres_paths[index]))[0] -1
            mesh_name1 = self.corres_paths[index].split("-")[0]
            mesh_name2 = self.corres_paths[index].split("-")[1][:-8]
            
            # for partial shapes
            try:
                mesh_x = trimesh.load(os.path.join(self.data_root, mesh_name1 + '.off'))
            except:
                null_folder = self.data_root[:-9]
                mesh_x = trimesh.load(os.path.join(null_folder, 'null', 'off', mesh_name1 + '.off'))
            mesh_y = trimesh.load(os.path.join(self.data_root, mesh_name2 + '.off'))
            
            print('mesh_x verts: {0} vs mesh_y verts: {1} vs. corres verts: {2} vs max corr {3}'.format(mesh_x.vertices.shape[0], mesh_y.vertices.shape[0], corr_xy.shape[0], np.max(corr_xy)))
            
            points = mesh_x.vertices
            
            # shape_scale = np.max([np.max(points[:,0])-np.min(points[:,0]),np.max(points[:,1])-np.min(points[:,1]),np.max(points[:,2])-np.min(points[:,2])])
            shape_center = [(np.max(points[:,0])+np.min(points[:,0]))/2, (np.max(points[:,1])+np.min(points[:,1]))/2, (np.max(points[:,2])+np.min(points[:,2]))/2]
            
            # mesh_x.vertices = mesh_x.vertices - shape_center
            # mesh_x.vertices = mesh_x.vertices / 1.5
            
            points = mesh_y.vertices
            # shape_scale = np.max([np.max(points[:,0])-np.min(points[:,0]),np.max(points[:,1])-np.min(points[:,1]),np.max(points[:,2])-np.min(points[:,2])])
            shape_center = [(np.max(points[:,0])+np.min(points[:,0]))/2, (np.max(points[:,1])+np.min(points[:,1]))/2, (np.max(points[:,2])+np.min(points[:,2]))/2]
            
            
            # mesh_y.vertices = mesh_y.vertices - shape_center
            
            if mesh_y.vertices.shape[0] < len(corr_xy):
                corr_xy = corr_xy[:mesh_y.vertices.shape[0]]
            # mesh_y.vertices = mesh_y.vertices / 1.5
            
            if (len(mesh_x.vertices) <= np.max(corr_xy)):
                # print(name_x, name_y)
                valid = np.nonzero(corr_xy < len(mesh_x.vertices))[0]
                verts_y = mesh_y.vertices[valid,:]
                normal_y = mesh_y.vertex_normals[valid,:]
                corr_xy = corr_xy[corr_xy < len(mesh_x.vertices)]
            else:
                verts_y = mesh_y.vertices
                normal_y = mesh_y.vertex_normals
                        

            verts_x = mesh_x.vertices[corr_xy].squeeze()
            normal_x = mesh_x.vertex_normals[corr_xy].squeeze()
            
            shape_x = PointShape(verts = verts_x, normals=normal_x, mesh=mesh_x, name=mesh_name1)
            shape_y = PointShape(verts=verts_y, normals=normal_y, mesh=mesh_y, name=mesh_name2)
            
            dptc_x = init_point_cloud(shape_x, point_num=self.point_num)
            dptc_y = init_point_cloud(shape_y, point_num=self.point_num)
            
            
            dptc_list = [dptc_x, dptc_y]

            filename = os.path.join(self.train_folder, self.corres_paths[index][:-8])

            bytes_data = serialization.to_bytes(dptc_list)
        
            with open(filename + '.flax', 'wb') as f:
                f.write(bytes_data)

            filename = os.path.join(self.mesh_folder, mesh_name1 + '.ply')
            mesh_x.export(filename)
            
            points = jnp.concatenate([dptc_x.verts, dptc_x.points])
            filename = os.path.join(self.ptc_folder, mesh_name1 + '.ply')
            mesh_utils.save_pointcloud(points=points, filename=filename)
        
        
            filename = os.path.join(self.mesh_folder, mesh_name2 + '.ply')
            mesh_y.export(filename)
            
            points = jnp.concatenate([dptc_y.verts, dptc_y.points])
            filename = os.path.join(self.ptc_folder, mesh_name2 + '.ply')
            mesh_utils.save_pointcloud(points=points, filename=filename)
        
        print('done.')


class TemplatePairGenerate:
    def __init__(self, data_root, save_dir=" ", point_num=20000, normalize=False):

        assert os.path.isdir(data_root), f'Invaid data root:{data_root}.'
        
        self.data_root = data_root
        self.save_dir = save_dir
        self.point_num=point_num
        self.normalize = normalize
        
        utils.mkdir_ifnotexists(self.save_dir)

        self.mesh_paths = [f for f in os.listdir(self.data_root) if f.endswith('.off')]
        self.mesh_paths = sort_list(self.mesh_paths)

        #create dataset
        self.mesh_folder = os.path.join(self.save_dir, 'mesh')
        utils.mkdir_ifnotexists(self.mesh_folder)
        
        self.ptc_folder = os.path.join(self.save_dir, 'ptc')
        utils.mkdir_ifnotexists(self.ptc_folder)
        
        self.flax_folder = os.path.join(self.save_dir, 'flax_ptc')
        utils.mkdir_ifnotexists(self.flax_folder)
        
        self.train_folder = os.path.join(self.save_dir, 'train')
        utils.mkdir_ifnotexists(self.train_folder)

        self.size = len(self.mesh_paths)
        self.combination = [[i,j] for i in range(self.size) for j in range(self.size) if i!=j]

    def __len__(self):
        return len(self.combination)
    
    def process(self):
        # create ptc
        for index in range(self.size):
            meshfile = os.path.join(self.data_root, self.mesh_paths[index])
            prefix = self.mesh_paths[index].split('.')[0]
            savefile = os.path.join(self.flax_folder, prefix)
            points, mesh = create_flax_pointcloud(meshfile=meshfile, savefile=savefile, point_num=self.point_num, normalize=self.normalize)
            filename = os.path.join(self.ptc_folder, prefix + '.ply')
            mesh_utils.save_pointcloud(points, filename)
            
            #save meshes
            filename = os.path.join(self.mesh_folder, prefix + '.ply')
            mesh_utils.save_mesh(mesh.vertices, mesh.faces, filename=filename)
        
        #create training pair
        dptc_list = generate_pesudo_dptc(self.point_num)
        dptc_x, dptc_y= dptc_list
        
        flax_paths = [f for f in os.listdir(self.flax_folder) if f.endswith('.flax')]
        flax_paths = sort_list(flax_paths)
        for pair in self.combination:
            i,j = pair
            dptc_x_file  = os.path.join(self.flax_folder, flax_paths[i])
            with open(dptc_x_file,'rb') as f:
                bytes_data = f.read()
                
            dptc_x = serialization.from_bytes(dptc_x, bytes_data)
            dptc_y_file  = os.path.join(self.flax_folder, flax_paths[j])
            with open(dptc_y_file,'rb') as f:
                bytes_data = f.read()
                
            dptc_y = serialization.from_bytes(dptc_y, bytes_data)
            
            dptc_list = [dptc_x, dptc_y]
            
            bytes_data = serialization.to_bytes(dptc_list)

            filename = os.path.join(self.train_folder, flax_paths[i].split('.')[-2] + '-' +flax_paths[j].split('.')[-2])
            
            with open(filename + '.flax', 'wb') as f:
                f.write(bytes_data)
            
            print('save {0}'.format(filename))
    
    
    

def create_flax_pointcloud(meshfile, savefile, point_num=20000, normalize=False):
    
    # normalize meshes
    if normalize:
        verts, normals, center, mesh, scale = mesh_utils.load_mesh(meshfile)
    else:
    # not normalize meshes
        mesh = trimesh.load(meshfile)
        verts = mesh.vertices
        normals = mesh.vertex_normals
        center = np.array([0,0,0])
        scale = 1.0
        
    
    name = meshfile.split('.')[-2]
    point_shape = PointShape(verts=verts, normals=normals, mesh=mesh, center=center, scale=scale, name=name)
    flax_ptc = init_point_cloud(point_shape, point_num=point_num)
    
    bytes_data = serialization.to_bytes(flax_ptc)
    
    with open(savefile + '.flax', 'wb') as f:
        f.write(bytes_data)
        
    print('save {0}'.format(savefile))
    
    points = jnp.concatenate([flax_ptc.verts, flax_ptc.points])
    return points, mesh

def generate_pesudo_dptc(point_num):
    dptc_x = DeformPointCloud(verts=jnp.zeros((5000,3)),
            verts_normals=jnp.zeros((5000,3)),
            points=jnp.zeros((point_num,3)),
            points_normals=jnp.zeros((point_num,3)), 
            local_sigma=jnp.zeros((point_num + 5000,3)),
            upper=jnp.ones((3)),
            lower=-jnp.ones((3)),
            features=jnp.array([])
    )  

    dptc_list = [dptc_x, dptc_x]
    
    return dptc_list


def init_point_cloud(shape_x, point_num=20000):
    points, points_normals = sampler.sample_points_with_normal(shape_x.mesh, batch_size=point_num)
    local_sigma = sampler.compute_local_sigma(jnp.concatenate((shape_x.verts, points)), k=50)
    lower, upper = sampler.get_bounding_box(points)
    dptc_x = DeformPointCloud(verts=shape_x.verts, 
                        verts_normals=shape_x.normals, 
                        points=points, points_normals=points_normals, local_sigma=local_sigma,
                        upper=upper,
                        lower=lower
                        )
    
    return dptc_x



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='original data folder path')
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--seq_num',type=int, default=19, help='only for 4d-dress')
    parser.add_argument('--corr_root',type=str, help='correspondences path from mesh matching method.')
    parser.add_argument('--data_type', type=str, help='FlaxPairGenerate or Dress4DShapeGenerate')
    
    args = parser.parse_args()
    
    if args.data_type =='matching':
        dataset = FlaxPairGenerate(data_root=args.data_root, corr_folder=args.corr_root, save_dir=args.save_dir)
        dataset.process()
    elif args.data_type == 'temporal':
        dataset = Dress4DShapeGenerate(data_root=args.data_root, seq_num=args.seq_num, save_dir=args.save_dir)
        dataset.process()
    elif args.data_type == 'template':
        dataset = TemplatePairGenerate(data_root=args.data_root, save_dir=args.save_dir)
        dataset.process()
    else:
        raise NotImplementedError
    
    