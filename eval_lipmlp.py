import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  
import numpy as np
import jax.numpy as jnp
import jax.random as jrnd
from functools import partial
import sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable
from pyhocon import ConfigFactory
import argparse
from tqdm import tqdm
from orbax.checkpoint import CheckpointManagerOptions, CheckpointManager, PyTreeCheckpointer
import orbax.checkpoint as ocp
from flax.training import orbax_utils
# from flax import serialization
from models.plot_manager import PlotManager
import utils.general as utils
import utils.mesh_utils as mesh_utils
from models.utils import normalize_params


@dataclass
class CustomCheckpointMangerOptions(CheckpointManagerOptions):
    save_interval_steps: int = 1000

    # name of metric to use for checkpointing
    best_metric: Optional[str] = None


def setup(args):

    modeldir = args.modeldir
    conf_path = os.path.join(modeldir,'runconf.conf')
    conf = ConfigFactory.parse_file(conf_path)

    conf.T = args.steps
    
    utils.mkdir_ifnotexists(os.path.join(modeldir, 'interp'))
    plots_dir = os.path.abspath(os.path.join(modeldir, 'interp'))

    checkpoint_option = CustomCheckpointMangerOptions(**conf.check_point)
    
    checkpoint_manager = CheckpointManager(
        directory=os.path.join(modeldir,'checkpoints'),
        checkpointers=PyTreeCheckpointer(),
        options=checkpoint_option
    )
    
    plot_manager = PlotManager(
        directory=plots_dir,
        resolution=args.mc_resolution,
        f_batch_size=10000
    )
    
    dataset = utils.get_class(conf.dataset_class)(**conf.datasets)

    return conf, checkpoint_manager, plot_manager, dataset

def eval_all(conf, checkpoint_manager, plot_manager, dataset):

    np.random.seed(conf.training.rng_seed)
    key = jrnd.PRNGKey(conf.training.rng_seed)
    
    implicit_net = utils.get_class(conf.implicit_class)(**conf.network.implicit_net)
    
    learning_rate_fn = partial(
        utils.step_learning_rate_decay,
        initial=conf.training.initial,
        interval=conf.training.interval,
        factor=conf.training.factor)
    
    model = utils.get_class(conf.method)(**conf.loss)

    key, = jrnd.split(key, 1)

    implicit_train_state = model.model_init(key, learning_rate_fn, implicit_net, conf.network.implicit_net)

    try:
        step = checkpoint_manager.latest_step()
        target = {'model':implicit_train_state, 'index': 0, 'pair': [0,1], 'upper':np.array([0.5,0.8,0.4]), 'lower': np.array([-0.5,-0.8,-0.4])}
        restored = checkpoint_manager.restore(step, items=target)
        implicit_train_state = restored['model']
        pair = restored['pair']
        lower = np.array([-0.5,-0.8,-0.4])
        upper = np.array([0.5,0.8,0.4])
        subindex = restored['index']
        pair = restored['pair']
        

    except TypeError:
        print('failed to load model.....')

    print(" interpolate for {0} -----> {1}...............".format(pair[0], pair[1]))
    
    dptc_list = dataset.generate_pesudo_dptc(20000)
   
    dptc_x, dptc_y = dataset.getitem(subindex, dptc_list)
    
    bounding_box = mesh_utils.get_bounding_box(jnp.concatenate((dptc_x.points, dptc_y.points)))
    prefix = dataset.mesh_paths[subindex][:-5] + '_'

    plot_manager(lower=bounding_box[0], upper=bounding_box[1], vertex_size = len(dptc_x.verts), prefix=prefix)
    # save gt shapes
    plot_manager.save_points_ply(jnp.concatenate((dptc_x.verts, dptc_x.points)), normals=jnp.concatenate((dptc_x.verts_normals, dptc_x.points_normals)), output_file=prefix + '0_gt')
    plot_manager.save_points_ply(jnp.concatenate((dptc_y.verts, dptc_y.points)), normals=jnp.concatenate((dptc_y.verts_normals, dptc_y.points_normals)), output_file=prefix + '1_gt')
    

    if not args.skip_recon:
        
        if conf.method[-3:] == 'Lip':
            params_final = normalize_params(implicit_train_state.params)
    
            implicit_fn = partial(implicit_train_state.apply_fn, params_final)
        else:
            
            implicit_fn = partial(implicit_train_state.apply_fn, implicit_train_state.params)
        
        for time_step in range(conf.T+1):
            print('interpolate step {0}...'.format(time_step))
            t = time_step / conf.T

            mesh = plot_manager.extract_mesh(implicit_fn, t)
            filename = plot_manager.prefix + '_time_' + str(time_step).zfill(2) + '_mesh'
            plot_manager.save_ply(mesh, filename)
            
    print('done.')


if __name__ == "__main__":
    
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default='./exp/fit_fraust/')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--mc_resolution',type=int, default=256)
    parser.add_argument('--skip_recon', action='store_true')

    args = parser.parse_args()
    
    conf, checkpoint_manager, plot_manager, dataset = setup(
        args
    )

    eval_all(conf, checkpoint_manager, plot_manager, dataset)