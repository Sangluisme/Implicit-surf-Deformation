import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  
import numpy as np
import jax.numpy as jnp
import jax.random as jrnd
import jax
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
from datasets.pointshape import DeformPointCloud
import utils.general as utils
import utils.mesh_utils as mesh_utils
import json
import time


@dataclass
class CustomCheckpointMangerOptions(CheckpointManagerOptions):
    save_interval_steps: int = 1000

    # name of metric to use for checkpointing
    best_metric: Optional[str] = None


def save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index):
    # save_args=ocp.args.StandardSave(save_args)
    checkpoint_manager.save(
        step=batch_index,
        items={'model': train_state, **checkpoint_info},
        save_kwargs={'save_args': save_args},
        # args=ocp.args.StandardSave(save_args),
        force=True,
    )

def check_best(metrics: list[dict], latest_metrics: dict, metric_name: str):
    if (metric_name is None) or (len(metrics) == 0):
        return True
    else:
        return latest_metrics[metric_name] < min([m[metric_name] for m in metrics])

def setup(args):
    devices = jax.local_devices()
    print(devices) # >>>
    
    conf = ConfigFactory.parse_file(args.conf)
    savedir = args.savedir
    expname = args.expname
    
    conf.wandb_log = True if args.log else False
    reset = True if args.reset else False
    conf.reset = reset
    
    # setup fine tune steps
    if conf.training.fine_tune:
        conf.training.end = conf.training.nepochs
        # conf.training.nepochs = int(conf.training.nepochs * 0.8)
        conf.training.nepochs = conf.training.full
    
    utils.mkdir_ifnotexists(os.path.join(savedir, expname))
    expdir = os.path.abspath(os.path.join(savedir, expname))
    
    if not reset:
        experiment_folders = os.listdir(expdir)
        if (len(experiment_folders)) == 0:
            reset = False
            timestamp = None
        else:
            timestamp = sorted(experiment_folders)[-1]
    
    else:
        timestamp = None
        
    if timestamp is None:
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    
    utils.mkdir_ifnotexists(os.path.join(expdir,timestamp))
    expdir = os.path.join(os.path.join(expdir,timestamp))

    conf.expdir = expdir

    checkpoints_path = os.path.join(expdir, 'checkpoints')
    utils.mkdir_ifnotexists(checkpoints_path)
    
    plots_dir = os.path.join(expdir, 'reconstructions')
    utils.mkdir_ifnotexists(plots_dir)

    # save config for later reference
    os.system("""cp -r {0} "{1}" """.format(args.conf, os.path.join(expdir, 'runconf.conf')))
    
    checkpoint_option = CustomCheckpointMangerOptions(**conf.check_point)
    
    checkpoint_manager = CheckpointManager(
        directory=checkpoints_path,
        checkpointers=PyTreeCheckpointer(),
        options=checkpoint_option
    )
    
    plot_manager = PlotManager(
        directory=plots_dir,
        **conf.plot
    )
    
    dataset = utils.get_class(conf.dataset_class)(index=args.index, subindex=args.subindex, **conf.datasets)

    return conf, checkpoint_manager, plot_manager, dataset


def run(conf, checkpoint_manager, plot_manager, dataset):
    
    np.random.seed(conf.training.rng_seed)
    key = jrnd.PRNGKey(conf.training.rng_seed)
    
    implicit_net = utils.get_class(conf.implicit_class)(**conf.network.implicit_net)
    
    velocity_net = utils.get_class(conf.velocity_class)(**conf.network.velocity_net)
    
    learning_rate_fn = partial(
        utils.step_learning_rate_decay,
        initial=conf.training.initial,
        interval=conf.training.interval, 
        factor=conf.training.factor)
    
    
    model = utils.get_class(conf.method)(T=conf.network.T, **conf.loss)

    key, = jrnd.split(key, 1)

    implicit_train_state, velocity_train_state = model.model_init(key, learning_rate_fn, implicit_net, velocity_net, conf)

    dptc_list = dataset.generate_pesudo_dptc(20000)

    try:
        step = checkpoint_manager.latest_step()
        start_epoch = step
        # target = {'model':implicit_train_state, 'index': 0, 'pair': [0,1]}
        target = {'model':(implicit_train_state, velocity_train_state), 'index': 0, 'subindex':1, 'pair': [0,1], 'upper':np.array([0.5,0.8,0.4]), 'lower': np.array([-0.5,-0.8,-0.4])}
        restored = checkpoint_manager.restore(step, items=target)
        implicit_train_state = restored['model'][0]
        velocity_train_state = restored['model'][1]
        
        index = restored['index']
        subindex=restored['subindex']
        
        internal_index = dataset.get_index([index, subindex])
        dptc_x, dptc_y = dataset.getitem(dptc_list, index=internal_index)
        
       

        
        
    except:
        index = args.index
        subindex = args.subindex
        start_epoch = 0
        internal_index = dataset.get_index(args.index, args.subindex)
        dptc_x, dptc_y = dataset.getitem(dptc_list)
        

    
    
    # pair = dataset.combinations[index]
    pair = [index, subindex]
    print(" train for {0} -----> {1}...............".format(pair[0], pair[1]))
        
        
    bounding_box = mesh_utils.get_bounding_box(jnp.concatenate((dptc_x.points, dptc_y.points)))
    prefix = dataset.mesh_paths[internal_index][:-5] + '_'
    
    plot_manager(lower=bounding_box[0], upper=bounding_box[1], vertex_size = len(dptc_x.verts), prefix=prefix)
    
    # save gt shapes
    plot_manager.save_points_ply(jnp.concatenate((dptc_x.verts, dptc_x.points)), normals=jnp.concatenate((dptc_x.verts_normals, dptc_x.points_normals)), output_file=prefix + '0_gt')
    plot_manager.save_points_ply(jnp.concatenate((dptc_y.verts, dptc_y.points)), normals=jnp.concatenate((dptc_y.verts_normals, dptc_y.points_normals)), output_file=prefix + '1_gt')

    
    # save check points and input shape
    checkpoint_info={
        'index': index,
        'subindex': subindex,
        'pair':pair,
        'upper':bounding_box[0],
        'lower':bounding_box[1],
    }
    
    save_args = orbax_utils.save_args_from_target({'model': (implicit_train_state, velocity_train_state) , **checkpoint_info})

    batch_metrics = {}
    for epoch in tqdm(range(start_epoch, conf.training.nepochs+1)):
        key, = jrnd.split(key, 1)

        # start_time = time.time()
        input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global = model.get_batch(key, batch_size=conf.datasets.batch_size, dptc_x=dptc_x, dptc_y=dptc_y)
        # print('sample batch {0} second...'.format(time.time()-start_time)) 

        if checkpoint_manager.should_save(epoch):
            save_checkpoint(checkpoint_manager, train_state=(implicit_train_state, velocity_train_state), checkpoint_info=checkpoint_info, save_args=save_args, batch_index=epoch)
            utils.save_csv(os.path.join(conf.expdir, 'loss.csv'), batch_metrics)
            checkpoint_manager.wait_until_finished()
            

        if plot_manager.should_save(epoch):
            implicit_fn = partial(implicit_train_state.apply_fn, implicit_train_state.params)
            velocity_fn = partial(velocity_train_state.apply_fn, velocity_train_state.params)
            model.visual(plot_manager, dptc_x, implicit_fn=implicit_fn, velocity_fn=velocity_fn, epoch=epoch)

        if conf.training.separate_train:
            if epoch < conf.training.v_warm_up:
                loss, metrics, implicit_train_state, velocity_train_state = model.train_step_frozen_sdf(input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, 1.0,implicit_state=implicit_train_state, velocity_state=velocity_train_state)
            else:
                loss, metrics, implicit_train_state, velocity_train_state = model.train_step_frozen_v(input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, 1.0,implicit_state=implicit_train_state, velocity_state=velocity_train_state)
        else:
            
            ratio = model.get_ratio(epoch, conf.training.v_warm_up, conf.training.full)
            if ratio[0] == 0.0:
                loss, metrics, implicit_train_state, velocity_train_state = model.train_step_frozen_v(input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, 1.0,implicit_state=implicit_train_state, velocity_state=velocity_train_state)
            elif ratio[1] == 0.0:
                loss, metrics, implicit_train_state, velocity_train_state = model.train_step_frozen_sdf(input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, 1.0,implicit_state=implicit_train_state, velocity_state=velocity_train_state)
            else:
                loss, metrics, implicit_train_state, velocity_train_state = model.train_step(input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio,implicit_state=implicit_train_state, velocity_state=velocity_train_state)
        

        if epoch % 100 == 0:
            print(
            '{0} [{1}] ({2}/{3}): loss {4} \n'.format(len(dataset), epoch, pair[0], pair[1], loss
            ))
            for key_name, value in metrics.items():
                print(f"{key_name}: {value}\t")
        
        for key_name, value in metrics.items():
            if key_name in batch_metrics:
                batch_metrics[key_name].append(value)
            else:
                batch_metrics[key_name] = [value]
            
        
    if conf.training.fine_tune:
        print('start fine tuning......')
        for epoch in tqdm(range(conf.training.nepochs+1, conf.training.end+1)):
            key, = jrnd.split(key, 1)
            input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global = model.get_batch(key, batch_size=conf.datasets.batch_size, dptc_x=dptc_x, dptc_y=dptc_y)
            
            loss, implicit_train_state, velocity_train_state = model.train_step_fine_tune_sdf(input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, 1.0,implicit_state=implicit_train_state, velocity_state=velocity_train_state)

            if epoch % 100 == 0:
                print(
                '{0} [{1}] ({2}/{3}): fine tuning loss {4} \n'.format(len(dataset), epoch, pair[0], pair[1], loss
                ))
            
            if checkpoint_manager.should_save(epoch):
                save_checkpoint(checkpoint_manager, train_state=(implicit_train_state, velocity_train_state), checkpoint_info=checkpoint_info, save_args=save_args, batch_index=epoch)
                utils.save_csv(os.path.join(conf.expdir, 'loss.csv'), batch_metrics)
                checkpoint_manager.wait_until_finished()


            if plot_manager.should_save(epoch):
                implicit_fn = partial(implicit_train_state.apply_fn, implicit_train_state.params)
                velocity_fn = partial(velocity_train_state.apply_fn, velocity_train_state.params)
                model.visual(plot_manager, dptc_x, implicit_fn=implicit_fn, velocity_fn=velocity_fn, epoch=epoch)
    
        
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./conf/faust.conf')
    parser.add_argument('--savedir', type=str, default='./exp/')
    parser.add_argument('--expname', type=str, default='fraust_r')
    parser.add_argument('--log', action='store_true', help="if log into wandb")
    parser.add_argument('--eval',action='store_true', help="if evaluate using higher resolution")
    parser.add_argument('--reset', action='store_true', help="if restart the experiment")
    parser.add_argument('--index', default=0, type=int, help="source index of the dataset")
    parser.add_argument('--subindex', default=1, type=int, help="target index of the dataset")
    
    args = parser.parse_args()
    
    conf, checkpoint_manager, plot_manager, dataset = setup(
        args
    )
    
    run(conf, checkpoint_manager, plot_manager, dataset)
    
