import os
from glob import glob
import jax.numpy as jnp
import csv
from flax import serialization
import numpy as np
import jax


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def learning_rate_decay(step,
                        initial,
                        final=5e-6,
                        interval=5000,
                        lr_delay_steps=2000,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is initial when step=0 and final when step=interval, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  initial*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    initial: float, the initial learning rate.
    final: float, the final learning rate.
    interval: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * jnp.sin(
        0.5 * jnp.pi * jnp.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = jnp.clip(step / interval, 0, 1)
  log_lerp = jnp.exp(jnp.log(initial) * (1 - t) + jnp.log(final) * t)
  return delay_rate * log_lerp



def step_learning_rate_decay(epoch, initial=0.005, interval=2000, factor=0.5):
    return jnp.maximum(initial * (factor ** (epoch // interval)), 5.0e-6)



def save_csv(filename, data):
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerow(data.keys())
        w.writerows(zip(*data.values()))
    f.close()
    
    
def save_latent_vectors(path, step, params):
    file_name = os.path.join(path, 'latent_vectors_'+str(step))
    params_numpy = jax.tree_util.tree_map(lambda p: np.array(p), params.block_until_ready())
    np.save(file_name, params_numpy)

     
def save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index, save_latent=False):
    # save_args=ocp.args.StandardSave(save_args)
    if len(checkpoint_info) > 0:
      checkpoint_manager.save(
          step=batch_index,
          items={'model': train_state, **checkpoint_info},
          save_kwargs={'save_args': save_args},
          # args=ocp.args.StandardSave(save_args),
          force=True,
      )
    else:
       checkpoint_manager.save(
          step=batch_index,
          items={'model': train_state},
          save_kwargs={'save_args': save_args},
          # args=ocp.args.StandardSave(save_args),
          force=True,
      )
    
    if save_latent: #it is not a good saving function since it assume either velocity is the last element of train_states vectors or only have one train_state
      try:
        try:
          params = train_state.params['params']['Embed_0']['embedding']
        except:
          params = train_state[-1].params['params']['Embed_0']['embedding']
        path = checkpoint_manager.directory
        save_latent_vectors(path, batch_index, params)
      except:
        print("failed to save latent vectors.")
        pass
      
    

def check_best(metrics: list[dict], latest_metrics: dict, metric_name: str):
    if (metric_name is None) or (len(metrics) == 0):
        return True
    else:
        return latest_metrics[metric_name] < min([m[metric_name] for m in metrics])