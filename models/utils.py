import jax 
import jax.numpy as jnp
from jax import vmap
from functools import partial
from datasets import sampler
from datasets.pointshape import DeformPointCloud
from flax.training.train_state import TrainState
import optax
import numpy as np

def soft_sign(x, eps=1e-12):
    n = jnp.sqrt(x**2+eps)
    return x / n

def sq_norm(a, *args, **kwargs):
    return (a ** 2).sum(*args, **kwargs)

def soft_norm(x, *arg, **kwargs):
    eps=1e-12
    """Use l2 for large values and squared l2 for small values to avoid grad=nan at x=0."""
    return eps * (jnp.sqrt(sq_norm(x, *arg, **kwargs) / eps ** 2 + 1) - 1)
    
def cosine_similarity(gt, df):

    scd = jnp.sum(df * gt, axis=1)

    scd = scd / jnp.linalg.norm(gt, axis=1)
    scd = scd / jnp.linalg.norm(df, axis=1)
    
    return 1-scd**2

def normal_loss(gt, df):

    # normalize
    # df_n = jnp.linalg.norm(df, axis=1)
    df_n = soft_norm(df, axis=1)
    df = df / df_n[:,None]

    # gt_n = jnp.linalg.norm(gt, axis=1)
    gt_n = soft_norm(df, axis=1)
    gt = gt / gt_n[:,None]

    loss = sq_norm(gt-df, axis=1)
    return loss.mean()
    

def any_nans(pytree):
    """Returns True if any leaf of the pytree contains a nan value."""
    return jnp.array(jax.tree_util.tree_flatten(jax.tree_map(lambda a: jnp.isnan(a).any(), pytree))[0]).any()

def safe_apply_grads(state, grads):
    nan_grads = any_nans(grads)
    state = jax.lax.cond(nan_grads, lambda: state, lambda: state.apply_gradients(grads=grads))
    return state, nan_grads

    
def get_gradient(f, params, points, t):
  df = jax.grad(f, argnums=1)(params, points, t)
  return df

def get_full_gradient(f, params, points, t):
    sdf, (df, dt) = jax.value_and_grad(f, argnums=(1,2))(params, points, t)
    return sdf, df, dt


def eikonal_loss(f, params, point, t):
    df = jax.jacobian(f, argnums=1)(params, point, t)
    eikonal_loss = (1-soft_norm(df))**2
    # eikonal_loss = (1-jnp.linalg.norm(df))**2
    return eikonal_loss

def get_jacobian(f, params, points, t=None):
    if not t is None:
        dv = jax.jacobian(f, argnums=1)(params, points, t)
    else:
        dv = jax.jacobian(f, argnums=1)(params, points)
    return dv


def get_hessian(f, params, points, t=None):
    if not t is None:
        Hv = jax.hessian(f, argnums=1)(params, points, t)
    else:
        Hv = jax.hessian(f, argnums=1)(params, points)
    return Hv


def divergence(f, params, points, t=None):
    if not t is None:
        dv = vmap(get_jacobian, in_axes=(None, None, 0, 0))(f, params, points, t)
    else:
        dv = vmap(get_jacobian, in_axes=(None, None, 0))(f, params, points)
    div_v = jnp.trace(dv, axis1=1, axis2=2)
    return jnp.abs(div_v)

def implicit_distance(f, points, t, distance_metric='l2'):
    sdf = f(points, t)
    if distance_metric == 'squared_l2':
        implicit_distance = (sdf ** 2).mean()
    elif distance_metric == 'l2':
        implicit_distance = jnp.abs(sdf).mean()
    else:
        raise ValueError(f'Unrecognized distance metric {distance_metric=}.')
    return implicit_distance

def sdf_loss(sdf, distance_metric='l2'):
    if distance_metric == 'squared_l2':
        implicit_distance = (sdf ** 2).mean()
    elif distance_metric == 'l2':
        implicit_distance = jnp.abs(sdf).mean()
    else:
        raise ValueError(f'Unrecognized distance metric {distance_metric=}.')
    return implicit_distance

def match_loss(pointx, pointy):
    loss = soft_norm(pointx-pointy, axis=1)**2
    # loss = (pointx-pointy)**2
    return loss.mean()



def normalize(df):
    # norm = jnp.linalg.norm(df, axis=-1)
    norm = soft_norm(df, axis=-1)
    n = df / norm[:,None]
    return n, norm


def R_term(df, dv):
    n, norm = normalize(df)
    n = n[:,:,None]
    R = jnp.sum(jnp.matmul(dv,n)* n, axis=1).squeeze() 
    return R


def get_implicit_surface_area_loss(f, points, t, alpha=100):
    return jnp.exp(-alpha * jnp.abs(f(points, t=t))).mean()



def model_init(key, learning_rate_fn, MLP, conf):
    d_in = conf.d_in
    if conf.timespace:
        d_in = d_in + 1
    if conf.feature_vector_size > 0:
        d_in = d_in + conf.feature_vector_size
    
    params = MLP.init(key, jnp.ones(d_in))

    trian_state = TrainState.create(
        apply_fn=MLP.apply,
        tx=optax.adam(learning_rate=learning_rate_fn),
        params=params
    )

    return trian_state



def weight_normalization(W, softplus_c):
    """
    Lipschitz weight normalization based on the L-infinity norm
    """
    absrowsum = jnp.sum(jnp.abs(W), axis=1)
    scale = jnp.minimum(1.0, softplus_c/absrowsum)
    return W * scale[:,None]

def softplus(x, beta=100):
    return jnp.logaddexp(0, beta * x) / beta

def safe_softplus(x, beta=100):
    # revert to linear function for large inputs, same as pytorch
    return jnp.where(x * beta > 20, x, softplus(x))

def get_lipschitz_loss(params):
    
    loss_lip = 1.0
    dims = len(params['params']) // 3
        
    for i in range(dims):
        # kernel = params['params'][f'kernel_{i}']
        # bias = params['params'][f'bias_{i}']
        c = params['params'][f'constant_{i}']
        loss_lip = loss_lip * safe_softplus(c)
    return loss_lip
        
    
def normalize_params(params):
    dims = len(params['params']) // 3
    
    params_final = params
    for i in range(dims):
        kernel = params['params'][f'kernel_{i}']
        bias = params['params'][f'bias_{i}']
        c = params['params'][f'constant_{i}']
        
        kernel = weight_normalization(kernel, safe_softplus(c))
        
        params_final['params'][f'kernel_{i}'] = kernel
        params_final['params'][f'bias_{i}'] = bias
        params_final['params'][f'constant_{i}'] = c  
    
    return params_final