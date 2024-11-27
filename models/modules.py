import sys
import jax
import jax.numpy as jnp
import jax.random as jrnd
from typing import Tuple, List
from dataclasses import dataclass, field

import enum

from flax import linen as nn
from flax.linen import relu, elu, linear


import numpy as np
import numpy.random as random

class ActivationFunction(enum.Enum):
    RELU = enum.auto()
    ELU = enum.auto()
    SIN = enum.auto()
    SOFTPLUS = enum.auto()
    
def get_activation_function(activation_function: ActivationFunction):
    return {
        ActivationFunction.RELU: relu,
        ActivationFunction.ELU: elu,
        ActivationFunction.SIN: jnp.sin,
        ActivationFunction.SOFTPLUS: safe_softplus,
    }[activation_function]

def softplus(x, beta=100):
    return jnp.logaddexp(0, beta * x) / beta

def safe_softplus(x, beta=100):
    # revert to linear function for large inputs, same as pytorch
    return jnp.where(x * beta > 20, x, softplus(x))

def non_zero_mean(key, shape, dtype=jnp.float32):
    normal_random_values = jrnd.normal(key, shape, dtype=dtype)
    mu = jnp.sqrt(jnp.pi) / jnp.sqrt(shape[0])
    return  mu + 0.00001 * normal_random_values

def zero_mean_multires(key, shape, dtype=jnp.float32):
    if shape[0] <= 3:
        normal_random_values = jrnd.normal(key, shape, dtype=dtype)
        sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
        init_weight = sigma * normal_random_values
    if shape[0] > 3:
        normal_random_values = jrnd.normal(key, (3, shape[1]), dtype=dtype)
        sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
        constant = jnp.zeros((shape[0]-3,shape[1]), dtype=dtype)
        weights = jnp.concatenate((normal_random_values, constant), axis=0)
        init_weight = sigma * weights
    return init_weight

def zero_mean(key, shape, dtype=jnp.float32):
    normal_random_values = jrnd.normal(key, shape, dtype=dtype)
    sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
    init_weight = sigma * normal_random_values
    return init_weight


class MLP(nn.Module):
    d_in: int=3
    d_out: int=1
    dims: List[int]=field(default_factory=list)
    skip_layers: Tuple[int,...]=(4,)
    # activation: ActivationFunction = ActivationFunction.SOFTPLUS
    activation: List[str] = field(default_factory=list)
    geometry_init: bool = False
    init_radius: float=1.0
    multires: int=0
    feature_vector_size: int=0
    timespace: bool=False
    
    @nn.compact
    def __call__(self, x, t=None, condition=None):
        
        activation = list(map({
            'softplus': safe_softplus,
            'relu': relu,
            'sin':jnp.sin,
            'elu': elu
        }.get, self.activation))[0]
        
        if t is not None:
            x = jnp.concatenate([x, t], axis=-1)

        # positional encoding
        if self.multires > 0:
            x = posenc(x, min_deg=0, max_deg=self.multires, invertable=True)

        if condition is not None:
            condition = jnp.tile(condition[None, :], (x.shape[0], 1))
            x = jnp.concatenate([x, condition], axis=-1)

        input_x = x
        dims = [x.shape[-1]] + [*self.dims] + [self.d_out]

        for i in range(len(dims)-2):
            
            out_dim = dims[i+1]
            
            if self.geometry_init:
                kernel_init = zero_mean
                if i == 0:
                    kernel_init = zero_mean_multires
            else:
                kernel_init = linear.default_kernel_init
                
            if i+1 in self.skip_layers:
                out_dim = dims[i+1] - dims[0]
                x = jnp.concatenate([x ,input_x], axis=-1) / jnp.sqrt(2)
            
            x = nn.Dense(features=out_dim, name=f'dense_{i}', kernel_init=kernel_init)(x)
            x = activation(x)
        
        # last layer
        kernel_init_final = non_zero_mean if self.geometry_init else linear.default_kernel_init
        bias_init_final = jax.nn.initializers.constant(-self.init_radius if self.geometry_init else 0.)
        x = nn.Dense(features=self.d_out, name=f'dense_{len(dims)-1}', kernel_init=kernel_init_final, bias_init=bias_init_final)(x)
    
        return x.squeeze()




###########################################################
def kernel_max(key, kernel):
    return jnp.max(jnp.sum(jnp.abs(kernel), axis=1))


class LIPMLP(nn.Module):
    d_in: int=3
    d_out: int=1
    dims: List[int]=field(default_factory=list)
    skip_layers: Tuple[int,...]=(4,)
    # activation: ActivationFunction = ActivationFunction.SOFTPLUS
    activation: str='softplus'
    geometry_init: bool = False
    init_radius: float=1.0
    multires: int=0
    feature_vector_size: int=0
    timespace: bool=False

        
    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c/absrowsum)
        return W * scale[:,None]
    
    @nn.compact
    def __call__(self, x, t=None):
        
        if t is not None:
            x = jnp.concatenate([x, t], axis=-1)
                
        dims = [x.shape[-1]] + [*self.dims] + [self.d_out]
        
        for i in range(len(dims)-2):
            out_dim = dims[i+1]
            kernel = self.param(f'kernel_{i}', zero_mean, (dims[i], out_dim))
            bias = self.param(f'bias_{i}', nn.initializers.zeros, (out_dim,))
            c = self.param(f'constant_{i}', kernel_max, kernel)
            
            kernel = self.weight_normalization(kernel, safe_softplus(c))
            x = safe_softplus(jnp.dot(x, kernel) + bias)
        
        # last layer
        kernel = self.param(f'kernel_{len(dims)-2}', zero_mean, (x.shape[-1], self.d_out))
        bias = self.param(f'bias_{len(dims)-2}', nn.initializers.zeros, (self.d_out,))
        c = self.param(f'constant_{len(dims)-2}', kernel_max, kernel)
        out = jnp.dot(x, kernel) + bias
        
        return out.squeeze()
        
    def get_lipschitz_loss(self, params):
        
        loss_lip = 1.0
        dims = [*self.dims] + [self.d_out]
            
        for i in range(len(dims)-1):
            # kernel = params['params'][f'kernel_{i}']
            # bias = params['params'][f'bias_{i}']
            c = params['params'][f'constant_{i}']
            loss_lip = loss_lip * jax.nn.softplus(c)
        return loss_lip
            
    
    def normalize_params(self, params):
        dims = [*self.dims] + [self.d_out]
        
        params_final = params
        for i in range(len(dims)-1):
            kernel = params['params'][f'kernel_{i}']
            bias = params['params'][f'bias_{i}']
            c = params['params'][f'constant_{i}']
            
            kernel = self.weight_normalization(kernel, jax.nn.softplus(c))
            
            params_final['params'][f'kernel_{i}'] = kernel
            params_final['params'][f'bias_{i}'] = bias
            params_final['params'][f'constant_{i}'] = c  
        
        return params_final



def posenc(x, min_deg, max_deg, legacy_posenc_order=False, invertable=False):
    if min_deg == max_deg:
        return x
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    if legacy_posenc_order:
        xb = x[Ellipsis, None, :] * scales[:,None]
        four_feat = jnp.reshape(jnp.sin(jnp.stack([xb, xb+0.5*jnp.pi], -2)), list(x.reshape[-1])+[-1])
    else:
        xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
        four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    encoded = jnp.concatenate([x] + [four_feat], axis=-1)

    if invertable:
        coeff = 1 / jnp.sqrt(2*max_deg+1)
        y = jnp.ones_like(x)
        yb = jnp.reshape((y[Ellipsis, None, :] * scales[:, None]),
                     list(y.shape[:-1]) + [-1])
        yb_norm = jnp.concatenate([yb, yb],axis=-1)
        yb_norm = jnp.concatenate([y] + [yb_norm], axis=-1)
        encoded = coeff * encoded / yb_norm 
    
    return encoded
