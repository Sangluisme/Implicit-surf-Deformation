import jax
import flax
from flax import linen as nn
from jax import random, jit, vmap
from functools import partial
import jax.random as jrnd
import jax.numpy as jnp
import numpy as np
import optax
from models.modules import MLP, LIPMLP
from models.modules import posenc
from flax.training.train_state import TrainState
from datasets.pointshape import PointShape
from models.utils import *
from datasets import sampler
from datasets.pointshape import DeformPointCloud


@flax.struct.dataclass
class TargetDeform:
    alpha: float = 0.05
    reinitial: float = 0
    eikonal: float = 20  
    laplacian: float = 10
    divergence: float = 10
    level_set: float = 500
    normal: float = 100
    match: float = 200
    manifold: float = 100
    nonmanifold: float = 10
    onsurf: float=10
    surf_area: float=10
    streching: float = 5
    T: int= 8
    corr_num: float=1.0

    def get_ratio(self, epoch, start_epoch, nepoch):
        
        # return get_ratio(epoch, start_epoch, nepoch)
        end = start_epoch * 2

        if epoch < start_epoch:
            ratio_v = 1.0
            ratio_i = 0.0
        elif epoch < nepoch:
            ratio_i = np.min([1.0, (epoch - start_epoch) / (end - start_epoch)])
            ratio_v = 1.0
            
        else:
            ratio_v = 0.0
            ratio_i = 1.0
        
        return [ratio_v, ratio_i]   
    
    
    def model_init(self, key, learning_rate_fn, implicit_net, velocity_net, conf):
        key_i, key_v = jrnd.split(key, 2)
        implicit_train_state = model_init(key_i, learning_rate_fn, implicit_net, conf.network.implicit_net)
        velocity_train_state = model_init(key_v, learning_rate_fn, velocity_net, conf.network.velocity_net)
        return implicit_train_state, velocity_train_state
    
    def visual(self, plot_manager, dpct, implicit_fn, velocity_fn, epoch):

        points = jnp.concatenate((dpct.verts, dpct.points))
        color = plot_manager.get_color(points)
    # points = model_input['input_x']
        for time in range(self.T+1):
            t = time / self.T
            print('meshing time step {0}.....'.format(t))
            
            mesh = plot_manager.extract_mesh(implicit_fn, t)
            filename = plot_manager.prefix + 'epoch_' + str(epoch) + '_time_' + str(time) + '_mesh'
            plot_manager.save_ply(mesh, filename)

            
            filename = plot_manager.prefix + 'epoch_' + str(epoch) + '_time_' + str(time) + '_ptc'
            plot_manager.save_points_ply(points=points, normals=None, color=color, output_file=filename)

            points = plot_manager.visualize_velocity(points, velocity_fn)
            
    
    @partial(jit, static_argnames=("self", "batch_size"))
    def get_batch(self, key, batch_size, dptc_x: DeformPointCloud, dptc_y: DeformPointCloud):
        key_v, key_p1, key_p2, key_local, key_global = jrnd.split(key, 5)

        x, nx, indices = sampler.sample_array(key_v, batch_size, dptc_x.verts, dptc_x.verts_normals)
        y = dptc_y.verts[indices]
        ny = dptc_y.verts_normals[indices]

        sx, snx, indices1 = sampler.sample_array(key_p1, batch_size, dptc_x.points, dptc_x.points_normals)
        # sy = dptc_y.points[indices]
        # sny = dptc_y.points_normals[indices]
        sy, sny, indices2 = sampler.sample_array(key_p2, batch_size, dptc_y.points, dptc_y.points_normals)

        sample_local_x = sampler.generate_local_samples(key_local, sx, dptc_x.local_sigma[indices1])

        sample_local_y = sampler.generate_local_samples(key_local, sy, dptc_y.local_sigma[indices2])

        sample_global = sampler.generate_global_samples(key_global, dptc_x.lower, dptc_x.upper, batch_size//4, 3)

        input_x = jnp.concatenate((x, sx))
        input_nx = jnp.concatenate((nx, snx))
        input_y = jnp.concatenate((y, sy)) 
        input_ny = jnp.concatenate((ny, sny))
        sample_local_x = jnp.concatenate((sample_local_x, sample_global))
        sample_local_y = jnp.concatenate((sample_local_y, sample_global))

        return input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global
    

    def get_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio):
        
        batch_size = input_x.shape[0] // 2
        
        points = input_x

        loss, loss_e, loss_ef, loss_sdf, loss_n, loss_surf = 0., 0., 0., 0., 0.,0.
        loss_lap, loss_div, loss_lse = 0., 0., 0.
        loss_area = 0.
        
        ratio_v, ratio_i = ratio

        # fit first t==0:
        t = jnp.tile(0., (input_x.shape[0],1))
        sdf_i, df_i, dt_i = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_x, t)

        v = partial(fv_fn, params_v)(points)
        dv = vmap(get_jacobian, in_axes=(None, None, 0))(fv_fn, params_v, points)

        sdf_s, df_s, dt_s = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, sample_local_x, jnp.tile(0.,(sample_local_x.shape[0],1)))

        if (self.manifold > 0):
            loss_sdf += sdf_loss(sdf_i, distance_metric='l2')
        if (self.normal > 0):
            loss_n += normal_loss(input_nx, df_i)
        # if (self.eikonal > 0):
        #     loss_e += ((1-soft_norm(df_i, axis=-1))**2).mean()
        if (self.nonmanifold > 0):
            loss_ef += ((1-soft_norm(df_s, axis=-1))**2).mean()
        if self.surf_area > 0:
            loss_area += get_implicit_surface_area_loss(partial(f_fn, params), sample_global, jnp.tile(0.,(sample_global.shape[0],1)))
        # level_set
        if self.level_set > 0:
            if self.reinitial > 0:
                R = self.eikonal * R_term(df_i, dv)
                loss_lse += sdf_loss(dt_i + jnp.sum(df_i * v, axis=-1) + sdf_i*R,'squared_l2')
            else:
                loss_lse += sdf_loss(dt_i + jnp.sum(df_i * v, axis=-1),'squared_l2')
                loss_e += ((1-soft_norm(df_i, axis=-1))**2).mean()

        if self.laplacian > 0.0:
            Hv = vmap(get_hessian, in_axes=(None, None, 0))(fv_fn, params_v, points)
            laplacian = jnp.trace(Hv, axis1=2, axis2=3)
            # loss_lap += jnp.linalg.norm(v - self.alpha * laplacian, axis=-1).mean()
            loss_lap += sq_norm(v - self.alpha * laplacian, axis=-1).mean()
                       
        if self.divergence > 0.0:
            dv = vmap(get_jacobian, in_axes=(None, None, 0))(fv_fn, params_v, points)
            div_v = jnp.trace(dv, axis1=1, axis2=2)
            loss_div += sdf_loss(div_v,'squared_l2')
            

        # move points
        points = points + v
        
        for time in range(1, self.T+1):

            t = jnp.tile(time/self.T, (points.shape[0],1))


            sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, points, t)
            
            v = partial(fv_fn, params_v)(points)

            if (self.eikonal > 0) and (self.reinitial == 0):
                # loss_e += ((1-jnp.linalg.norm(df, axis=-1))**2).mean()
                loss_e += ((1-soft_norm(df, axis=-1))**2).mean()
            
            if (self.nonmanifold > 0) and (self.reinitial == 0):
                loss_ef += vmap(eikonal_loss, in_axes=(None,None,0,0))(f_fn, params, sample_local_x, jnp.tile(time/self.T,(sample_local_x.shape[0],1))).mean()
                v_sample = partial(fv_fn, params_v)(sample_local_x)
                sample_local_x = sample_local_x + v_sample
            if self.surf_area > 0:
                loss_area += get_implicit_surface_area_loss(partial(f_fn, params), sample_global, jnp.tile(time/self.T,(sample_global.shape[0],1)))

            if self.level_set > 0:
                if self.reinitial > 0:
                    R = (self.eikonal/self.level_set) * R_term(df_i, dv)
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1) + sdf*R,'squared_l2')
                else:
                    loss_lse += sdf_loss(dt_i + jnp.sum(df * v, axis=-1), 'squared_l2')

            if (self.onsurf > 0) and (time < self.T):
                loss_surf += sdf_loss(sdf, distance_metric='l2')
            
            if (self.manifold > 0) and (time == self.T):
                loss_sdf += sdf_loss(sdf, 'l2')
            
            if (self.normal > 0) and (time==self.T):
                loss_n += normal_loss(df, input_ny)

            if (self.laplacian > 0.0) and (time < self.T):
                Hv = vmap(get_hessian, in_axes=(None, None, 0))(fv_fn, params_v, points)
                laplacian = jnp.trace(Hv, axis1=2, axis2=3)
                # loss_lap += jnp.linalg.norm(v - self.alpha * laplacian, axis=-1).mean()
                loss_lap += sq_norm(v - self.alpha * laplacian, axis=-1).mean()                   

            if (self.divergence > 0.0) and (time < self.T):
                dv = vmap(get_jacobian, in_axes=(None, None, 0))(fv_fn, params_v, points)
                div_v = jnp.trace(dv, axis1=1, axis2=2)
                loss_div += sdf_loss(div_v,'squared_l2')
            
            if (self.match > 0) and (time==self.T):
                loss_match = match_loss(points[:batch_size,:], input_y[:batch_size,:])
            
            # move points
            points = points + v
        
        # fit second
        sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_y, t)


        metrics = {}
        if self.eikonal > 0:
            loss_e = loss_e / self.T
            loss += self.eikonal * ratio_i * loss_e
            metrics.update({'eikonal': loss_e})

        if self.nonmanifold > 0:
            loss_ef = loss_ef / self.T
            loss += self.nonmanifold * ratio_i * loss_ef
            metrics.update({'nonmanifold': loss_ef})

        if self.divergence > 0:
            loss_div = loss_div / self.T
            loss += self.divergence * loss_div
            metrics.update({'divergence': loss_div})

        if self.laplacian > 0:
            loss_lap = loss_lap / self.T
            loss += self.laplacian * loss_lap
            metrics.update({'laplacian': loss_lap})
        
        if self.level_set > 0:
            loss_lse = loss_lse / self.T
            loss += self.level_set * ratio_i * loss_lse
            metrics.update({'level_set': loss_lse})
        
        if self.manifold > 0:
            loss_sdf += sdf_loss(sdf,'l2')
            loss += self.manifold * ratio_i * loss_sdf  / 2
            metrics.update({'manifold': loss_sdf})
        
        if self.normal > 0:
            loss_n += normal_loss(df, input_ny)
            loss += self.normal * ratio_i * loss_n / 3
            metrics.update({'normal': loss_n / 3})
        
        if self.match > 0:
            loss += self.match * loss_match
            metrics.update({'match': loss_match})
        
        if self.onsurf > 0:
            loss += self.onsurf * ratio_i * loss_surf / (self.T-1)
            metrics.update({'onsurf': loss_surf/ (self.T-1)})
        
        if self.surf_area > 0:
            loss += self.surf_area * ratio_i * loss_area / self.T
            metrics.update({'surf_area': loss_area / self.T})


        metrics.update({'loss':loss})
        return loss, metrics
    

    def get_velocity_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio):

        batch_size = input_x.shape[0] // 2
        
        points = input_x

       
        loss_lap, loss_div, loss = 0., 0., 0.
       
        for time in range(self.T):

            t = jnp.tile(time/self.T, (points.shape[0],1))

            v = partial(fv_fn, params_v)(points)

            if (self.laplacian > 0.0):
                Hv = vmap(get_hessian, in_axes=(None, None, 0))(fv_fn, params_v, points)
                laplacian = jnp.trace(Hv, axis1=2, axis2=3)
                # loss_lap += jnp.linalg.norm(v - self.alpha * laplacian, axis=-1).mean()
                loss_lap += sq_norm(v - self.alpha * laplacian, axis=-1).mean()                   

            if (self.divergence > 0.0):
                dv = vmap(get_jacobian, in_axes=(None, None, 0))(fv_fn, params_v, points)
                div_v = jnp.trace(dv, axis1=1, axis2=2)
                loss_div += sdf_loss(div_v,'squared_l2')
            
             # move points
            points = points + v
        
        #matching
        loss_match = match_loss(points[:batch_size,:], input_y[:batch_size,:])

        metrics = {}
        if self.divergence > 0:
            loss_div = loss_div / self.T
            loss += self.divergence * loss_div
            metrics.update({'divergence': loss_div})

        if self.laplacian > 0:
            loss_lap = loss_lap / self.T
            loss += self.laplacian * loss_lap
            metrics.update({'laplacian': loss_lap})
        
        if self.match > 0:
            loss += self.match * loss_match
            metrics.update({'match': loss_match})
        
        metrics.update({'loss':loss})
        return loss, metrics
    

    def get_sdf_loss(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio):
        
        batch_size = input_x.shape[0] // 2
        
        points = input_x

        loss, loss_lse, loss_e, loss_ef, loss_sdf, loss_n, loss_surf = 0., 0., 0., 0., 0., 0., 0.0
        loss_area = 0.0

        # fit the first
        t = jnp.tile(0., (input_x.shape[0],1))
        sdf_i, df_i, dt_i = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_x, t)

        v = partial(fv_fn, params_v)(points)

        if (self.manifold > 0):
            loss_sdf += sdf_loss(sdf_i, distance_metric='l2')
        if (self.normal > 0):
            loss_n += normal_loss(input_nx, df_i)
        # if (self.eikonal > 0):
        #     # loss_e += ((1-jnp.linalg.norm(df_i, axis=-1))**2).mean()
        #     loss_e += ((1-soft_norm(df_i, axis=-1))**2).mean()
        if (self.nonmanifold > 0):
            sdf_s, df_s, dt_s = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, sample_local_x, jnp.tile(0.,(sample_local_x.shape[0],1)))
            v_sample = partial(fv_fn, params_v)(sample_local_x)
            sample_local_x = sample_local_x + v_sample
            # loss_ef += ((1-jnp.linalg.norm(df_s, axis=-1))**2).mean()
            loss_ef += ((1-soft_norm(df_s, axis=-1))**2).mean()
        if self.surf_area > 0:
            loss_area += get_implicit_surface_area_loss(partial(f_fn, params), sample_global, jnp.tile(0.,(sample_global.shape[0],1)))
            
        if self.level_set > 0:
            if self.reinitial > 0:
                dv = vmap(get_jacobian, in_axes=(None, None, 0))(fv_fn, params_v, points)
                R = (self.eikonal/self.level_set) * R_term(df_i, dv)
                loss_lse += sdf_loss(dt_i + jnp.sum(df_i * v, axis=-1) + sdf_i*R,'squared_l2')
            else:
                loss_lse += sdf_loss(dt_i + jnp.sum(df_i * v, axis=-1),'squared_l2')
                loss_e += ((1-soft_norm(df_i, axis=-1))**2).mean()

        # move points
        points = points + v

        for time in range(1, self.T+1):

            t = jnp.tile(time/self.T, (points.shape[0],1))


            sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, points, t)
            
            v = partial(fv_fn, params_v)(points)
            dv = vmap(get_jacobian, in_axes=(None, None, 0))(fv_fn, params_v, points)

            if (self.eikonal > 0) and (self.reinitial == 0):
                loss_e += ((1-soft_norm(df, axis=-1))**2).mean()
            
            if (self.nonmanifold > 0) and (self.reinitial == 0):
                loss_ef += vmap(eikonal_loss, in_axes=(None,None,0,0))(f_fn, params, sample_local_x, jnp.tile(time/self.T,(sample_local_x.shape[0],1))).mean()
                v_sample = partial(fv_fn, params_v)(sample_local_x)
                sample_local_x = sample_local_x + v_sample
                
            if self.surf_area > 0:
                loss_area += get_implicit_surface_area_loss(partial(f_fn, params), sample_global, jnp.tile(time/self.T,(sample_global.shape[0],1)))
                
            if self.level_set > 0:
                if self.reinitial > 0:
                    R = (self.eikonal/self.level_set) * R_term(df, dv)
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1) + sdf*R,'squared_l2')
                else:
                    loss_lse += sdf_loss(dt + jnp.sum(df * v, axis=-1), 'squared_l2')

            if (self.onsurf > 0) and (time < self.T):
                loss_surf += sdf_loss(sdf, distance_metric='l2')
            
            if (self.manifold > 0) and (time == self.T):
                loss_sdf += sdf_loss(sdf, 'l2')
            
            points = points + v
        
        # fit the second
        sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_y, t)


        metrics = {}
        if self.eikonal > 0:
            loss_e = loss_e / self.T
            loss += self.eikonal * loss_e
            metrics.update({'eikonal': loss_e})

        if self.nonmanifold > 0:
            loss_ef = loss_ef / self.T
            loss += self.nonmanifold * loss_ef
            metrics.update({'nonmanifold': loss_ef})

        if self.level_set > 0:
            loss_lse = loss_lse / self.T
            loss += self.level_set * loss_lse
            metrics.update({'level_set': loss_lse})
        
        if self.manifold > 0:
            loss_sdf += sdf_loss(sdf,'l2')
            loss += self.manifold * loss_sdf  / 2
            metrics.update({'manifold': loss_sdf})
        
        if self.normal > 0:
            loss_n += normal_loss(df, input_ny)
            loss += self.normal * loss_n / 3
            metrics.update({'normal': loss_n / 3})
        
        if self.onsurf > 0:
            loss += self.onsurf * loss_surf / (self.T-1)
            metrics.update({'onsurf': loss_surf/ (self.T-1)})
        
        if self.surf_area > 0:
            loss += self.surf_area * ratio * loss_area / self.T
            metrics.update({'surf_area': loss_area / self.T})

        metrics.update({'loss':loss})
        return loss, metrics

    def get_sdf_separate(self, f_fn, params, fv_fn, params_v, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio, time_step):
        
        # fine tunning for intermiddate steps
        
        points = input_x
        
        loss, loss_e, loss_ef, loss_area, loss_n, loss_lse, loss_surf = 0., 0., 0., 0., 0.,0., 0.
        # fit the first
        t = jnp.tile(time_step/self.T, (input_x.shape[0],1))
        
        if (time_step < self.T):
            sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_x, t)
        elif (time_step == self.T):
            sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, input_y, t)

        v = partial(fv_fn, params_v)(points)

     
        if (self.normal > 0):
            if time_step == 0:
                loss_n += normal_loss(input_nx, df)
            elif time_step == self.T:
                loss_n += normal_loss(input_ny, df)
            loss += self.normal * loss_n
        
        if (self.eikonal > 0):
            # loss_e += ((1-jnp.linalg.norm(df_i, axis=-1))**2).mean()
            loss_e += ((1-soft_norm(df, axis=-1))**2).mean()
            loss += self.eikonal * loss_e
            
        if (self.nonmanifold > 0):
            sdf_s, df_s, dt_s = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, sample_local_x, jnp.tile(0.,(sample_local_x.shape[0],1)))
            v_sample = partial(fv_fn, params_v)(sample_local_x)
            sample_local_x = sample_local_x + v_sample
            # loss_ef += ((1-jnp.linalg.norm(df_s, axis=-1))**2).mean()
            loss_ef += ((1-soft_norm(df_s, axis=-1))**2).mean()
            loss += self.nonmanifold * loss_ef
        
        if self.surf_area > 0:
            loss_area += get_implicit_surface_area_loss(partial(f_fn, params), sample_global, jnp.tile(0.,(sample_global.shape[0],1)))
            loss += self.surf_area * loss_area
        
        
        loss_surf += sdf_loss(sdf, distance_metric='l2')
        loss += 3e3 * loss_surf
        
        points = points + v
        
        return loss, (points, sample_local_x)
        

    
    @partial(jit, static_argnames=("self"))
    def train_step(self, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        (loss, metrics), (grads_f, grads_v) = jax.value_and_grad(self.get_loss, argnums=(1,3), has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio)
        
        implicit_state, nan_grads = safe_apply_grads(implicit_state, grads_f)
        
        velocity_state, nan_grads = safe_apply_grads(velocity_state, grads_v)

        return loss, metrics, implicit_state, velocity_state
    
    @partial(jit, static_argnames=("self"))
    def train_step_frozen_v(self, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        (loss, metrics), grads_f = jax.value_and_grad(self.get_sdf_loss, argnums=1, has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio)
        
        implicit_state, nan_grads = safe_apply_grads(implicit_state, grads_f)
        
        return loss, metrics, implicit_state, velocity_state

    @partial(jit, static_argnames=("self"))
    def train_step_frozen_sdf(self, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        (loss, metrics), grads_v = jax.value_and_grad(self.get_velocity_loss, argnums=3, has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio)
        
        velocity_state, nan_grads = safe_apply_grads(velocity_state, grads_v)
        
        return loss, metrics, implicit_state, velocity_state
    
    @partial(jit, static_argnames=("self"))
    def train_step_fine_tune_sdf(self, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio, implicit_state: TrainState, velocity_state: TrainState):
        
        for time_step in range(self.T+1):
            (loss, (input_x, sample_local_x)), grads_f = jax.value_and_grad(self.get_sdf_separate, argnums=1, has_aux=True)(implicit_state.apply_fn, implicit_state.params, velocity_state.apply_fn, velocity_state.params, input_x, input_nx, input_y, input_ny, sample_local_x, sample_local_y, sample_global, ratio, time_step)
            implicit_state, nan_grads = safe_apply_grads(implicit_state, grads_f)
    
        return loss, implicit_state, velocity_state




#####################################################################################
@flax.struct.dataclass
class FitImplicitLip:
    manifold: float = 100
    nonmanifold: float = 10
    eikonal: float = 20  
    normal: float = 50
    lipschitz: float = 1e-10

    
    
    def model_init(self, key, learning_rate_fn, implicit_net, conf):
        implicit_train_state = model_init(key, learning_rate_fn, implicit_net, conf)
        return implicit_train_state
    
    
    def visual(self, plot_manager, implicit_train_state, epoch, time_steps=[0,1]):
        
        params_final = normalize_params(implicit_train_state.params)
        
        implicit_fn = partial(implicit_train_state.apply_fn, params_final)
        
        for t in time_steps:
    
            mesh = plot_manager.extract_mesh(implicit_fn, t)
            filename = plot_manager.prefix + 'epoch_' + str(epoch) + '_time_' + str(t).zfill(2) + '_mesh'
            plot_manager.save_ply(mesh, filename)
            
        
    
    @partial(jit, static_argnames=("self", "batch_size"))
    def get_batch(self, key, batch_size, dptc_x: DeformPointCloud, dptc_y: DeformPointCloud):
        key_v, key_p1, key_p2, key_local, key_global = jrnd.split(key, 5)

        sx, snx, indices1 = sampler.sample_array(key_p1, batch_size, dptc_x.points, dptc_x.points_normals)
        # sy = dptc_y.points[indices]
        # sny = dptc_y.points_normals[indices]
        sy, sny, indices2 = sampler.sample_array(key_p2, batch_size, dptc_y.points, dptc_y.points_normals)
        
        sample_local_x = sampler.generate_local_samples(key_local, sx, dptc_x.local_sigma[indices1])

        sample_local_y = sampler.generate_local_samples(key_local, sy, dptc_y.local_sigma[indices2])

        sample_global = sampler.generate_global_samples(key_global, dptc_x.lower, dptc_x.upper, batch_size//4, 3)
        
        return sx, snx, sy, sny, sample_local_x, sample_local_y, sample_global

    
    
    def get_loss(self, f_fn, params, sx, snx, sy, sny, sample_local_x, sample_local_y, sample_global):
        
        # fit first
        t = jnp.tile(0., (sx.shape[0],1))
        loss_sdf = implicit_distance(partial(f_fn, params), sx, t, distance_metric='l2')
        sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, sx, t)
        
        df = jnp.nan_to_num(df, copy=False, nan=0.001)
        
        loss_n = normal_loss(snx, df).mean()
        # loss_ef = (1-soft_norm(df))**2
        loss_ef = (1-soft_norm(df, axis=-1))**2
        loss_ee = vmap(eikonal_loss, in_axes=(None,None,0,0))(f_fn, params, sample_local_x, t)
        loss_e = self.eikonal*loss_ef.mean() + self.nonmanifold*loss_ee.mean()


        # fit second
        t = jnp.tile(1., (sx.shape[0],1))
        loss_sdf += implicit_distance(partial(f_fn, params), sy, t, distance_metric='l2')
        sdf, df, dt = vmap(get_full_gradient, in_axes=(None, None, 0, 0))(f_fn, params, sy, t)
        
        df = jnp.nan_to_num(df, copy=False, nan=0.001)
        
        
        loss_n += normal_loss(sny, df).mean()
        # loss_ef = (1-soft_norm(df))**2
        loss_ef = (1-soft_norm(df, axis=-1))**2
        loss_ee = vmap(eikonal_loss, in_axes=(None,None,0,0))(f_fn, params, sample_local_y, t)
        loss_e += self.eikonal*loss_ef.mean() + self.nonmanifold*loss_ee.mean()

        
        # lipschitz loss
        loss_lip = get_lipschitz_loss(params)
        
        loss = self.manifold * loss_sdf + loss_e
   
        if self.normal > 0:
            loss += self.normal * loss_n
        if self.lipschitz > 0:
            loss += self.lipschitz * loss_lip
    

        # loss_sdf = self.manifold*loss_sdf
        # loss_n = self.normal*loss_n
        # loss_lip = self.lipschitz * loss_lip

        metrics = {
            'loss': loss,
            'eikonal':loss_e,
            'normal':loss_n,
            'manifold':loss_sdf,
            'lipschitz': loss_lip
        }

        return loss, metrics
    

    @partial(jit, static_argnames=("self"))
    def train_step(self, sx, snx, sy, sny, sample_local_x, sample_local_y, sample_global, implicit_state: TrainState):
        (loss, metrics), grads_f = jax.value_and_grad(self.get_loss, argnums=1, has_aux=True)(implicit_state.apply_fn, implicit_state.params, sx, snx, sy, sny, sample_local_x, sample_local_y, sample_global)

        implicit_state, nan_grads = safe_apply_grads(implicit_state, grads_f)

        return loss, metrics, implicit_state