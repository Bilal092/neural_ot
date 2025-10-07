
import jax
import flax
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
import math

def get_model_fn(model, params):
    def model_fn(t,x):
        return model.apply(params, t, x)
    return model_fn

def sample_t(u0, n, t0=0.0, t1=1.0):
    u = (u0 + math.sqrt(2) * jnp.arange(n + 1)) % 1  # Generate n+1 samples and apply modulo 1
    u = u.reshape([-1, 1])                           # Reshape u to a column vector
    return u[:-1] * (t1 - t0) + t0, u[-1]

def get_loss_s(config, model_s, model_q):
    c = config.c 
    wgf_steps = config.train.wgf_steps
    wgf_step_size = config.train.wgf_step_size
    
    def loss_s(params_s, params_q, data_batch, time_batch, key):
        s = get_model_fn(model_s, params_s)
        dsdtdx_fn = jax.grad(lambda t,x: s(t,x).sum(), argnums=[0,1])
        acceleration_fn = jax.grad(lambda t, x: potential(t, x).sum(), argnums=1)
        # print("time shape", time_batch.shape)
        # print("data_batch shape", data_batch[0].shape)
        
        def potential(t, x):    
          dsdt, dsdx = dsdtdx_fn(t, x)
          return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True)
            
        # def wgf_step_func(carry, _):
        #     x_t, t, update = carry
        #     update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
        #     x_t = x_t + wgf_step_size * jnp.clip(update, -1, 1)
        #     carry = (x_t, t, update)
        #     return carry, _ 
            
        t=time_batch
        # t_0, t_1 = jnp.zeros([len(t), 1]),  jnp.ones([len(t), 1])
        t_0, t_1 = jnp.zeros_like(t), jnp.ones_like(t)
        x_0, x_1, x_1p =  data_batch[0], data_batch[1], data_batch[2]
        x_t = model_q.apply(params_q, t, x_0, x_1p)
  
        for wgf_iter in range(0, wgf_steps):
            update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
            x_t = x_t + wgf_step_size * t * jnp.clip(update, -1, 1)
           
        x_t = jax.lax.stop_gradient(x_t)
        
        loss = c*jax.nn.relu(-s(t_1, x_1)).mean() + s(t_0, x_0).mean()
        loss += potential(t, x_t).mean()
        
        update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
        
        loss = loss.mean()
        
        return loss, jnp.sqrt((update**2).sum(1)).mean()
    return loss_s

def get_loss_q(config, model_s, model_q):
    c = config.c 
    wgf_steps = config.train.wgf_steps
    wgf_step_size = config.train.wgf_step_size
    
    def loss_q(params_s, params_q, data_batch, time_batch, key):
        s_stopped = get_model_fn(model_s, jax.lax.stop_gradient(params_s))
        dsdtdx_fn_stopped = jax.grad(lambda t,x: s_stopped(t,x).sum(), argnums=[0,1])
        
        def potential_stopped(t, x):    
          dsdt, dsdx = dsdtdx_fn_stopped(t, x)
          return dsdt + 0.5*(dsdx**2).sum(axis=1, keepdims=True)
        acceleration_fn = jax.grad(lambda t, x: potential_stopped(t, x).sum(), argnums=1)
        
        t=time_batch
        # t_0, t_1 = jnp.zeros([len(t), 1]),  jnp.ones([len(t), 1])
        t_0, t_1 = jnp.zeros_like(t), jnp.ones_like(t)
        x_0, x_1p =  data_batch[0], data_batch[2]
        x_t = model_q.apply(params_q, t, x_0, x_1p)
        # print(x_t.shape, 'x_t.shape', flush=True)
        update = jax.lax.stop_gradient(acceleration_fn(t, x_t))
        loss = - potential_stopped(t,x_t).mean()
        return loss,  jnp.sqrt((update**2).sum(1)).mean()
    return loss_q

def get_step_fn_s(optimizer_s, optimizer_q, loss_fn_s):
    def step_fn_s(carry_state, batch):
        (key, state_s, state_q) = carry_state
        # jax.debug.print("{}", key)
        key, step_key = jax.random.split(key)
        grad_fn_s = jax.value_and_grad(loss_fn_s, argnums=[0], has_aux=True)
        
        u0 = jax.random.uniform(key, 1)
        t, _ = sample_t(u0, batch[0].shape[0])
        
        # t = random.uniform(key, batch[0].shape[:-1] + (1,))
        # print("s t shape", t.shape)
        # print("s t_shape1:",batch[0].shape[:-1] + (1,))
        (loss_s_val, gradV), grads_s = grad_fn_s(state_s.model_params, state_q.model_params, batch, t, key)

        grads_sp = jax.lax.pmean(grads_s, axis_name='batch')
        loss_sp = jax.lax.pmean(loss_s_val, axis_name='batch')
        gradV_sp = jax.lax.pmean(gradV, axis_name='batch')
        
        
        def update(optimizer, grad, state):
            updates, opt_state = optimizer.update(grad[0], state.opt_state, state.model_params)
            new_params = optax.apply_updates(state.model_params, updates)
            new_params_ema = jax.tree_util.tree_map(lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate), state.params_ema, new_params)
            new_state = state.replace(step=state.step+1,
                                      opt_state=opt_state, 
                                      model_params=new_params,
                                      params_ema=new_params_ema,
                                      key=key)
            return new_state

        new_state_s = update(optimizer_s, grads_sp, state_s)
        norm_grad_sp = jnp.hstack(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grads_sp[0]))[0]).sum()
        new_carry_state = (step_key, new_state_s, state_q)
        
        return new_carry_state, (loss_sp, gradV_sp, norm_grad_sp) 
    return step_fn_s

def get_step_fn_q(optimizer_s, optimizer_q, loss_fn_q):
    def step_fn_q(carry_state, batch):
        (key, state_s, state_q) = carry_state
        key, step_key = jax.random.split(key)
        grad_fn_q = jax.value_and_grad(loss_fn_q, argnums=[1], has_aux=True)
        
        u0 = jax.random.uniform(key, 1)
        t, _ = sample_t(u0, batch[0].shape[0])
        
        # t = jax.random.uniform(key, (batch[0].shape[0],1))
        # print("q t shape", t.shape)
        # print("q1 t shape", batch[0].shape[:-1] + (1,))
        # t = random.uniform(key, batch[0].shape[:-1] + (1,)) 
        
        (loss_q_val, gradV), grads_q = grad_fn_q(state_s.model_params, state_q.model_params, batch, t, key)
        
        grads_qp = jax.lax.pmean(grads_q, axis_name='batch')
        loss_qp = jax.lax.pmean(loss_q_val, axis_name='batch')
        gradV_qp = jax.lax.pmean(gradV, axis_name='batch')

        def update(optimizer, grad, state):
            updates, opt_state = optimizer.update(grad[0], state.opt_state, state.model_params)
            new_params = optax.apply_updates(state.model_params, updates)
            new_params_ema = jax.tree_util.tree_map(lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate), state.params_ema, new_params)
            new_state = state.replace(step=state.step+1,
                                      opt_state=opt_state, 
                                      model_params=new_params, 
                                      params_ema=new_params_ema,
                                      key=key)
            return new_state

        new_state_q = update(optimizer_q, grads_qp, state_q)
        norm_grad_sq = jnp.hstack(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x)**2, grads_qp[0]))[0]).sum()
        new_carry_state = (step_key, state_s, new_state_q)
        
        return new_carry_state, (loss_qp, gradV_qp, norm_grad_sq)
    return step_fn_q