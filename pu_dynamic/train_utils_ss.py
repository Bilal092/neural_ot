from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np
from typing import NamedTuple, Any

@flax.struct.dataclass
class State:
  step: int
  opt_state: Any
  model_params: Any
  ema_rate: Any
  params_ema: Any
  key: Any
  c: Any

def get_optimizer(config):
    return optax.adam(learning_rate=config.lr)

from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np
from typing import NamedTuple, Any

@flax.struct.dataclass
class State:
  step: int
  opt_state: Any
  model_params: Any
  ema_rate: Any
  params_ema: Any
  key: Any
  c: Any

def get_optimizer(config):
    return optax.adam(learning_rate=config.lr)

def get_eps_scheduler(config):
    return optax.piecewise_interpolate_schedule(interpolate_type='linear', init_value=1.0, boundaries_and_scales={1: 1.0, int(config.train.num_train_steps//2): 0.0})
    

def get_step_fn_s(config, optimizer_s, optimizer_q, loss_fn_s):
    def step_fn_s(carry_state, batch):
        key, state_s, state_q = carry_state
        key, step_key = jax.random.split(key)

        grad_fn_s = jax.value_and_grad(loss_fn_s, argnums=0, has_aux=True)
        (loss_s_val, translation), grads_s = grad_fn_s(state_s.model_params, state_q.model_params, batch, key)

        grads_sp = jax.lax.pmean(grads_s, axis_name='batch')
        loss_sp = jax.lax.pmean(loss_s_val, axis_name='batch')
        translation_p = jax.lax.pmean(translation, axis_name='batch')

        updates, new_opt_state = optimizer_s.update(grads_sp, state_s.opt_state, params=state_s.model_params)
        new_params = optax.apply_updates(state_s.model_params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * state_s.ema_rate + p * (1. - state_s.ema_rate),
            state_s.params_ema, new_params)

        new_state_s = state_s.replace(
            step=state_s.step + 1,
            opt_state=new_opt_state,
            model_params=new_params,
            params_ema=new_params_ema,
            key=key,
        )
        carry_state = (step_key, new_state_s, state_q)

        return carry_state, (loss_sp, translation_p)
    return step_fn_s


def get_step_fn_q(config, optimizer_s, optimizer_q, loss_fn_eta):
    def step_fn_q(carry_state, batch):
        key, state_s, state_q = carry_state
        key, step_key = jax.random.split(key)

        grad_fn_q = jax.value_and_grad(loss_fn_eta, argnums=1, has_aux=True)
        (loss_q_val, translation), grads_q = grad_fn_q(state_s.model_params, state_q.model_params, batch, key)

        grads_qp = jax.lax.pmean(grads_q, axis_name='batch')
        loss_qp = jax.lax.pmean(loss_q_val, axis_name='batch')
        translation_p = jax.lax.pmean(translation, axis_name='batch')

        updates, new_opt_state = optimizer_q.update(grads_qp, state_q.opt_state, params=state_q.model_params)
        new_params = optax.apply_updates(state_q.model_params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * state_q.ema_rate + p * (1. - state_q.ema_rate),
            state_q.params_ema, new_params)

        new_state_q = state_q.replace(
            step=state_q.step + 1,
            opt_state=new_opt_state,
            model_params=new_params,
            params_ema=new_params_ema,
            key=key,
        )
        carry_state = (step_key, state_s, new_state_q)

        return carry_state, (loss_qp, translation_p)
    return step_fn_q


def get_s_eval_func(model_s):
    def eval_func(state_s, x):
        ones = jnp.ones(x.shape[:-1]+(1,))
        return model_s.apply(state_s.params_ema, ones, x)
        # return model_s.apply(state_s.model_params, ones, x) # this line can be uncommented and above can be commented in case direct model params are to be used for model evaluation
    return eval_func
    
def evaluate_pu(eval_ds, state_s, eval_fn):
    all_preds = []
    all_labels = []
    for data, labels in eval_ds:
        s_val = eval_fn(state_s, jnp.array(data))
        
        all_preds.append(np.array(s_val).reshape(-1))
        all_labels.append(np.array(labels).reshape(-1))

    preds = jnp.concatenate(all_preds, axis=0)
    labels = jnp.concatenate(all_labels, axis=0)

    return labels, preds

def eval_init_velocities(eval_ds, state_s, eval_fn):
    all_v = []
    s_fn = lambda state_s, t, x: eval_fn(state_s,t,x).sum()
    grad_fn = jax.grad(s_fn, argnums=2)
    for data, label in eval_ds:
        zeros = jnp.zeros(data.shape[:-1]+(1,))
        g = grad_fn(state_s, zeros, jnp.array(data))
        g_norm = jnp.sum(g**2, axis=tuple(range(1, g.ndim)))
        all_v.append(np.array(g_norm).reshape(-1))
    
    return np.array(all_v)
        

def get_model_fn(model, params):
    def model_fn(t, x):
        return model.apply(params, t, x)
    return model_fn

def get_generator_ode(model, dt):
    def artifact_gen(x0, state):
        s = get_model_fn(model, params=state.model_params)
        # s = get_model_fn(model, params=state.params_ema)
        ts = jnp.linspace(0, 1.0, int(1/dt)+1)
        dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
        vector_field = lambda _t,_x,_args: dsdx(_t,_x)
        solve = partial(diffrax.diffeqsolve, 
                            terms=diffrax.ODETerm(vector_field), 
                            solver=diffrax.Dopri5(), 
                            t0=0.0, t1=1.0, dt0=dt, 
                            saveat=diffrax.SaveAt(ts=ts),
                            stepsize_controller=diffrax.ConstantStepSize(True), 
                            adjoint=diffrax.NoAdjoint())
        
        solution = solve(y0=x0, args=state)
        ode_int_artifacts = solution
        return ode_int_artifacts
    return artifact_gen

def get_s_func(model_s):
    def s_func(state_s, t, x):
        # print(x.shape)
        # ones = jnp.ones(x.shape[:-1]+(1,))
        return model_s.apply(state_s.model_params, t, x)
    return s_func

def get_q_func(model_q):
    def q_func(state_q, t, x0, x1):
        return model_q.apply(state_q.model_params, t, x0, x1)
    return q_func


