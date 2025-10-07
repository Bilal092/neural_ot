from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np
from typing import NamedTuple, Any

def get_eta_eval_func(model_eta):
    def eval_func(state_eta, x):
        return model_eta.apply(state_eta.params_ema, x)
        # ones = jnp.ones(x.shape[:-1]+(1,))
        # return model_s.apply(state_s.params_ema, ones, x)
         
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

# def eval_init_velocities(eval_ds, state_s, eval_fn):
#     all_v = []
#     s_fn = lambda state_s, t, x: eval_fn(state_s,t,x).sum()
#     grad_fn = jax.grad(s_fn, argnums=2)
#     for data, label in eval_ds:
#         zeros = jnp.zeros(data.shape[:-1]+(1,))
#         g = grad_fn(state_s, zeros, jnp.array(data))
#         g_norm = jnp.sum(g**2, axis=tuple(range(1, g.ndim)))
#         all_v.append(np.array(g_norm).reshape(-1))
    
#     return np.array(all_v)
        

# def get_model_fn(model, params):
#     def model_fn(t, x):
#         return model.apply(params, t, x)
#     return model_fn

# def get_generator_ode(model, dt):
#     def artifact_gen(x0, state):
#         s = get_model_fn(model, params=state.model_params)
#         # s = get_model_fn(model, params=state.params_ema)
#         ts = jnp.linspace(0, 1.0, int(1/dt)+1)
#         dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
#         vector_field = lambda _t,_x,_args: dsdx(_t,_x)
#         solve = partial(diffrax.diffeqsolve, 
#                             terms=diffrax.ODETerm(vector_field), 
#                             solver=diffrax.Dopri5(), 
#                             t0=0.0, t1=1.0, dt0=dt, 
#                             saveat=diffrax.SaveAt(ts=ts),
#                             stepsize_controller=diffrax.ConstantStepSize(True), 
#                             adjoint=diffrax.NoAdjoint())
        
#         solution = solve(y0=x0, args=state)
#         ode_int_artifacts = solution
#         return ode_int_artifacts
#     return artifact_gen

# def get_s_func(model_s):
#     def s_func(state_s, t, x):
#         # print(x.shape)
#         # ones = jnp.ones(x.shape[:-1]+(1,))
#         return model_s.apply(state_s.model_params, t, x)
#     return s_func

# def get_q_func(model_q):
#     def q_func(state_q, t, x0, x1):
#         return model_q.apply(state_q.model_params, t, x0, x1)
#     return q_func


