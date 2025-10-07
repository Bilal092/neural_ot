import jax
from jax import numpy as jnp
from jax import random as random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import os
import pandas as pd

from ml_collections.config_flags import config_flags
import jax 

import orbax.checkpoint as ocp
import flax
import jax.random as random
from functools import partial
import wandb

import evaluation
import models
import train_utils
import dataloaders

import subset_losses as losses

jax.clear_caches()


def train(config, workdir):
    
    key = random.PRNGKey(config.seed)
    EXPERIMENT_NAME = f'{config.data.source}_{config.data.target}'
    checkpoint_dir = os.path.join(workdir, "checkpoints", EXPERIMENT_NAME + "/{}".format(config.c) )
    figure_dir = os.path.join(workdir, "figures", EXPERIMENT_NAME + "/{}".format(config.c) )
    key, *init_keys = random.split(key, 3)

    model_s = models.Smodel(config.model_s.num_hid, config.model_s.num_out, config.model_s.t_embed_dim)
    model_s_eval = models.Smodel_eval(config.model_s.num_hid, config.model_s.num_out, config.model_s.t_embed_dim)
    model_q = models.Qmodel(config.model_q.num_hid, config.model_q.num_out, config.model_q.t_embed_dim)

    optimizer_s = train_utils.get_optimizer(config.optimizer_s)
    optimizer_q = train_utils.get_optimizer(config.optimizer_q)

    BS = config.train.batch_size
    DIM =  config.data.dim

    init_params_s = model_s.init(init_keys[0], np.ones([BS,1]), np.ones([BS, DIM]))
    init_params_q = model_q.init(init_keys[1], np.ones([BS,1]), np.ones([BS, DIM]),np.ones([BS, DIM]))

    opt_state_s = optimizer_s.init(init_params_s)
    opt_state_q = optimizer_q.init(init_params_q)

    state_s = models.State(step=0, opt_state=opt_state_s, model_params=init_params_s, ema_rate=config.model_s.ema_rate, params_ema=init_params_s, c=config.c, key=key)
    state_q = models.State(step=0, opt_state=opt_state_q, model_params=init_params_q, ema_rate=config.model_q.ema_rate, params_ema=init_params_q, c=config.c, key=key)

    # options = ocp.CheckpointManagerOptions(max_to_keep=60, save_interval_steps=1000)
    # orbax checkpointer are used to stoe checkpoints
    options = ocp.CheckpointManagerOptions(max_to_keep=10000)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=options, item_names={"state_s", "state_q"})
    ckpt_mgr.save(0, args=ocp.args.Composite(state_s=ocp.args.StandardSave(state_s),state_q=ocp.args.StandardSave(state_q)))
    ckpt_mgr.wait_until_finished()
    
    source_name = config.data.source
    batch_size = config.train.batch_size
    additional_dim=config.data.additional_dim
    source_train_ds, source_test_ds  = dataloaders.get_dataset(source_name, batch_size, additional_dim=additional_dim, source=True )

    target_name = config.data.target
    target_train_ds, _  = dataloaders.get_dataset(target_name, batch_size, additional_dim=additional_dim, source=False )
    
    source_train_iter = iter(source_train_ds.repeat(count=None))
    target_train_iter = iter(target_train_ds.repeat(count=None))

    def batch_iterator():
        return next(source_train_iter)._numpy(), next(target_train_iter)._numpy() 
    
    if jax.process_index() == 0:
        wandb.init(name=EXPERIMENT_NAME+"_c={}".format(config.c), 
                project='dynamic_SS' + '_' + config.data.source + "_" + config.data.target, 
                resume="allow",
                config=json.loads(config.to_json_best_effort()))
        os.environ["WANDB_RESUME"] = "allow"
    
    loss_s =  losses.get_loss_s(config, model_s, model_q)
    loss_q = losses.get_loss_q(config, model_s, model_q)

    step_fn_s = losses.get_step_fn_s(optimizer_s, optimizer_q, loss_s)
    step_fn_s =jax.pmap(partial(jax.lax.scan, step_fn_s), axis_name = 'batch')

    step_fn_q = losses.get_step_fn_q(optimizer_s, optimizer_q, loss_q)
    step_fn_q =jax.pmap(partial(jax.lax.scan, step_fn_q), axis_name = 'batch')

    gen_one_step = evaluation.get_generator_one_step(model_s_eval)
    gen_one_step = jax.pmap(gen_one_step, axis_name="batch")

    gen_ode = evaluation.get_generator_ode(model_s_eval)
    gen_ode = jax.pmap(gen_ode, axis_name="batch")
    
    input_shape = (config.train.batch_size, config.data.dim)
    input_shape_p = (jax.device_count(), batch_size//jax.device_count(), config.data.dim)
    
    def get_phi_fn(config, model_s):
        def phi_fn(params, x):
            return model_s.apply(params, jnp.ones(x.shape[:-1] + (1,)), x)
        return phi_fn
    
    phi_fn = get_phi_fn(config, model_s)
    phi_fn = jax.pmap(phi_fn)
    
    state_s = flax.jax_utils.replicate(state_s) 
    state_q = flax.jax_utils.replicate(state_q)
    
    # __init__(self, config, base_dataset, phi_fn,  phi_model, params, eps_scheduler, threshold=1e-5,
    #                  evaluation=False, additional_dim=None, uniform_dequantization=False)
    
    if config.c != 1.0:
        eps_scheduler = train_utils.get_eps_scheduler(config)
        target_ds_filter = dataloaders.PhiFilteredDS(config, target_train_ds, phi_fn, model_s, state_s.model_params, eps_scheduler)
        rejection_sample_iterator = target_ds_filter.create_batch_iterator()
    
    for step in range(config.train.num_train_steps+1):
        if config.c != 1:
            if step % config.phi_sampler.refresh_every == 0 and step >= config.phi_sampler.min_update_step:
                target_ds_filter.update_if_needed(step, state_s.model_params)
                rejection_sample_iterator = target_ds_filter.create_batch_iterator()
        
        for s_step in range(0, config.train.num_s_steps):
            key, loc_key = random.split(key)
            batch = batch_iterator()
            if config.c == 1.0: 
                batch = batch + (batch[1],)
            else: 
                if step >= config.phi_sampler.min_update_step:
                    batch = batch + (rejection_sample_iterator(),)
                else:
                    batch = batch + (batch[1],)
                    
            key, *next_key = random.split(key, num=jax.local_device_count() + 1)
            next_key = jnp.asarray(next_key)
            (_, state_s, state_q), (total_loss_s, gradV_s, norm_grad_s) = step_fn_s((next_key, state_s, state_q), batch)
            total_loss_s = flax.jax_utils.unreplicate(total_loss_s).mean()
            gradV_s = flax.jax_utils.unreplicate(gradV_s).mean()
            norm_grad_s = flax.jax_utils.unreplicate(norm_grad_s).mean()
      
        for q_step in range(0, config.train.num_q_steps):
            key, loc_key = random.split(key)
            batch = batch_iterator()
            if config.c == 1.0: 
                batch = batch + (batch[1],)
            else: 
                if step >= config.phi_sampler.min_update_step:
                    batch = batch + (rejection_sample_iterator(),)
                else:
                    batch = batch + (batch[1],)
                    
            key, *next_key = random.split(key, num=jax.local_device_count() + 1)
            next_key = jnp.asarray(next_key)
            (_, state_s, state_q), (total_loss_q, gradV_q, norm_grad_q) = step_fn_q((next_key, state_s, state_q), batch)
            total_loss_q = flax.jax_utils.unreplicate(total_loss_q).mean()
            gradV_q = flax.jax_utils.unreplicate(gradV_q).mean()
            norm_grad_q = flax.jax_utils.unreplicate(norm_grad_q).mean()

        if (step % config.train.eval_interval_steps == 0) and (jax.process_index() == 0):
            acc_one_step, target_acc_one_step = evaluation.evaluate(config, model_s_eval, state_s, source_test_ds, gen_one_step)
            acc_ode, target_acc_ode = evaluation.evaluate(config, model_s_eval, state_s, source_test_ds, gen_ode)
            
            wandb.log(dict(loss_s=total_loss_s), step=step)
            wandb.log(dict(gradV_s=gradV_s), step=step)
            wandb.log(dict(norm_grad_s=norm_grad_s), step=step)
            
            wandb.log(dict(loss_q=total_loss_q), step=step)
            wandb.log(dict(gradV_q=gradV_q), step=step)
            wandb.log(dict(norm_grad_q=norm_grad_q), step=step)
            
            wandb.log(dict(acc_one_step=100*acc_one_step, target_acc_one_step=100*target_acc_one_step), step=step)
            wandb.log(dict(acc_ode=100*acc_ode, target_acc_ode=100*target_acc_ode), step=step)
        
        if  step % config.train.save_interval_steps == 0:
            state_s_saved =  flax.jax_utils.unreplicate(state_s)
            state_s_saved =  state_s_saved.replace(key=key)
            state_q_saved =  flax.jax_utils.unreplicate(state_q)
            state_q_saved =  state_q_saved.replace(key=key)
            ckpt_mgr.save(step, args=ocp.args.Composite(state_s=ocp.args.StandardSave(state_s_saved),state_q=ocp.args.StandardSave(state_q_saved)))
            ckpt_mgr.wait_until_finished()
            
    
            
            
    
    
    
  