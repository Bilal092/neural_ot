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

import models

import subset_losses as losses
import train_utils_ss as train_utils
import eval_utils_ss as eval_utils
from data_utils import create_dataset_tf as create_dataset_tf
from data_utils import create_pu_dataset_tf as create_pu_dataset_tf
from data_utils import PhiFilteredDS as PhiFilteredDS
from sklearn.metrics import balanced_accuracy_score
from threshold_optimizer import ThresholdOptimizer

def train(config, workdir):
    EXPERIMENT_NAME = f'{config.data.source}_{config.data.target}_{config.train.mode}'
    checkpoint_dir = os.path.join(workdir, "checkpoints", EXPERIMENT_NAME + "/{}".format(config.c) )
    figure_dir = os.path.join(workdir, "figures", EXPERIMENT_NAME + "/{}".format(config.c) )
    key, *init_keys = random.split(random.PRNGKey(config.seed), 3)
    
    BS = config.data.batch_size
    DIM =  config.data.dim
 
    
    # data_config = config.data
    # p_ds, ul_ds, val_ds, vec_size = create_dataset_tf(data_config.dataset, train_ratio=data_config.train_ratio, batch_size=data_config.batch_size, additional_dim=data_config.additional_dim, seed=0)
    
    model_s = models.Smodel(num_hid=config.model_s.num_hid, num_out=config.model_s.num_out)
    model_q = models.Qmodel(num_hid=config.model_q.num_hid, num_out=config.model_q.num_out)
    
    optimizer_s = train_utils.get_optimizer(config.optimizer_s)
    optimizer_q = train_utils.get_optimizer(config.optimizer_q)

    
    init_params_s = model_s.init(init_keys[0], np.ones([BS,1]), np.ones([BS,DIM]))
    init_params_q = model_q.init(init_keys[1], np.ones([BS,1]), np.ones([BS,DIM]), np.ones([BS,DIM]))

    opt_state_s = optimizer_s.init(init_params_s)
    opt_state_q = optimizer_q.init(init_params_q)

    state_s = train_utils.State(step=0, opt_state=opt_state_s, model_params=init_params_s,params_ema =init_params_s, ema_rate=config.model_s.ema_rate, c=config.c, key=random.PRNGKey(0))
    state_q = train_utils.State(step=0, opt_state=opt_state_q, model_params=init_params_q, params_ema =init_params_q, ema_rate=config.model_q.ema_rate, c=config.c, key=random.PRNGKey(0)) 
    
    options = ocp.CheckpointManagerOptions(max_to_keep=10000)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=options, item_names={"state_s", "state_q"})
    ckpt_mgr.save(0, args=ocp.args.Composite(state_s=ocp.args.StandardSave(state_s),state_q=ocp.args.StandardSave(state_q)))
    ckpt_mgr.wait_until_finished()
    

    data_config = config.data
    # p_ds, ul_ds, val_ds, vec_size = create_dataset_tf(data_config.dataset, train_ratio=data_config.train_ratio, batch_size=data_config.batch_size, additional_dim=data_config.additional_dim, seed=0)
    # def create_pu_dataset_tf(config, dataset_p, dataset_u, size_p, size_u_eval, prior, seed_nb=None, additional_dim=None, seed=None):
    p_ds, ul_ds, val_ds, vec_size = create_pu_dataset_tf(config, config.data.source, config.data.target, 
                                                         config.data.size_p, config.data.size_u_val, 
                                                         config.prior, seed_nb=config.seed, additional_dim=config.train.n_jitted_steps)
    
    p_train_iter = iter(p_ds.repeat(count=None))
    ul_train_iter = iter(ul_ds.repeat(count=None))
    

    def batch_iterator():
        return next(p_train_iter), next(ul_train_iter)
    
    if jax.process_index() == 0:
        wandb.init(name="c={}".format(config.c) + "_seed_{}".format(config.seed), 
                project='PU_learning_dynamic_' + EXPERIMENT_NAME, 
                resume="allow",
                config = json.loads(config.to_json_best_effort()))
        os.environ["WANDB_RESUME"] = "allow"
    
    loss_s =  losses.get_loss_s(config, model_s, model_q)
    loss_q = losses.get_loss_q(config, model_s, model_q)

    step_fn_s = train_utils.get_step_fn_s(config, optimizer_s, optimizer_q, loss_s)
    step_fn_s =jax.pmap(partial(jax.lax.scan, step_fn_s), axis_name = 'batch')

    step_fn_q = train_utils.get_step_fn_q(config, optimizer_s, optimizer_q, loss_q)
    step_fn_q =jax.pmap(partial(jax.lax.scan, step_fn_q), axis_name = 'batch')

    s_eval_fn = train_utils.get_s_eval_func(model_s)
    s_eval_fn = jax.pmap(s_eval_fn, axis_name="batch")

    q_func = train_utils.get_q_func(model_q)
    q_func = jax.pmap(q_func, axis_name='batch')
    
    gen_ode = train_utils.get_generator_ode(model_s, dt=config.train.dt)
    gen_ode = jax.pmap(gen_ode, axis_name="batch")
    
    input_shape = (data_config.batch_size, config.data.dim)
    input_shape_p = (jax.device_count(), data_config.batch_size//jax.device_count(), config.data.dim)
    
    
    def get_phi_fn(config, model_s):
        def phi_fn(params, x):
            return model_s.apply(params, jnp.ones(x.shape[:-1] + (1,)), x)
        return phi_fn
    
    phi_fn = get_phi_fn(config, model_s)
    phi_fn = jax.pmap(phi_fn)
    
    state_s = flax.jax_utils.replicate(state_s) 
    state_q = flax.jax_utils.replicate(state_q)

    # phi_fn = jax.pmap(model_s.apply)
    
    calibrator = ThresholdOptimizer(k=3, n=100)   
    if config.c != 1.0:
        eps_scheduler = train_utils.get_eps_scheduler(config)
        target_ds_filter = PhiFilteredDS(config, ul_ds, phi_fn, model_s, state_s.params_ema, eps_scheduler)
        rejection_sample_iterator = target_ds_filter.create_batch_iterator()
    
    for step in range(config.train.num_train_steps+1):
        if config.c != 1:
            if step % config.phi_sampler.refresh_every == 0 and step >= config.phi_sampler.min_update_step:
                target_ds_filter.update_if_needed(step, state_s.model_params)
                rejection_sample_iterator = target_ds_filter.create_batch_iterator()
        
        for s_step in range(0, config.train.num_s_steps):
            key, loc_key = random.split(key)
            (x0_, y0_), (x1_, y1_) = batch_iterator()
            batch = (x0_.numpy(), x1_.numpy())
            if config.c == 1.0: 
                batch = batch + (batch[1],)
            else: 
                if step >= config.phi_sampler.min_update_step:
                    batch = batch + (rejection_sample_iterator()[0],)
                else:
                    batch = batch + (batch[1],)
                
            key, *next_key = random.split(key, num=jax.local_device_count() + 1)
            next_key = jnp.asarray(next_key)
            (_, state_s, state_q), (total_loss_s, gradV_s) = step_fn_s((next_key, state_s, state_q), batch)
            total_loss_s = flax.jax_utils.unreplicate(total_loss_s).mean()
            gradV_s = flax.jax_utils.unreplicate(gradV_s).mean()
   
      
        for q_step in range(0, config.train.num_q_steps):
            key, loc_key = random.split(key)
            (x0_, y0_), (x1_, y1_) = batch_iterator()
            batch = (x0_.numpy(), x1_.numpy())
            if config.c == 1.0: 
                batch = batch + (batch[1],)
            else: 
                if step >= config.phi_sampler.min_update_step:
                    batch = batch + (rejection_sample_iterator()[0],)
                else:
                    batch = batch + (batch[1],)
                
            key, *next_key = random.split(key, num=jax.local_device_count() + 1)
            next_key = jnp.asarray(next_key)
            (_, state_s, state_q), (total_loss_q, gradV_q) = step_fn_q((next_key, state_s, state_q), batch)
            total_loss_q = flax.jax_utils.unreplicate(total_loss_q).mean()
            gradV_q = flax.jax_utils.unreplicate(gradV_q).mean()

        if (step % config.train.log_interval_steps == 0) and (jax.process_index() == 0):
            wandb.log(dict(loss_s=total_loss_s), step=step)
            wandb.log(dict(gradV_s=gradV_s), step=step)
            
            wandb.log(dict(loss_q=total_loss_q), step=step)
            wandb.log(dict(gradV_q=gradV_q), step=step)


        if (step % config.train.eval_interval_steps == 0) and (jax.process_index() == 0):
            # acc_one_step, target_acc_one_step = evaluation.evaluate(config, model_s_eval, state_s, source_test_ds, gen_one_step)
            # acc_ode, target_acc_ode = evaluation.evaluate(config, model_s_eval, state_s, source_test_ds, gen_ode)
            #evaluate_pu_fn = get_pu_evaluation_fn(config, model_s, state_s.model_params)
            # acc_PU = evaluate_pu_fn(state_s.model_params, val_ds)
            
            label_train, pred_train = train_utils.evaluate_pu(ul_ds, state_s, s_eval_fn) # evaluate_pu return jax.numpy.array
            label_val, pred_val = train_utils.evaluate_pu(val_ds, state_s, s_eval_fn) # evaluate_pu return jax.numpy.array
            
            # avg_v0_train = train_utils.eval_init_velocities(p_ds, state_s, s_eval_fn).mean()
            # avg_v0_val = train_utils.eval_init_velocities(val_ds, state_s, s_eval_fn).mean()
            print("label.shape", label_val.shape)
            print("pred.shape", pred_val.shape)

            true_prior_val = label_val.mean()
            true_prior_train = label_train.mean()
            
            # acc_PU_phi1_train = 100*(label_train == (pred_train<=0)).mean()
            # acc_PU_phi1_val = 100*(label_val == (pred_val<=0)).mean()
            
            pred_args_sort_train = np.argsort(pred_train) 
            label_hat_train = np.zeros_like(pred_train, np.uint8)
            label_hat_pi_train =  np.zeros_like(pred_train, np.uint8)
            label_hat_train[pred_args_sort_train[:int(true_prior_train*len(pred_train))]] = 1 # jax.numpy.array does not support inline operations
            label_hat_pi_train[pred_args_sort_train[:int(config.prior*len(pred_train))]] = 1 # jax.numpy.array does not support inline operations
              
            pred_args_sort_val = np.argsort(pred_val) 
            label_hat_val = np.zeros_like(pred_val, np.uint8)
            label_hat_pi_val = np.zeros_like(pred_val, np.uint8)
            label_hat_val[pred_args_sort_val[:int(true_prior_val*len(pred_val))]] = 1 # jax.numpy.array does not support inline operations
            label_hat_pi_val[pred_args_sort_val[:int(config.prior*len(pred_val))]] = 1 # jax.numpy.array does not support inline operations
            

            if config.train.mode != "propensity":
                acc_eta_train = 100*(label_train == (pred_train<=0)).mean()
                acc_eta_val = 100*(label_val == (pred_val<=0)).mean()
                acc_PU_sorted_train = 100*(label_train == label_hat_train).mean()
                acc_PU_sorted_val = 100*(label_val == label_hat_val).mean()
                acc_PU_sorted_pi_train = 100*(label_train == label_hat_pi_train).mean()
                acc_PU_sorted_pi_val = 100*(label_val == label_hat_pi_val).mean()
            else:
                acc_eta_train = balanced_accuracy_score(label_train, (pred_train<=0).astype(np.uint8))
                acc_eta_val = balanced_accuracy_score(label_val, (pred_val<=0).astype(np.uint8))
                acc_PU_sorted_train = balanced_accuracy_score(label_train, label_hat_train)
                acc_PU_sorted_val = balanced_accuracy_score(label_val, label_hat_val)   
                acc_PU_sorted_pi_train = balanced_accuracy_score(label_train, label_hat_pi_train)
                acc_PU_sorted_pi_val = balanced_accuracy_score(label_val, label_hat_pi_val)
                
                if step == config.train.num_train_steps:
                    calibrated_threshold = calibrator.find_threshold(pred_train)
                    acc_eta_calibrated_train = balanced_accuracy_score(label_train, (pred_train>=calibrated_threshold))
                    acc_eta_calibrated_val = balanced_accuracy_score(label_val, (pred_val>=calibrated_threshold))

            wandb.log(dict(loss_s=total_loss_s), step=step)
            wandb.log(dict(gradV_s=gradV_s), step=step)
            wandb.log(dict(loss_q=total_loss_q), step=step)
            wandb.log(dict(gradV_q=gradV_q), step=step)

            wandb.log(dict(acc_eta_train=acc_eta_train, acc_sorted_train=acc_PU_sorted_train), step=step)
            wandb.log(dict(acc_eta_val=acc_eta_val, acc_sorted_val=acc_PU_sorted_val), step=step)
            wandb.log(dict(acc_sorted_pi_train=acc_PU_sorted_pi_train), step=step)
            wandb.log(dict(acc_sorted_pi_val=acc_PU_sorted_pi_val), step=step)
            
            if step == config.train.num_train_steps:
                wandb.log(dict(acc_eta_train_calibrated=acc_eta_calibrated_train), step=step)
                wandb.log(dict(acc_eta_val_calibrated=acc_eta_calibrated_val), step=step)
        
        if  step % config.train.save_interval_steps == 0:
            state_s_saved =  flax.jax_utils.unreplicate(state_s)
            state_s_saved =  state_s_saved.replace(key=key)
            state_q_saved =  flax.jax_utils.unreplicate(state_q)
            state_q_saved =  state_q_saved.replace(key=key)
            ckpt_mgr.save(step, args=ocp.args.Composite(state_s=ocp.args.StandardSave(state_s_saved),state_q=ocp.args.StandardSave(state_q_saved)))
            ckpt_mgr.wait_until_finished()
        
    if jax.process_index() == 0:
            wandb.finish()
    
    def evaluate(config, workdir):
        
        EXPERIMENT_NAME = f'{config.data.source}_{config.data.target}'
        checkpoint_dir = os.path.join(workdir, "checkpoints", EXPERIMENT_NAME + "/{}".format(config.c) )
        figure_dir = os.path.join(workdir, "figures", EXPERIMENT_NAME + "/{}".format(config.c) )
        key, *init_keys = random.split(key, 3)
        
        model_s = models.Smodel(num_hid=config.s.num_hid, num_out=config.s.num_out)
        model_q = models.Qmodel(num_hid=config.q.num_hid, num_out=config.data.dim) 
        
        options = ocp.CheckpointManagerOptions(max_to_keep=10000)
        ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=options, item_names={"state_s", "state_q"})
        eval_state = ckpt_mgr.eval(config.eval.step, args=ocp.args.Composite(state_s=ocp.args.StandardSave(state_s_saved),state_q=ocp.args.StandardSave(state_q_saved)))
        
        data_config = config.data
        p_ds, ul_ds, val_ds, vec_size = create_dataset_tf(data_config.dataset, train_ratio=data_config.train_ratio, batch_size=data_config.batch_size, additional_dim=data_config.additional_dim, seed=0)
    
        p_train_iter = iter(p_ds.repeat())
        ul_train_iter = iter(ul_ds.repeat())  
        
        if eval.dataset == "train":
            eval_ds =  p_train_iter
        elif eval.dataset == "eval":
            eval_ds = ul_train_iter
            
            
        labels, pred = eval_utils.evaluate_pu(val_ds, state_s, s_eval_fn)
        acc_PU = (labels == pred).mean()
        
        return acc_PU, (labels, pred)
        
        
    
    
    