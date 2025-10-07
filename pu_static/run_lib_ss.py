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
from sklearn.metrics import balanced_accuracy_score
from threshold_optimizer import ThresholdOptimizer


def train(config, workdir):
    if config.train.mode == "propensity":
        EXPERIMENT_NAME = f'{config.data.source}_{config.data.target}_{config.train.mode}_{config.train.label_strat}_'
    else:    
        EXPERIMENT_NAME = f'{config.data.source}_{config.data.target}_{config.train.mode}' 
        
    checkpoint_dir = os.path.join(workdir, "checkpoints", EXPERIMENT_NAME + "/{}".format(config.c) + "/seed_{}".format(config.seed) )
    figure_dir = os.path.join(workdir, "figures", EXPERIMENT_NAME + "/{}".format(config.c)+ "/seed_{}".format(config.seed) )
    key, *init_keys = random.split(random.PRNGKey(config.seed), 3)
    
    BS = config.data.batch_size
    DIM =  config.data.dim
 
    # data_config = config.data
    # p_ds, ul_ds, val_ds, vec_size = create_dataset_tf(data_config.dataset, train_ratio=data_config.train_ratio, batch_size=data_config.batch_size, additional_dim=data_config.additional_dim, seed=0)
    
    model_T = models.Tmodel(num_hid=config.model_T.num_hid, num_out=config.model_T.num_out)
    model_eta = models.etamodel(num_hid=config.model_eta.num_hid, num_out=config.model_eta.num_out)
    
    optimizer_T = train_utils.get_optimizer(config.optimizer_T)
    optimizer_eta = train_utils.get_optimizer(config.optimizer_eta)
    
    init_params_T = model_T.init(init_keys[0], np.ones([BS,DIM]))
    init_params_eta = model_eta.init(init_keys[1], np.ones([BS,DIM]))

    opt_state_T = optimizer_T.init(init_params_T)
    opt_state_eta = optimizer_eta.init(init_params_eta)

    state_T = train_utils.State(step=0, opt_state=opt_state_T, model_params=init_params_T,params_ema=init_params_T, ema_rate=config.model_T.ema_rate, c=config.c, key=random.PRNGKey(0))
    state_eta = train_utils.State(step=0, opt_state=opt_state_eta, model_params=init_params_eta, params_ema =init_params_eta, ema_rate=config.model_eta.ema_rate, c=config.c, key=random.PRNGKey(0))
    
    options = ocp.CheckpointManagerOptions(max_to_keep=10000)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=options, item_names={"state_T", "state_eta"})
    ckpt_mgr.save(0, args=ocp.args.Composite(state_T=ocp.args.StandardSave(state_T),state_eta=ocp.args.StandardSave(state_eta)))
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
        wandb.init(name=EXPERIMENT_NAME+"_c={}_".format(config.c) + "_seed_{}".format(config.seed), 
                project='PU_learning_static_' + EXPERIMENT_NAME, 
                resume="allow",
                config = json.loads(config.to_json_best_effort()))
        os.environ["WANDB_RESUME"] = "allow"
    
    loss_T =  losses.get_loss_T(config, model_T, model_eta)
    loss_eta = losses.get_loss_eta(config, model_T, model_eta)

    step_fn_T = train_utils.get_step_fn_T(config, optimizer_T, optimizer_eta, loss_T)
    step_fn_T =jax.pmap(partial(jax.lax.scan, step_fn_T), axis_name = 'batch')

    step_fn_eta = train_utils.get_step_fn_eta(config, optimizer_T, optimizer_eta, loss_eta)
    step_fn_eta =jax.pmap(partial(jax.lax.scan, step_fn_eta), axis_name = 'batch')

    eta_eval_fn = train_utils.get_eta_eval_func(model_eta)
    eta_eval_fn = jax.pmap(eta_eval_fn, axis_name="batch")

    # q_func = train_utils.get_q_func(model_q)
    # q_func = jax.pmap(q_func, axis_name='batch')
    
    # gen_ode = train_utils.get_generator_ode(model_s, dt=config.train.dt)
    # gen_ode = jax.pmap(gen_ode, axis_name="batch")
    
    input_shape = (data_config.batch_size, config.data.dim)
    input_shape_p = (jax.device_count(), data_config.batch_size//jax.device_count(), config.data.dim)

    state_T = flax.jax_utils.replicate(state_T) 
    state_eta = flax.jax_utils.replicate(state_eta)
    
    calibrator = ThresholdOptimizer(k=3, n=100)

    for step in range(config.train.num_train_steps+1):
        for T_iter in range(0, config.train.num_T_steps):
            key, loc_key = random.split(key)
            (x0_, y0_), (x1_, y1_) = batch_iterator()
            batch = (x0_.numpy(), x1_.numpy())

            key, *next_key = random.split(key, num=jax.local_device_count() + 1)
            next_key = jnp.asarray(next_key)
            (_, state_T, state_eta), (total_loss_T, translation_T) = step_fn_T((next_key, state_T, state_eta), batch)
            total_loss_T = flax.jax_utils.unreplicate(total_loss_T).mean()
            translation_T = flax.jax_utils.unreplicate(translation_T).mean()
   
      
        for eta_iter in range(0, config.train.num_eta_steps):
            key, loc_key = random.split(key)
            
            (x0_, y0_), (x1_, y1_) = batch_iterator()
            batch = (x0_.numpy(), x1_.numpy())
            key, *next_key = random.split(key, num=jax.local_device_count() + 1)
            next_key = jnp.asarray(next_key)
            (_, state_T, state_eta), (total_loss_eta, translation_eta) = step_fn_eta((next_key, state_T, state_eta), batch)
            total_loss_eta = flax.jax_utils.unreplicate(total_loss_eta).mean()
            translation_eta = flax.jax_utils.unreplicate(translation_eta).mean()

        if (step % config.train.log_interval_steps == 0) and (jax.process_index() == 0):
            wandb.log(dict(loss_T=total_loss_T), step=step)
            wandb.log(dict(translation_T = translation_T), step=step)
            
            wandb.log(dict(loss_eta=total_loss_eta), step=step)
            wandb.log(dict(translation_eta = translation_eta), step=step)

        if (step % config.train.eval_interval_steps == 0) and (jax.process_index() == 0):
            
            label_train, pred_train = train_utils.evaluate_pu(ul_ds, state_eta, eta_eval_fn) # evaluate_pu return jax.numpy.array
            label_val, pred_val = train_utils.evaluate_pu(val_ds, state_eta, eta_eval_fn) # evaluate_pu return jax.numpy.array
            print("label.shape", label_train.shape)
            print("pred.shape", pred_train.shape)
            
            true_prior_val = label_val.mean()
            true_prior_train = label_train.mean()
            
            pred_args_sort_train = np.argsort(-pred_train) 
            label_hat_train = np.zeros_like(pred_train, np.uint8)
            label_hat_pi_train =  np.zeros_like(pred_train, np.uint8)
            label_hat_train[pred_args_sort_train[:int(true_prior_train*len(pred_train))]] = 1 # jax.numpy.array does not support inline operations
            label_hat_pi_train[pred_args_sort_train[:int(config.prior*len(pred_train))]] = 1 # jax.numpy.array does not support inline operations
            
            # these statistics are evaluated for given value of prior
            pred_args_sort_val = np.argsort(-pred_val) 
            label_hat_val = np.zeros_like(pred_val, np.uint8)
            label_hat_pi_val = np.zeros_like(pred_val, np.uint8)
            label_hat_val[pred_args_sort_val[:int(true_prior_val*len(pred_val))]] = 1 # jax.numpy.array does not support inline operations
            label_hat_pi_val[pred_args_sort_val[:int(config.prior*len(pred_val))]] = 1 # jax.numpy.array does not support inline operations
            
            if config.train.mode != "propensity":
                acc_eta_train = 100*(label_train == (pred_train>=0)).mean()
                acc_eta_val = 100*(label_val == (pred_val>=0)).mean()
                acc_PU_sorted_train = 100*(label_train == label_hat_train).mean()
                acc_PU_sorted_val = 100*(label_val == label_hat_val).mean()
                acc_PU_sorted_pi_train = 100*(label_train == label_hat_pi_train).mean()
                acc_PU_sorted_pi_val = 100*(label_val == label_hat_pi_val).mean()
            else:
                acc_eta_train = balanced_accuracy_score(label_train, (pred_train>=0).astype(np.uint8))
                acc_eta_val = balanced_accuracy_score(label_val, (pred_val>=0).astype(np.uint8))
                acc_PU_sorted_train = balanced_accuracy_score(label_train, label_hat_train)
                acc_PU_sorted_val = balanced_accuracy_score(label_val, label_hat_val)   
                acc_PU_sorted_pi_train = balanced_accuracy_score(label_train, label_hat_pi_train)
                acc_PU_sorted_pi_val = balanced_accuracy_score(label_val, label_hat_pi_val)
                
                if step == config.train.num_train_steps:
                    calibrated_threshold = calibrator.find_threshold(pred_train)
                    acc_eta_calibrated_train = balanced_accuracy_score(label_train, (pred_train>=calibrated_threshold))
                    acc_eta_calibrated_val = balanced_accuracy_score(label_val, (pred_val>=calibrated_threshold))
                
            # if config.train.mode != "propensity":
            #     acc_eta_train = 100*(label_train == (pred_train>=0)).mean()
            #     acc_eta_val = 100*(label_val == (pred_val>=0)).mean()
            #     acc_PU_sorted_train = 100*(label_train == label_hat_train).mean()
            #     acc_PU_sorted_val = 100*(label_val == label_hat_val).mean()
                
            # else:
            #     acc_eta_train = balanced_accuracy_score(label_train, (pred_train>=0).astype(np.uint8))
            #     acc_eta_val = balanced_accuracy_score(label_val, (pred_val>=0).astype(np.uint8))
            #     acc_PU_sorted_train = balanced_accuracy_score(label_train, label_hat_train)
            #     acc_PU_sorted_val = balanced_accuracy_score(label_val, label_hat_val)
                
            #     calibrated_threshold = calibrator.find_threshold(pred_train)
            #     acc_eta_train_calibrated = balanced_accuracy_score(label_train, (pred_train>=calibrated_threshold))
            #     acc_eta_val_calibrated = balanced_accuracy_score(label_val, (pred_val>=calibrated_threshold))
                

            wandb.log(dict(acc_eta_train=acc_eta_train, acc_sorted_train=acc_PU_sorted_train), step=step)
            wandb.log(dict(acc_eta_val=acc_eta_val, acc_sorted_val=acc_PU_sorted_val), step=step)
            wandb.log(dict(acc_sorted_pi_train=acc_PU_sorted_pi_train), step=step)
            wandb.log(dict(acc_sorted_pi_val=acc_PU_sorted_pi_val), step=step)
            
            if step == config.train.num_train_steps:
                wandb.log(dict(acc_eta_train_calibrated=acc_eta_calibrated_train), step=step)
                wandb.log(dict(acc_eta_val_calibrated=acc_eta_calibrated_val), step=step)

        if  step % config.train.save_interval_steps == 0:
            state_T_saved =  flax.jax_utils.unreplicate(state_T)
            state_T_saved =  state_T_saved.replace(key=key)
            state_eta_saved =  flax.jax_utils.unreplicate(state_eta)
            state_eta_saved =  state_eta_saved.replace(key=key)
            ckpt_mgr.save(step, args=ocp.args.Composite(state_T=ocp.args.StandardSave(state_T_saved),state_eta=ocp.args.StandardSave(state_eta_saved)))
            ckpt_mgr.wait_until_finished()
            
    if jax.process_index() == 0:
            wandb.finish()
    
    
    # def evaluate(config, workdir):
        
    #     EXPERIMENT_NAME = f'{config.data.source}_{config.data.target}_{config.train.mode}_{config.train.label_strat}_'
    #     checkpoint_dir = os.path.join(workdir, "checkpoints", EXPERIMENT_NAME + "/{}".format(config.c) )
    #     figure_dir = os.path.join(workdir, "figures", EXPERIMENT_NAME + "/{}".format(config.c) )
    #     key, *init_keys = random.split(key, 3)
        
    #     model_T = models.Tmodel(num_hid=config.model_T.num_hid, num_out=config.model_T.num_out)
    #     model_eta = models.etamodel(num_hid=config.model_eta.num_hid, num_out=config.model_eta.num_out)
        
    #     options = ocp.CheckpointManagerOptions(max_to_keep=10000)
    #     ckpt_mgr = ocp.CheckpointManager(checkpoint_dir, options=options, item_names={"state_T", "state_eta"})
    #     eval_state = ckpt_mgr.restore(config.eval.checkpoint_step, args=ocp.args.Composite(state_T=ocp.args.StandardRestore(state_T), state_eta=ocp.args.StandardRestore(state_eta)))
        
    #     data_config = config.data
    #     p_ds, ul_ds, val_ds, vec_size = create_dataset_tf(data_config.dataset, train_ratio=data_config.train_ratio, batch_size=data_config.batch_size, additional_dim=data_config.additional_dim, seed=0)
    
    #     p_train_iter = iter(p_ds.repeat())
    #     ul_train_iter = iter(ul_ds.repeat())  
        
    #     eta_eval_fn = train_utils.get_eta_eval_func(model_eta)
    #     eta_eval_fn = jax.pmap(eta_eval_fn, axis_name="batch")
        
    #     if eval.dataset == "train":
    #         eval_ds =  p_train_iter
    #     elif eval.dataset == "eval":
    #         eval_ds = ul_train_iter
            
            
    #     labels, pred = eval_utils.evaluate_pu(val_ds, state_eta, eta_eval_fn)
    #     acc_PU = (labels == pred).mean()
        
        # return acc_PU, (labels, pred)
        
        
    
    
    