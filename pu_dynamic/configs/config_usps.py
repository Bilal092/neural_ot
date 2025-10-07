from ml_collections import config_dict


c_ph = config_dict.FieldReference(1.0)
def get_config():
    config = config_dict.ConfigDict()
    config.c = c_ph
    config.seed = 0
    config.dataset = "usps"
    config.prior = 0.167
    
    config.data = data =  config_dict.ConfigDict()
    data.dataset = "usps"
    data.source = "usps"
    data.target = "usps"
    data.uniform_dequantization = False
    # these configurations are used when train.mode = "ul_batch" 
    data.size_p = 400
    data.size_ul = 800
    data.size_u_val = 800
    # batch_size configuration is used in both cases it should be divisble with jax.local_device_count()
    data.batch_size = 20 
    data.dim = 256
    data.additional_dim = 1
    # these configurations are used when train.model = ul_dataset
    data.train_ratio = 0.9
    data.eval_ratio = 0.1 # in order to run evaluation on whole data set it equal to 1.0
    
    
    config.model_s = model_s = config_dict.ConfigDict()
    model_s.embedding_dim = 128
    model_s.num_hid = 1024
    model_s.num_out = 1
    model_s.ema_rate = 0.999
    
    config.optimizer_s = optimizer_s = config_dict.ConfigDict()
    optimizer_s.lr = 1e-5
    # optimizer_s.lr = 0.5e-5
    
    config.optimizer_q = optimizer_q = config_dict.ConfigDict()
    optimizer_q.lr = 1e-5
    # optimizer_q.lr = 0.5e-5
    
    config.model_q = model_q = config_dict.ConfigDict()
    model_q.embedding_dim = 128
    model_q.num_hid = 1024
    model_q.num_out = 256
    model_q.ema_rate = 0.999
    
    
    config.train = train = config_dict.ConfigDict()
    train.wgf_steps = 0
    train.wgf_step_size = 1e-3
    train.num_train_steps = 20_000
    train.n_jitted_steps = 1
    train.dt=1e-2
    train.num_s_steps = 2
    train.num_q_steps = 1
    train.mode = "ul_batch"  # other mode is "ul_dataset"
    train.eval_interval_steps = 1_000
    train.save_interval_steps = 1_000
    train.log_interval_steps = 1_000
    train.batch_size = data.batch_size
    
    config.eval = eval = config_dict.ConfigDict()
    eval.dt = 1e-2
    eval.step = 500_000
    eval.dataset = "train"
    
    config.phi_sampler = phi_sampler = config_dict.ConfigDict()
    phi_sampler.refresh_every = 500
    phi_sampler.min_update_step = 5_000
    phi_sampler.candidate_batch_size=20
    phi_sampler.min_acceptance_ratio = 1.0/ (config.c - config.c/10)
    
    
    return config