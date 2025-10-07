from ml_collections import config_dict


c_ph = config_dict.FieldReference(1.0)
def get_config():
    config = config_dict.ConfigDict()
    config.c = c_ph
    config.seed = 0
    config.dataset = "spambase"
    config.prior = 0.394
    
    config.data = data =  config_dict.ConfigDict()
    data.dataset = "spambase"
    data.source = "spambase"
    data.target = "spambase"
    data.uniform_dequantization = False
    # these configurations are used when train.mode = "ul_batch" 
    data.size_p = 400
    data.size_ul = 800
    data.size_u_val = 800
    # batch_size configuration is used in both cases it should be divisble with jax.local_device_count()
    data.batch_size = 100 
    data.dim = 57
    data.additional_dim = 1
    # these configurations are used when train.model = ul_dataset
    data.train_ratio = 0.9
    data.eval_ratio = 0.1 # in order to run evaluation on whole data set it equal to 1.0
    
    
    config.model_T = model_T = config_dict.ConfigDict()
    model_T.embedding_dim = 128
    model_T.num_hid = 1024
    model_T.num_out = 57
    model_T.ema_rate = 0.999
    
    config.optimizer_T = optimizer_T = config_dict.ConfigDict()
    optimizer_T.lr = 1e-4
    # optimizer_T.lr = 0.5e-5
    
    config.optimizer_eta = optimizer_eta = config_dict.ConfigDict()
    optimizer_eta.lr = 1e-4
    # optimizer_eta.lr = 0.5e-5
    
    config.model_eta = model_eta = config_dict.ConfigDict()
    model_eta.embedding_dim = 128
    model_eta.num_hid = 1024
    model_eta.num_out = 1
    model_eta.ema_rate = 0.999
    
    
    config.train = train = config_dict.ConfigDict()
    train.wgf_steps = 0
    train.wgf_step_size = 1e-3
    train.num_train_steps = 20_000
    train.n_jitted_steps = 1
    train.dt=1e-2
    train.num_T_steps = 10
    train.num_eta_steps = 1
    train.mode = "ul_batch"  # other mode is "ul_dataset"
    train.eval_interval_steps = 100
    train.save_interval_steps = 1_000
    train.log_interval_steps = 100
    
    config.eval = eval = config_dict.ConfigDict()
    eval.dt = 1e-2
    eval.step = 500_000
    eval.dataset = "train"
    
    
    return config