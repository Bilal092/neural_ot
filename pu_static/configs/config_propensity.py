from ml_collections import config_dict


data_dict = config_dict.ConfigDict({
    "Abalone":   config_dict.ConfigDict({"n": 4177, "batch_size": 20, "d": 8,   "pi": 0.16}),
    "Banknote":  config_dict.ConfigDict({"n": 1372, "batch_size": 10, "d": 4,   "pi": 0.44}),
    "Breast-w":  config_dict.ConfigDict({"n": 699,  "batch_size": 10, "d": 9,   "pi": 0.34}),
    "Diabetes":  config_dict.ConfigDict({"n": 768,  "batch_size": 6,  "d": 8,   "pi": 0.35}),
    "Haberman":  config_dict.ConfigDict({"n": 306,  "batch_size": 6,  "d": 3,   "pi": 0.26}),
    "Heart":     config_dict.ConfigDict({"n": 270,  "batch_size": 6,  "d": 13,  "pi": 0.44}),
    "Ionosphere":config_dict.ConfigDict({"n": 351,  "batch_size": 6,  "d": 34,  "pi": 0.64}),
    "Isolet":    config_dict.ConfigDict({"n": 7797, "batch_size": 4, "d": 617, "pi": 0.04}),
    "Jm1":       config_dict.ConfigDict({"n": 10885,"batch_size": 20, "d": 21,  "pi": 0.19}),
    "Kc1":       config_dict.ConfigDict({"n": 2109, "batch_size": 20, "d": 21,  "pi": 0.15}),
    "Madelon":   config_dict.ConfigDict({"n": 2600, "batch_size": 20, "d": 500, "pi": 0.50}),
    "Musk":      config_dict.ConfigDict({"n": 6598, "batch_size": 20, "d": 166, "pi": 0.15}),
    "Segment":   config_dict.ConfigDict({"n": 2310, "batch_size": 20, "d": 19,  "pi": 0.14}),
    "Semeion":   config_dict.ConfigDict({"n": 1593, "batch_size": 4, "d": 256, "pi": 0.10}),
    "Sonar":     config_dict.ConfigDict({"n": 208,  "batch_size": 4,  "d": 60,  "pi": 0.53}),
    "Spambase_n":  config_dict.ConfigDict({"n": 4601, "batch_size": 20, "d": 57,  "pi": 0.39}),
    "Vehicle":   config_dict.ConfigDict({"n": 846,  "batch_size": 6,  "d": 18,  "pi": 0.26}),
    "Waveform":  config_dict.ConfigDict({"n": 5000, "batch_size": 20, "d": 40,  "pi": 0.34}),
    "Wdbc":      config_dict.ConfigDict({"n": 569,  "batch_size": 6,  "d": 30,  "pi": 0.37}),
    "Yeast":     config_dict.ConfigDict({"n": 1484, "batch_size": 10, "d": 8,   "pi": 0.31}),
})


c_ph = config_dict.FieldReference(1.0)
dataset_placeholder = config_dict.FieldReference("enter valid data set")

def get_config(dataset_placeholder):
    config = config_dict.ConfigDict()
    config.seed = 0
    config.dataset = dataset_placeholder
    config.prior = data_dict[dataset_placeholder]["pi"]
    config.c = 1/config.prior
    
    config.data = data =  config_dict.ConfigDict()
    data.dataset = dataset_placeholder
    data.source = dataset_placeholder
    data.target = dataset_placeholder
    data.uniform_dequantization = False
    # these configurations are used when train.mode = "ul_batch" 
    data.size_p = 400
    data.size_ul = 800
    data.size_u_val = 800
    # batch_size configuration is used in both cases it should be divisble with jax.local_device_count()
    data.batch_size = data_dict[dataset_placeholder]["batch_size"] 
    data.dim = data_dict[dataset_placeholder]["d"]
    data.additional_dim = 1
    # these configurations are used when train.model = ul_dataset
    data.train_ratio = 0.9
    data.eval_ratio = 0.1 # in order to run evaluation on whole data set it equal to 1.0
    # this configuration is used in propensity based models
    data.test_size = 0.25
    
    
    config.model_T = model_T = config_dict.ConfigDict()
    model_T.embedding_dim = 128
    model_T.num_hid = 1024
    model_T.num_out = data_dict[dataset_placeholder]["d"]
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
    train.mode = "propensity" # other modes are "ul_batch and ul_dataset"
    train.eval_interval_steps = 1_000
    train.save_interval_steps = 1_000
    train.log_interval_steps = 1_000
    train.batch_size = data.batch_size
    train.label_strat = 'S1'
    train.num_test = 20
    
    config.eval = eval = config_dict.ConfigDict()
    eval.dt = 1e-2
    eval.step = 500_000
    eval.dataset = "train"
    
    
    return config




# def get_config():
#     config = config_dict.ConfigDict()
#     config.c = c_ph
#     config.seed = 0
#     config.dataset = "Abalone"
#     config.prior = data_dict[config.dataset]["pi"]
    
#     config.data = data =  config_dict.ConfigDict()
#     data.dataset = "Abalone"
#     data.source = "Abalone"
#     data.target = "Abalone"
#     data.uniform_dequantization = False
#     # these configurations are used when train.mode = "ul_batch" 
#     data.size_p = 400
#     data.size_ul = 800
#     data.size_u_val = 800
#     # batch_size configuration is used in both cases it should be divisble with jax.local_device_count()
#     data.batch_size = 20 
#     data.dim = data_dict[config.dataset]["d"]
#     data.additional_dim = 1
#     # these configurations are used when train.model = ul_dataset
#     data.train_ratio = 0.9
#     data.eval_ratio = 0.1 # in order to run evaluation on whole data set it equal to 1.0
#     # these settings are used for train.model = propensity 
    
    
#     config.model_s = model_s = config_dict.ConfigDict()
#     model_s.embedding_dim = 128
#     model_s.num_hid = 1024
#     model_s.num_out = 1
#     model_s.ema_rate = 0.999
    
#     config.optimizer_s = optimizer_s = config_dict.ConfigDict()
#     optimizer_s.lr = 1e-5
#     # optimizer_s.lr = 0.5e-5
    
#     config.optimizer_q = optimizer_q = config_dict.ConfigDict()
#     optimizer_q.lr = 1e-5
#     # optimizer_q.lr = 0.5e-5
    
#     config.model_q = model_q = config_dict.ConfigDict()
#     model_q.embedding_dim = 128
#     model_q.num_hid = 1024
#     model_q.num_out = data_dict[config.dataset]["d"]
#     model_q.ema_rate = 0.999
    
    
#     config.train = train = config_dict.ConfigDict()
#     train.wgf_steps = 0
#     train.wgf_step_size = 1e-3
#     train.num_train_steps = 20_000
#     train.n_jitted_steps = 1
#     train.dt=1e-2
#     train.num_s_steps = 2
#     train.num_q_steps = 1
#     train.mode = "propensity"  # other mode is "ul_dataset"
#     train.eval_interval_steps = 1_000
#     train.save_interval_steps = 1_000
#     train.log_interval_steps = 1_000
#     train.batch_size = data.batch_size
#     train.label_strat = 'S1'
#     train.num_test = 20
    
    
#     config.eval = eval = config_dict.ConfigDict()
#     eval.dt = 1e-2
#     eval.step = 500_000
#     eval.dataset = "train"
    
#     config.phi_sampler = phi_sampler = config_dict.ConfigDict()
#     phi_sampler.refresh_every = 500
#     phi_sampler.min_update_step = 5_000
#     phi_sampler.candidate_batch_size=20
#     phi_sampler.min_acceptance_ratio = 1.0/ (config.c - config.c/10)
    
    
#     return config