import ml_collections


c_placeholder = ml_collections.FieldReference(1.0)
def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.c = c_placeholder
    
    config.data = data = ml_collections.ConfigDict()
    data.source = "ffhq_MAN"
    data.target = "ffhq_WOMAN"
    
    # data.source = "ffhq_WOMAN"
    # data.target = "ffhq_MAN"
    
    # data.source = "ffhq_YOUNG"
    # data.target = "ffhq_ADULT"
    
    # data.source = "ffhq_ADULT"
    # data.target = "ffhq_YOUNG"
    

    data.additional_dim = 1
    data.dim = 512
    data.uniform_dequantization = False
    
    config.model_s = model_s = ml_collections.ConfigDict()
    model_s.num_hid = 1024
    model_s.num_out = 1
    model_s.ema_rate = 0.999
    model_s.t_embed_dim = 128
    
    config.model_q = model_q = ml_collections.ConfigDict()
    model_q.num_hid = 1024
    model_q.num_out = 512
    model_q.ema_rate = 0.999
    model_q.t_embed_dim = 128
    
    config.optimizer_s = optimizer_s = ml_collections.ConfigDict()
    optimizer_s.lr = 1e-5
    # optimizer_s.lr = 0.5e-4
    optimizer_s.grad_clip = 1.0

    config.optimizer_q = optimizer_q = ml_collections.ConfigDict()
    optimizer_q.lr = 1e-5
    # optimizer_q.lr = 0.5e-4
    optimizer_q.grad_clip = 1.0

    config.train = train = ml_collections.ConfigDict()
    config.train.num_train_steps = 500_000
    # train.num_s_steps = 1
    # train.num_q_steps = 5
    train.num_s_steps = 2
    train.num_q_steps = 1
    train.batch_size = 128
    train.save_interval_steps = 5000
    train.eval_interval_steps = 5000
    train.wgf_steps = 0
    train.wgf_step_size = 1e-2
    config.train.n_jitted_steps = 1

    config.phi_sampler = phi_sampler = ml_collections.ConfigDict()
    phi_sampler.refresh_every = 1000
    phi_sampler.min_update_step = 1000
    phi_sampler.candidate_batch_size=train.batch_size
    phi_sampler.min_acceptance_ratio = 1.0/ (c_placeholder.get() -0.1)
    
    return config



