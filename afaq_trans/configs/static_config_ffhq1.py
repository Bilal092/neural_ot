import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.batch_size = 1024 

    
    # config.source = "MAN"
    # config.target = "WOMAN"
    
    # config.source = "WOMAN"
    # config.target = "MAN"
    
    # config.source = "YOUNG"
    # config.target = "ADULT"
    
    config.source = "ADULT"
    config.target = "YOUNG"
    
    config.T_lr = 1e-5
    config.f_lr = 1e-5
    config.c = 1.0
    config.T_steps = 5
    config.f_steps = 1
    config.max_steps = 50_000
    config.eval_steps = 1000
    config.save_steps = 1000
    
    return config