In order to run, download code from 
https://github.com/milenagazdieva/LightUnbalancedOptimalTransport
and add path of pretrained classifier and embeddings to evaluation.py and dataloaders.
We only need ALAE embeddings for training. 
See ffhq{1,2,3,4}.qs files to run dyanmic.
and s_ffhq{1,2,3,4}.qs files to run static.
For each dataset config files at ./configs contain experiment configuration.
Configurations for dynamic are obvous from their names, 
whereas for static static_config_ffhq.py corresponds to young to adult, static static_config_ffhq1.py corresponds to adult to young,
static_config_ffhq2.py corresponds to woman to man, static static_config_ffhq3.py corresponds to man to woman.
