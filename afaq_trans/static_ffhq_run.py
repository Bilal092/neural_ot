import os, sys
sys.path.append("..")
sys.path.append("../ALAE")

import torch
import numpy as np
import ot

from src.distributions import LoaderSampler, TensorSampler
from src.ulight_ot import ULightOT
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.optim.swa_utils import AveragedModel, update_bn

import wandb
import matplotlib
from matplotlib import pyplot as plt
# from torch.optim.lr_scheduler import MultiStepLR
from IPython.display import clear_output
from torch.optim.swa_utils import AveragedModel, update_bn, get_ema_multi_avg_fn


# from ALAE.alae_ffhq_inference import load_model, encode, decode


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"

num_gpus = torch.cuda.device_count()

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import Parameter
# import torch.optim.lr_scheduler as lr_scheduler
# import torchvision.transforms as transforms



from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
flags.DEFINE_float("c", 1.0, "subsetting scalar")


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(128, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.sigmoid(self.layer3(out))
        return out


class TransportNET(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TransportNET, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, input_dim))
        self.ResConnect = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        output = self.MLP(inputs) + self.ResConnect(inputs)
        return output

class DualNET(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualNET, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                nn.SiLU(),
                                 nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        return self.MLP(x)
    

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        m.bias.data.fill_(2.0)

    elif type(m) == nn.BatchNorm1d:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(2.0)


def train(argv):
    if isinstance(FLAGS.c, float or int):
        if FLAGS.c >=1.0:
          FLAGS.config.c = FLAGS.c
        else:
          raise ValueError(f"c must be float or int >=1.0.")
    else:
        raise TypeError(f"c must be float or scalar >= 1.0.")
    
    # FLAGS.config, FLAGS.workdir, c
    
    config = FLAGS.config
    workdir = FLAGS.workdir
    
    BATCH_SIZE = config.batch_size
    source = config.source
    target = config.target
    max_iter = config.max_steps
    T_STEPS = config.T_steps
    f_STEPS = config.f_steps
    T_lr = config.T_lr 
    f_lr = config.f_lr
    EVAL_STEPS = config.eval_steps
    SAVE_STEPS = config.save_steps
    c = FLAGS.c
    
    checkpoint_dir = os.path.join(workdir, "static_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True) 
    
    INPUT_DATA = source 
    TARGET_DATA =  target

    # To download data use
    train_size = 60000
    test_size = 10000

    latents = np.load("/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/latents.npy")
    gender = np.load("/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/gender.npy")
    age = np.load("/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/age.npy")
    test_inp_images = np.load("/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/data/test_images.npy")

    train_latents, test_latents = latents[:train_size], latents[train_size:]
    train_gender, test_gender = gender[:train_size], gender[train_size:]
    train_age, test_age = age[:train_size], age[train_size:]

        
    if INPUT_DATA == "MAN":
        x_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif INPUT_DATA == "WOMAN":
        x_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif INPUT_DATA == "ADULT":
        x_inds_train = np.arange(train_size)[
            (train_age > 44).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            (test_age > 44).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif INPUT_DATA == "YOUNG":
        x_inds_train = np.arange(train_size)[
            ((train_age > 16) & (train_age <= 44)).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            ((test_age > 16) & (test_age <= 44)).reshape(-1)*(test_age != -1).reshape(-1)
        ]

    if TARGET_DATA == "MAN":
        y_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif TARGET_DATA == "WOMAN":
        y_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif TARGET_DATA == "ADULT":
        y_inds_train = np.arange(train_size)[
            (train_age > 44).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            (test_age > 44).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif TARGET_DATA == "ADULT-MAN":
        male_train = np.arange(train_size)[(train_gender == "male")]
        male_test = np.arange(test_size)[(test_gender == "male")]
        
        y_inds_train = male_train[(train_age[male_train].reshape(-1) > 44)]#*(train_age != -1)
        y_inds_test = male_test[(test_age[male_test].reshape(-1) > 44)]#*(test_age != -1).reshape(-1)
    elif TARGET_DATA == "YOUNG":
        y_inds_train = np.arange(train_size)[
            ((train_age > 16) & (train_age <= 44)).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            ((test_age > 16) & (test_age <= 44)).reshape(-1)*(test_age != -1).reshape(-1)
        ]
        
    if INPUT_DATA in ['ADULT', 'YOUNG']:
        mlp_classifier = BinaryClassifier()
        mlp_classifier.load_state_dict(torch.load('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/classifier_checkpoints/male_female_classifier.pth', map_location=device))
        
        target_mlp_classifier = BinaryClassifier()
        target_mlp_classifier.load_state_dict(torch.load('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/classifier_checkpoints/young_old_classifier.pth', map_location=device))
    elif INPUT_DATA in ['MAN', 'WOMAN']:
        mlp_classifier = BinaryClassifier()
        mlp_classifier.load_state_dict(torch.load('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/classifier_checkpoints/young_old_classifier.pth', map_location=device))
        
        target_mlp_classifier = BinaryClassifier()
        target_mlp_classifier.load_state_dict(torch.load('/lustre/cniel/neural-ot-ss/lagrangian-pot-flows/subset_selection_embeddings/classifier_checkpoints/male_female_classifier.pth', map_location=device))

    x_data_train = train_latents[x_inds_train]
    x_data_test = test_latents[x_inds_test]
    x_data_test_gender = test_gender[x_inds_test]
    x_data_test_age = test_age[x_inds_test]

    inds_to_map = np.random.choice(np.arange((x_inds_test < 300).sum()), size=10, replace=False)
    number_of_samples = 1
    mapped_all = []
    latent_to_map = torch.tensor(test_latents[x_inds_test[inds_to_map]])
    inp_images = test_inp_images[x_inds_test[inds_to_map]]
        
    y_data_train = train_latents[y_inds_train]
    y_data_test = test_latents[y_inds_test]

    X_train = torch.tensor(x_data_train)
    Y_train = torch.tensor(y_data_train)

    X_test = torch.tensor(x_data_test)
    Y_test = torch.tensor(y_data_test)

    X_sampler = TensorSampler(X_train, device=device)
    Y_sampler = TensorSampler(Y_train, device=device)
    
    INPUT_DATA = source 
    TARGET_DATA =  target
    
    beta_ema = 0.999

    netT = TransportNET(512, 1024).to(device)
    netf = DualNET(512, 1024).to(device)
    
    if num_gpus > 1:
       netT = nn.DataParallel(netT)
       netf = nn.DataParallel(netf)
       
    ema_netT = AveragedModel(netT, multi_avg_fn=get_ema_multi_avg_fn(beta_ema))
    ema_netT = ema_netT.to(device)
    ema_netf = AveragedModel(netf, multi_avg_fn=get_ema_multi_avg_fn(beta_ema))
    ema_netf = ema_netf.to(device)

    # ema_netT = AveragedModel(netT, avg_fn=lambda avg_p, new_p, num_avg: beta_ema * avg_p + (1 - beta_ema) * new_p)
    # ema_netT = ema_netT.to(device)
    # ema_netf = AveragedModel(netf, avg_fn=lambda avg_p, new_p, num_avg: beta_ema * avg_p + (1 - beta_ema) * new_p).to(device)
    # ema_netf = ema_netf.to(device)
    
    optimizerT = optim.AdamW(netT.parameters(), lr = T_lr,  weight_decay=1e-8)
    optimizerf = optim.AdamW(netf.parameters(), lr = f_lr,  weight_decay=1e-8)

    ReLU = nn.ReLU()

    def sq_euclidean_cost(x,y):
            return torch.norm(x - y, dim=-1)**2


    EXPERIMENT_NAME = "ffhq_" + source + "_" + target 
    SAVE_PATH = os.path.join(checkpoint_dir, EXPERIMENT_NAME,  "c={}".format(c))
    os.makedirs(SAVE_PATH, exist_ok=True)

    wandb.init(name=EXPERIMENT_NAME + "_c={}".format(c), project="static_ss_" + EXPERIMENT_NAME)

    X_test = X_test.to(device)
    mlp_classifier = mlp_classifier.to(device)
    target_mlp_classifier = target_mlp_classifier.to(device)
    
    best_target_accuracy = 0
    
    for step in range(0, max_iter+1):
        for iter_theta in range(0, T_STEPS):

            xk = X_sampler.sample(BATCH_SIZE)
            yk = Y_sampler.sample(BATCH_SIZE)
            
            netT.zero_grad()
            netf.zero_grad()
            netT.train()
        
            for params in netT.parameters():
                params.requires_grad = True
            
            for params in netf.parameters():
                params.requires_grad = False
            
            netT_xk = netT(xk)
            xk_hat = netT_xk
            monge_cost = sq_euclidean_cost(xk, xk_hat).mean()
            lagrange_cost0 = netf(xk_hat).mean()
            theta_cost = (monge_cost + lagrange_cost0)
            
            theta_cost.backward()
            optimizerT.step()
            
            ema_netT.update_parameters(netT)
            
        
        for iter_eta in range(0, f_STEPS):
            netT.zero_grad()
            netf.zero_grad()
            netf.train()
                    
            for params in netT.parameters():
                params.requires_grad = False
            
            for params in netf.parameters():
                params.requires_grad = True
            
            netT_xk = netT(xk)
            xk_hat = netT_xk
            lagrange_cost0 = netf(xk_hat).mean()
            lagrange_cost1 = c*ReLU(netf(yk)).mean()
            lagrange_cost = -(lagrange_cost0 - lagrange_cost1)
            
            lagrange_cost.backward()
            optimizerf.step()
            
            ema_netf.update_parameters(netf)

        if step % EVAL_STEPS == 0:
            ema_netT.eval()
            D_test = ema_netT(X_test)
            mlp_classifier.eval()
            pred_labels = mlp_classifier(D_test)
            pred_labels = torch.round(pred_labels.squeeze())
            
            pred_labels_np = pred_labels.data
            
            target_mlp_classifier.eval()
            target_pred_labels = target_mlp_classifier(D_test)
            target_pred_labels = torch.round(target_pred_labels.squeeze())
            
            target_pred_labels_np = target_pred_labels.data
            
            if INPUT_DATA == 'ADULT' or INPUT_DATA == 'YOUNG':
                actual_labels_np = np.where(x_data_test_gender == 'male', 1, 0)
                if INPUT_DATA == 'YOUNG':
                    target_actual_labels_np = np.ones(x_data_test_gender.shape[0])
                elif INPUT_DATA == 'ADULT':
                    target_actual_labels_np = np.zeros(x_data_test_gender.shape[0])
            elif INPUT_DATA in ['MAN', 'WOMAN']:
                actual_labels_np = (x_data_test_age.reshape(-1) > 44)*1
                if INPUT_DATA == 'WOMAN':
                    target_actual_labels_np = np.ones(x_data_test_gender.shape[0])
                elif INPUT_DATA == 'MAN':
                    target_actual_labels_np = np.zeros(x_data_test_gender.shape[0])
            accuracy = accuracy_score(pred_labels_np.cpu().numpy(), actual_labels_np)
            target_accuracy = accuracy_score(target_pred_labels_np.cpu().numpy(), target_actual_labels_np)
            wandb.log(dict(accuracy=100*accuracy, target_accuracy=100*target_accuracy), step)
            wandb.log(dict(monge_cost=monge_cost.detach().cpu().numpy().item()), step=step)
            wandb.log(dict(lagrange_cost=lagrange_cost.detach().cpu().numpy().item()), step=step)
        
        if step % SAVE_STEPS == 0: 
            checkpoint = {"netT_dict": netT.state_dict(), "netf_dict": netf.state_dict(), 
                          "ema_netT_dict":ema_netT.state_dict(), "ema_netf_dict":ema_netf.state_dict()}
            torch.save(checkpoint, os.path.join(SAVE_PATH, "checkpoint_{}.pth".format(step)))
            
            # if target_accuracy >= best_target_accuracy:
            #     best_target_accuracy = target_accuracy 
            #     checkpoint = {"netT_dict": netT.state_dict(), "netf_dict": netf.state_dict(), 
            #               "ema_netT_dict":ema_netT.state_dict(), "ema_netf_dict":ema_netf.state_dict()}
            #     #  "netT_ema":ema.get_model(beta).state_dict() 
            #     torch.save(checkpoint, os.path.join(SAVE_PATH, "checkpoint_best.pth"))
        

if __name__ == "__main__":
  app.run(train)


