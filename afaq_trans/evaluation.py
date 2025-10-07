import torch.nn as t_nn 
import torch 
from sklearn.metrics import accuracy_score
import jax
import optax 
import jax.numpy as jnp
import numpy as np
from functools import partial
import diffrax
import math


def sample_t(u0, n, t0=0.0, t1=1.0):
    u = (u0 + math.sqrt(2) * jnp.arange(n + 1)) % 1  # Generate n+1 samples and apply modulo 1
    u = u.reshape([-1, 1])                           # Reshape u to a column vector
    return u[:-1] * (t1 - t0) + t0, u[-1]


def get_model_fn(model, params):
    def model_fn(t,x):
        return model.apply(params, t, x)
    return model_fn

class BinaryClassifier(t_nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = t_nn.Sequential(
            t_nn.Linear(512, 256),
            t_nn.BatchNorm1d(256),
            t_nn.Dropout(0.5),
            t_nn.ReLU()
        )
        self.layer2 = t_nn.Sequential(
            t_nn.Linear(256, 128),
            t_nn.BatchNorm1d(128),
            t_nn.Dropout(0.5),
            t_nn.ReLU()
        )
        self.layer3 = t_nn.Linear(128, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.sigmoid(self.layer3(out))
        return out
    
    
def get_generator_one_step(model):
    def artifact_gen(x0, state):
        s = get_model_fn(model, params=state.params_ema)
        dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
        vector_field = lambda _t,_x,_args: dsdx(_t,_x)
        one_step_artifacts = x0 + dsdx(jnp.zeros((x0.shape[0], 1)), x0)
        return one_step_artifacts
    return artifact_gen


def get_generator_ode(model):
    def artifact_gen(x0, state):
        s = get_model_fn(model, params=state.params_ema)
        dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
        vector_field = lambda _t,_x,_args: dsdx(_t,_x)
        solve = partial(diffrax.diffeqsolve, 
                            terms=diffrax.ODETerm(vector_field), 
                            solver=diffrax.Euler(), 
                            t0=0.0, t1=1.0, dt0=1e-3, 
                            saveat=diffrax.SaveAt(ts=[1.0]),
                            stepsize_controller=diffrax.ConstantStepSize(True), 
                            adjoint=diffrax.NoAdjoint())
        
        solution = solve(y0=x0, args=state)
        ode_int_artifacts = solution.ys[-1]
        return ode_int_artifacts
    return artifact_gen


def evaluate_ffhq(config, x, x_sex, x_age, mlp_classifier, target_mlp_classifier):
    D_test = torch.tensor(x)
    
    mlp_classifier.eval()
    pred_labels = mlp_classifier(D_test)
    pred_labels = torch.round(pred_labels.squeeze())
    
    pred_labels_np = pred_labels.data
    
    target_mlp_classifier.eval()
    target_pred_labels = target_mlp_classifier(D_test)
    target_pred_labels = torch.round(target_pred_labels.squeeze())

    target_pred_labels_np = target_pred_labels.data
    
    if config.data.source == 'ffhq_ADULT' or config.data.source == 'ffhq_YOUNG':
        actual_labels_np = np.where(x_sex == b'male', 1, 0)
        if config.data.source == 'ffhq_YOUNG':
            target_actual_labels_np = np.ones(x_sex.shape[0])
        elif config.data.source == 'ffhq_ADULT':
            target_actual_labels_np = np.zeros(x_sex.shape[0])
    elif config.data.source in ['ffhq_MAN', 'ffhq_WOMAN']:
        actual_labels_np = (x_age.reshape(-1) > 44.0)*1
        if config.data.source == 'ffhq_WOMAN':
            target_actual_labels_np = np.ones(x_sex.shape[0])
        elif config.data.source == 'ffhq_MAN':
            target_actual_labels_np = np.zeros(x_sex.shape[0])
    # print("actual_labels_np",actual_labels_np.shape)
    # print("pred_labels_np",pred_labels_np.shape)
    # print("pred_labels_np", pred_labels_np.shape)
    # print("actual_labels_np", actual_labels_np.shape)
    # print("target_pred_labels_np", target_pred_labels_np.shape)
    # print("target_actual_labels_np", target_actual_labels_np.shape)
    
    
    accuracy = accuracy_score(pred_labels_np, actual_labels_np)
    target_accuracy = accuracy_score(target_pred_labels_np, target_actual_labels_np)

    # del (D_test)
    # gc.collect()
    # jax.clear_caches()
    
    return accuracy, target_accuracy
    
def evaluate(config, model, state, dataset, gen_func):
    
    input_shape = (jax.device_count(), config.train.batch_size//jax.device_count(), config.data.dim)
    
    if config.data.source in ["ffhq_MAN", "ffhq_WOMAN", "ffhq_ADULT", "ffhq_YOUNG"]:  
        DEVICE = 'cpu'
          
        gen_outputs = []
        sexes = [] 
        ages = []
        for x0, sex, age in dataset:
            x0 = jnp.reshape(x0.numpy(), input_shape)
            outputs = gen_func(x0, state)
            gen_outputs.append(outputs.reshape(-1,config.data.dim))
            sexes.append(sex.numpy().reshape(-1,))
            ages.append(age.numpy().reshape(-1,))
            
        
        gen_outputs_ = np.vstack(gen_outputs)
        # print(gen_outputs_.shape)
        sexes = np.hstack(sexes)
        # print(sexes.shape)
        ages = np.hstack(ages)
        # print(ages.shape)
        
        if config.data.source in ["ffhq_ADULT", "ffhq_YOUNG"]:
            mlp_classifier = BinaryClassifier()
            mlp_classifier.load_state_dict(torch.load('../classifier_checkpoints/male_female_classifier.pth', map_location=DEVICE, weights_only=True))
            
            target_mlp_classifier = BinaryClassifier()
            target_mlp_classifier.load_state_dict(torch.load('../classifier_checkpoints/young_old_classifier.pth', map_location=DEVICE, weights_only=True))
        
        elif config.data.source in ['ffhq_MAN', 'ffhq_WOMAN']:
            mlp_classifier = BinaryClassifier()
            mlp_classifier.load_state_dict(torch.load('../classifier_checkpoints/young_old_classifier.pth', map_location=DEVICE, weights_only=True))
    
            target_mlp_classifier = BinaryClassifier()
            target_mlp_classifier.load_state_dict(torch.load('../classifier_checkpoints/male_female_classifier.pth', map_location=DEVICE, weights_only=True))        
        
        return evaluate_ffhq(config, gen_outputs_, sexes, ages, mlp_classifier, target_mlp_classifier)
    
    
    
        
        
            
            
            
            
        
        
        
    
    
    
    