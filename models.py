import torch.nn as nn
from argparse import Namespace
import os.path
from contin_vae.models import VAE
from contin_infomax.models import InfoMax
from agents.utils import *


def cnn_base(ndim, img_stack, act_func=nn.ReLU(), freeze_w=False):
    cnn = nn.Sequential(  # input shape (4, 96, 96)
        nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
        act_func,  # activation
        nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
        act_func,  # activation
        nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
        act_func,  # activation
        nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
        act_func,  # activation
        nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
        act_func,  # activation
        nn.Conv2d(128, ndim, kernel_size=3, stride=1),  # (128, 3, 3)
        act_func,  # activation
        Flatten()
    )
    cnn.apply(weights_init_xavier)
    if freeze_w:
        print("Freezing ConvNet weights")
        freeze_weights(cnn)
    return cnn


def v_base(ndim, action_vec=0, act_func=nn.ReLU()):
    v = nn.Sequential(nn.Linear(ndim + action_vec * 3, 100), act_func, nn.Linear(100, 1))
    v.apply(weights_init_xavier)
    return v


def fc_base(ndim, action_vec=0, act_func_lin=nn.ReLU()):
    fc = nn.Sequential(nn.Linear(ndim + action_vec * 3, 100), act_func_lin)
    fc.apply(weights_init_xavier)
    return fc


def ddpg_base(ndim, action_vec=0, act_func_lin_1=nn.ReLU(), act_func_lin_2=nn.Tanh()):
    # Action Vector implicitly not used for actor in DDPG by takig away action vec
    fc_1 = nn.Sequential(nn.Linear(ndim + action_vec * 3, 30),
                       act_func_lin_1)
    fc_1.apply(weights_init_xavier)
    fc_2 =  nn.Sequential(nn.Linear(30, 3),
    act_func_lin_2)
    fc_2.apply(weights_init_xavier_tanh)
    fc = nn.Sequential(*fc_1.children(), *fc_2.children())
    return fc


def alpha_beta_head(ndim=100, act_func=nn.Softplus()):
    alpha_head = nn.Sequential(nn.Linear(ndim, 3), act_func)
    beta_head = nn.Sequential(nn.Linear(ndim, 3), act_func)
    return alpha_head, beta_head


def vae_base(vae_path="contin_vae/pretrained_vae_64_stack4_conti.ckpt", ndim=64, img_stack=4, freeze_w=False,
             device="cuda"):
    if vae_path:
        assert os.path.isfile(vae_path)
        hparams = Namespace(**torch.load(vae_path)["hparams"])
        state_dict = torch.load(vae_path)["state_dict"]
        assert hparams.ndim == ndim
        assert hparams.img_stack == img_stack
        vae = VAE(hparams).to(device)
        vae.load_state_dict(state_dict)
        if freeze_w:
            print("Freezing VAE weights")
            freeze_weights(vae)
        return vae
    else:
        # Todo: Initialize VAE for training
        return None


def infomax_base(infomax_path="contin_infomax/pretrained_infomax_64_stack4_action_conti.ckpt", ndim=64, freeze_w=False,
                 device="cuda"):
    if infomax_path:
        assert os.path.isfile(infomax_path)
        hparams = Namespace(**torch.load(infomax_path)["hparams"])
        state_dict = torch.load(infomax_path)["state_dict"]
        assert hparams.ndim == ndim
        infomax = InfoMax(hparams).to(device)
        infomax.load_state_dict(state_dict)
        if freeze_w:
            print("Freezing InfoMax weights")
            freeze_weights(infomax)
        return infomax
    else:
        # Todo: Initialize infomax for training
        return None
    pass
