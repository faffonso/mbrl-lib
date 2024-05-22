# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
from collections import OrderedDict
import torch
from torch import nn as nn
from torch.nn import functional as F

import mbrl.util.math

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init

class DeepKoopman(Ensemble):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        act_size: int,
        affine: bool,
        device: Union[str, torch.device],
        enc_size: int = 12,
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,        
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,   
        verbose: bool = False   
    ):

        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        koopman =  enc_size + out_size

        self.enc_size = enc_size
        self.in_size  = in_size
        self.out_size = out_size
        self.act_size = act_size
        self.verbose  = verbose
        self.koopman = koopman

        enc_layers  = [out_size]+[hid_size]*num_layers+[enc_size]
        bic_layers  = [out_size]+[hid_size]*num_layers+[act_size]

    
        ELayers = OrderedDict()
        for layer_i in range(len(enc_layers)-1):
            ELayers["linear_{}".format(layer_i)] = nn.Linear(enc_layers[layer_i],enc_layers[layer_i+1])
            if layer_i != len(enc_layers)-2:
                ELayers["relu_{}".format(layer_i)] = nn.ReLU()
        
        BLayers = OrderedDict()
        for layer_i in range(len(bic_layers)-1):
            BLayers["linear_{}".format(layer_i)] = nn.Linear(bic_layers[layer_i],bic_layers[layer_i+1])
            if layer_i != len(bic_layers)-2:
                BLayers["relu_{}".format(layer_i)] = nn.ReLU()

        self.encode_net = nn.Sequential(ELayers)
        self.bilinear_net = nn.Sequential(BLayers)         

        self.lA = nn.Linear(koopman,koopman,bias=False)
        self.lA.weight.data = self._gaussian_init_(koopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9

        self.lB = nn.Linear(bic_layers[-1],koopman,bias=False)

        self.to(self.device)

        if verbose:
            print("Deep Koopmam network")
            print(ELayers)
            print(BLayers)
            print(self)

    # Encode state information [x_t => x_t; z_t]
    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)
    
    # Bicode affine function [x_t, u_t => ü_t]
    def bicode(self, x, u):
        return (self.bilinear_net(x) * u)
    
    # Forward method using linear state space [z_{t+1} => A * z_t + B * ü_t]
    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(use_propagation)

        return self._default_forward(x)

    def _default_forward(self, model_in):
        x_state = model_in[..., :self.out_size]
        x_act = model_in[..., self.out_size:]

        dim = x_act.dim()

        print(model_in.device)

        if (dim == 2):
            Z = torch.empty(x_act.shape[0], self.out_size, dtype=model_in.dtype, device=model_in.device)
        elif (dim == 3):
            Z = torch.empty(self.num_members, x_act.shape[1], self.out_size, dtype=model_in.dtype, device=model_in.device)


        z_k = self.encode(x_state[0, :])

        for i, act in enumerate(x_act):
            u_k = self.bicode(z_k[..., :self.out_size], act)
            z_k = self.lA(z_k) + self.lB(u_k.detach())  # Detach u_k here


            Z[i] = z_k[..., :self.out_size]

        return Z, None

    def _gaussian_init_(self, n_units, std=1):    
        sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
        omega = sampler.sample((n_units, n_units))[..., 0]  
        return omega

    # TODO: 
    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum(), {}
    
    # def Klinear_loss(data,net,mse_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0,detach=0):
    # steps,train_traj_num,NKoopman = data.shape
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = torch.DoubleTensor(data).to(device)
    # X_current = net.encode(data[0,:,u_dim:])
    # beta = 1.0
    # beta_sum = 0.0
    # loss = torch.zeros(1,dtype=torch.float64).to(device)
    # Augloss = torch.zeros(1,dtype=torch.float64).to(device)
    # for i in range(steps-1):
    #     bilinear = net.bicode(X_current[:,:Nstate].detach(),data[i,:,:u_dim]) #detach's problem 
    #     X_current = net.forward(X_current,bilinear)
    #     beta_sum += beta
    #     if not all_loss:
    #         loss += beta*mse_loss(X_current[:,:Nstate],data[i+1,:,u_dim:])
    #     else:
    #         Y = net.encode(data[i+1,:,u_dim:])
    #         loss += beta*mse_loss(X_current,Y)
    #     X_current_encoded = net.encode(X_current[:,:Nstate])
    #     Augloss += mse_loss(X_current_encoded,X_current)
    #     beta *= gamma
    # Augloss = Augloss/beta_sum
    # return loss+0.5*Augloss

    # TODO:
    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            #target = target.repeat((self.num_members, 1, 1))

            return F.mse_loss(pred_mean, target, reduction="none"), {}
    
    # TODO:
    def sample_propagation_indices(
        self, batch_size: int, rng: torch.Generator
    ) -> torch.Tensor:
        pass