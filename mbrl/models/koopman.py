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
        device: Union[str, torch.device],
        enc_size: int = 12,
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,        
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,   
        verbose: bool = False   
    ):

        super().__init__(
            ensemble_size, device, propagation_method, deterministic=True
        )

        koopman =  enc_size + out_size

        self.enc_size = enc_size
        self.in_size  = in_size
        self.out_size = out_size
        self.act_size = act_size
        self.verbose  = verbose
        self.koopman = koopman

        self.elite_models: List[int] = None


        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        enc_layers  = [out_size]+[hid_size]*num_layers+[enc_size]
        bic_layers  = [out_size]+[hid_size]*num_layers+[act_size]

        ELayers = OrderedDict()
        for layer_i in range(len(enc_layers)-1):
            ELayers["linear_{}".format(layer_i)] = create_linear_layer(enc_layers[layer_i],enc_layers[layer_i+1])
            if layer_i != len(enc_layers)-2:
                ELayers["relu_{}".format(layer_i)] = create_activation()
        
        BLayers = OrderedDict()
        for layer_i in range(len(bic_layers)-1):
            BLayers["linear_{}".format(layer_i)] = create_linear_layer(bic_layers[layer_i],bic_layers[layer_i+1])
            if layer_i != len(bic_layers)-2:
                BLayers["relu_{}".format(layer_i)] = create_activation()

        self.encode_net = nn.Sequential(ELayers)
        self.bilinear_net = nn.Sequential(BLayers)         

        self.lA = create_linear_layer(koopman,koopman)
        self.lA.weight.data = self._gaussian_init_(koopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9

        self.lB = create_linear_layer(bic_layers[-1],koopman)

        self.to(self.device)

        if verbose:
            print("Deep Koopmam network")
            print(ELayers)
            print(BLayers)
            print(self)

    # Encode state information [x_t => x_t; z_t]
    def encode(self, x, dim):
        if (dim == 2):
            x_aux = x[None, None, :] 
            x_aux = x_aux.repeat((self.num_members, 1, 1))
        elif (dim == 3):
            x_aux = x.unsqueeze(1)

        return torch.cat([x_aux, self.encode_net(x_aux)], axis=-1)
    
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
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _default_forward(self, model_in):
        dim = model_in.dim()
        x_state = model_in[..., :self.out_size]
        x_act = model_in[..., self.out_size:]

        dim = x_state.dim()
        
        X_shape = (self.num_members, x_act.shape[0], self.out_size) if dim == 2 else (self.num_members, x_act.shape[1], self.out_size)
        X = torch.empty(*X_shape, device=model_in.device)

        z_k = self.encode(x_state[:, 0, :] if dim == 3 else x_state[0, :], dim)

        for i, act in enumerate(x_act):
            u_k = self.bicode(z_k[..., :self.out_size], act)
            z_k = self.lA(z_k) + self.lB(u_k.detach())  # Detach u_k here

            X[:, i, :] = z_k[:, 0,  :self.out_size]

        return X, None

    # FIXME:
    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)

        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    # FIXME:
    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    # FIXME:
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
        return F.mse_loss(pred_mean, target, reduction="none").sum(), {}

    # TODO:
    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert model_in.ndim == 2 and target.ndim == 2


        with torch.no_grad():
            target = target.repeat((self.num_members, 1, 1))
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            return F.mse_loss(pred_mean, target, reduction="none"), {}
    
    # TODO:
    def sample_propagation_indices(
        self, batch_size: int, rng: torch.Generator
    ) -> torch.Tensor:
        pass