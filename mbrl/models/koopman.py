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
            ensemble_size, device, propagation_method, deterministic=False
        )

        self.debug_c = 0
        koopman =  enc_size + out_size

        self.enc_size = enc_size
        self.in_size  = in_size
        self.out_size = out_size
        self.act_size = act_size
        self.verbose  = verbose
        self.koopman = koopman
        self.deterministic = False

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

        self.min_logvar = nn.Parameter(
            -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
        )
        self.max_logvar = nn.Parameter(
            0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
        )

        enc_hidden_layers = [
            nn.Sequential(create_linear_layer(out_size, hid_size), create_activation())
        ]
        bic_hidden_layers = [
            nn.Sequential(create_linear_layer(out_size, hid_size), create_activation())
        ]
        cov_hidden_layers = [
            nn.Sequential(create_linear_layer(enc_size+act_size, hid_size), create_activation())
        ]

        for i in range(num_layers - 1):
            enc_hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        for i in range(num_layers - 1):
            bic_hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        for i in range(num_layers - 1):
            cov_hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )

        self.enc_hidden_layers = nn.Sequential(*enc_hidden_layers)
        self.bic_hidden_layers = nn.Sequential(*bic_hidden_layers)
        self.cov_hidden_layers = nn.Sequential(*cov_hidden_layers)

        self.encode_net = create_linear_layer(hid_size, 2 * enc_size)
        self.bicode_net = create_linear_layer(hid_size, 2 * act_size)
        self.covariance_net = create_linear_layer(hid_size, out_size)

        self.lA = create_linear_layer(koopman, koopman)
        self.lB = create_linear_layer(act_size, koopman)
        self.lC = create_linear_layer(enc_size, out_size)
        self.lD = create_linear_layer(act_size, out_size)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    def _encode(self, x: torch.Tensor, dim: int, trajectory: bool = True):
        enc = self.enc_hidden_layers(x)
        enc = self.encode_net(enc)

        x = x if dim == 3 else x.repeat((self.num_members, 1, 1))

        return torch.cat([x, enc], axis=-1)

    def _bicode(self, x: torch.Tensor, u: torch.Tensor):
        g_x = self.bic_hidden_layers(x[..., :self.out_size])
        return self.bicode_net(g_x)[..., self.act_size:] * u, self.bicode_net(g_x)[..., :self.act_size] * u

    #TODO: Change logvar for a function, think about input
    def _default_forward(self, x: torch.Tensor, only_elite: bool = False, trajectory = True, **_kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_state = x[..., :self.out_size]
        x_act = x[..., self.out_size:]

        dim = x.dim()
        B = x.shape[0] if dim == 2 else x.shape[1]

        z_k = self._encode(x_state, dim, trajectory = True)

        z_mean = z_k[..., :self.enc_size+self.out_size]
        z_logvar = z_k[..., self.enc_size+self.out_size:]

        u_mean, u_logvar = self._bicode(z_mean, x_act)

        mean = self.lA(z_mean) + self.lB(u_mean)

        logvar = self.lC(z_logvar) + self.lD(u_logvar)

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean[..., :self.out_size], logvar

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            # print(self.debug_c)
            # self.debug_c += 1
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    # FIXME:
    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
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
            mean, logvar = self._default_forward(x, only_elite=True)
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

        # print("TEM QUE SER AQUI")
        # print(num_models)
        # print(model_shuffle_indices.shape)
        # print(x.shape)
        # print(shuffled_x.shape)

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        else:
            return self._nll_loss(model_in, target), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2

        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))

            return F.mse_loss(pred_mean, target, reduction="none"), {}

    # FIXME:
    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)
    