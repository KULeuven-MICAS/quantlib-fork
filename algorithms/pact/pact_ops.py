# 
# pact_ops.py
# 
# Author(s):
# Francesco Conti <f.conti@unibo.it>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from .pact_functions import PACTQuantize, AlmostSymmQuantFunc, PACTQuantFunc
import torch
from torch import nn


__all__ = [
    'PACTUnsignedAct',
    'PACTAsymmetricAct',
    'PACTConv2d',
    'PACTConv1d',
    'PACTLinear',
    'PACTQuantize',
]

def assert_param_valid(module : nn.Module, value, param_name : str, valid_values : list):
    error_str = f"[{module.__class__.__name__}]  Invalid argument {param_name}: Got {value}, expected {valid_values[0] if len(valid_values)==1 else ', '.join(valid_values[:-1]) + ' or ' + str(valid_values[-1])}"
    assert value in valid_values, error_str


class PACTUnsignedAct(nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation, considering unsigned outputs.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`torch.nn.ReLU`, :py:class:`torch.nn.ReLU6` and
    similar activations in a PACT-quantized network.

    This layer can also operate in a special mode, defined by the `statistics` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9

    """

    def __init__(
            self,
            n_levels=256,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            leaky=0.1,
            nb_std=3
    ):

        r"""Constructor.

        :param bits: currently targeted quantization level (default `None`).
        :type  bits: int or float
        :param clip: the value of the clipping factor :math:`\alpha`.
        :type  clip: `torch.Tensor` or float
        :param learn_clip: default `True`; if `False`, do not update the value of the clipping factor `\alpha` with backpropagation.
        :type  learn_clip: bool
        :param act_kind: 'relu', 'relu6', 'leaky_relu'
        :type  act_kind: string
        :param init_clip: 'max' for initialization of clip_hi (on activation of quantization)
                          with max value, 'std' for initialization to mean + nb_std*standard_dev
        :type  init_clip: string
        :param leaky:     leakiness parameter for leaky ReLU activation; unused if act_kind is not 'leaky_relu'
        :param nb_std:    number of standard deviations from mean to initialize the clipping value
        :type  nb_std:    float or int
        """

        super(PACTUnsignedAct, self).__init__()
        act_kind = act_kind.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, act_kind, 'act_kind', ['relu', 'relu6', 'leaky_relu'])
        assert_param_valid(self, init_clip, 'init_clip',  ['max', 'std', 'const'])
        self.n_levels = n_levels
        self.clip_hi = torch.nn.Parameter(torch.Tensor((1.,)), requires_grad=learn_clip)
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'high':self.clip_hi}
        self.act_kind = act_kind
        self.init_clip = init_clip
        self.nb_std = nb_std
        self.leaky = leaky
        # this is switched on/off by the PACTActController
        self.started = False

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.clip_hi.data), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.clip_hi.data),  requires_grad=False)

    def get_eps(self, *args):
        return self.clip_hi/(self.n_levels-1)

    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.

        See :py:class:`nemo.quant.pact_quant.PACTQuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`

        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """
        # in statistics collection mode, the activation works like a
        # relu/relu6/leaky_relu
        if not self.started:
            if self.act_kind == 'relu':
                x = torch.nn.functional.relu(x)
            elif self.act_kind == 'relu6':
                x = torch.nn.functional.relu6(x)
            elif self.act_kind == 'leaky_relu':
                x = torch.nn.functional.leaky_relu(x, self.leaky)
            with torch.no_grad():
                cur_max = torch.max(x)
                cur_min = torch.min(x)
                self.max.data[:] = torch.maximum(self.max.data, cur_max)
                self.min.data[:] = torch.minimum(self.min.data, cur_min)
                self.running_mean.data[:] = 0.9 * self.running_mean.data + 0.1 * torch.mean(x)
                self.running_var.data[:] = 0.9 * self.running_var.data  + 0.1 * torch.std(x)**2
            return x
        # in normal mode, PACTUnsignedAct uses the PACTQuantFunc
        else:
            eps = self.get_eps()
            # TODO why clip_hi+eps???
            return PACTQuantize(x, eps, torch.zeros(1, device=self.clip_hi.device), self.clip_hi, clip_gradient=torch.tensor(True, device=self.clip_hi.device)) # clip_gradient=True keeps NEMO compatibility


class PACTAsymmetricAct(nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation, considering signed outputs, not necessarily symmetric.

    Implements a :py:class:`torch.nn.Module` to implement PACT-style quantization functions.

    This layer can also operate in a special mode, defined by the `statistics` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9

    """

    def __init__(
            self,
            n_levels=256,
            init_clip='max',
            learn_clip=True,
            act_kind='relu',
            leaky=0.1,
            symm=False,
            nb_std=3
    ):

        r"""Constructor.
        :param n_levels: number of quantization levels
        :type  n_levels: int
        :param learn_clip: default `True`; if `False`, do not update the value of the clipping factors `\alpha`,`\beta` with backpropagation.
        :type  learn_clip: bool
        :param act_kind: activation type to use in statistics mode
        :type  act_kind: str
        :param symm:     whether or not to enforce (almost-)symmetricity of the clipping range
        :type  symm:     bool
        :param nb_std:   Distance (in number of standard deviations) from mean to set upper/lower clipping bounds if init_clip is 'std'

        """

        super(PACTAsymmetricAct, self).__init__()
        act_kind = act_kind.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, act_kind, 'act_kind', ['identity', 'relu', 'relu6', 'leaky_relu'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'const'])

        self.n_levels = n_levels
        self.clip_lo = torch.nn.Parameter(torch.Tensor((-1.,)), requires_grad=learn_clip)
        self.clip_hi  = torch.nn.Parameter(torch.Tensor((1.,)),  requires_grad=learn_clip and (not symm))
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}
        self.act_kind = act_kind
        self.leaky = leaky
        self.init_clip = init_clip
        self.nb_std = nb_std
        self.symm = symm
        # this is switched on/off by the PACTActController
        self.started = False

        # these are only used to gather statistics
        self.max          = torch.nn.Parameter(torch.zeros_like(self.alpha.data), requires_grad=False)
        self.min          = torch.nn.Parameter(torch.zeros_like(self.alpha.data), requires_grad=False)
        self.running_mean = torch.nn.Parameter(torch.zeros_like(self.alpha.data), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.ones_like(self.alpha.data),  requires_grad=False)

    def get_eps(self, *args):
        return (self.clip_hi-self.clip_lo)/(self.n_levels-1)

    def forward(self, x):
        r"""Forward-prop function for PACT-quantized activations.

        See :py:class:`nemo.quant.pact_quant.PACTQuantFunc` for details on the normal operation performed by this layer.
        In statistics mode, it uses a normal ReLU and collects statistics in the background.

        :param x: input activations tensor.
        :type  x: :py:class:`torch.Tensor`

        :return: output activations tensor.
        :rtype:  :py:class:`torch.Tensor`

        """

        # in statistics collection mode, the activation works like an identity function (is this intended?)
        if not self.started:
            with torch.no_grad():
                self.max[:] = max(self.max.item(), x.max())
                self.min[:] = min(self.min.item(), x.min())
                self.running_mean[:] = 0.9 * self.running_mean.item() + 0.1 * x.mean()
                self.running_var[:]  = 0.9 * self.running_var.item()  + 0.1 * x.std()*x.std()
            if self.act_kind == 'identity':
                return x
            elif self.act_kind == 'relu':
                return torch.nn.functional.relu(x)
            elif self.act_kind == 'relu6':
                return torch.nn.functional.relu6(x)
            elif self.act_kind == 'leaky_relu':
                return torch.nn.functional.leaky_relu(x, self.leaky)
        # in normal mode, PACTUnsignedAct uses
        else:
            eps = self.get_eps()
            if self.learn_clip and self.symm:
                clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
            else:
                clip_upper = self.clip_hi
            #TODO: why was this clip_hi+eps??
            return PACTQuantize(x, eps, self.clip_lo, clip_upper, clip_gradient=torch.tensor(True, device=self.clip_lo.device))


class PACTConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            n_levels = 256,
            quantize = 'per_layer',
            init_clip = 'sawb',
            learn_clip = False,
            symm_wts = True,
            nb_std = 3,
            **kwargs
    ):
        """

        :param in_channels: See torch.nn.Conv2d
        :param out_channels: See torch.nn.Conv2d
        :param kernel_size: See torch.nn.Conv2d
        :param n_levels: Number of weight quantization levels
        :param quantize: how to quantize weights - 'per_layer' or 'per_channel'
        :type  quantize: str
        :param init_clip: how weight clipping parameters should be initialized - 'sawb', 'max' or 'std'
        :param learn_clip: whether clipping bound(s) should be learned
        :param symm_wts: Indicates that the weights should cover a symmetrical range around 0. If n_levels is an odd number,
               the integer representations of the weights will go from -n_levels/2 to n_levels/2-1, and the clipping range will
               be set accordingly. If init_clip is 'sawb', the symm_wts parameter has no effect.
        :param kwargs: passed to Conv2d constructor
        # todo: quantize bias??
        """
        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb'])

        super(PACTConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.n_levels = n_levels
        self.quantize = quantize
        self.init_clip = init_clip
        self.learn_clip = learn_clip
        # this member indicates that quantization is enabled
        self.started = False
        self.symm_wts = symm_wts
        self.nb_std = nb_std
        clip_lo = torch.tensor(-1.)
        # clip_lo & clip_hi should have dimension (out_channels, 1, 1, 1) in case of per-channel quantization.
        # The PACTController will take care of managing them according to the configuration (per-channel, per-layer)
        clip_lo = self.expand_bounds(clip_lo)
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip)
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        # in the case when learn_clip and symm_wts are both True, clip_hi is not actually used;
        # instead the upper clipping bound is calculated from clip_lo with AlmostSymmQuantFunc.
        # This way, only the lower clip bound is
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=(learn_clip and not symm_wts))
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        # this member indicates that the module's clipping bounds should not be touched. it is set by the controller
        self.frozen = False

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1, 1))
        return t

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return (self.clip_hi-self.clip_lo)/(self.n_levels-1)

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

    def forward(self, x):
        if self.started:
            if self.learn_clip and self.symm_wts:
                clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
            else:
                clip_upper = self.clip_hi
            w = PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, clip_upper, clip_gradient=torch.tensor(True, device=self.clip_lo.device))
        else:
            w = self.weight
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_conv2d(cls, c : nn.Conv2d, **kwargs):
        # kwargs should be arguments to PACTConv2d
        return cls(in_channels=c.in_channels,
                   out_channels=c.out_channels,
                   kernel_size=c.kernel_size,
                   stride=c.stride,
                   padding=c.padding,
                   dilation=c.dilation,
                   groups=c.groups,
                   bias=(c.bias is not None),
                   padding_mode=c.padding_mode,
                   **kwargs)


class PACTConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            n_levels = 256,
            quantize = 'per_layer',
            init_clip = 'sawb',
            learn_clip = False,
            symm_wts = True,
            nb_std = 3,
            **kwargs
    ):
        """
        :param in_channels: See torch.nn.Conv2d
        :param out_channels: See torch.nn.Conv2d
        :param kernel_size: See torch.nn.Conv2d
        :param n_levels: Number of weight quantization levels
        :param quantize: how to quantize weights - 'per_layer' or 'per_channel'
        :type  quantize: str
        :param init_clip: how weight clipping parameters should be initialized - 'sawb', 'max' or 'std'
        :param learn_clip: whether clipping bound(s) should be learned
        :param symm_wts: Indicates that the weights should cover a symmetrical range around 0. If n_levels is an odd number,
               the integer representations of the weights will go from -n_levels/2 to n_levels/2-1, and the clipping range will
               be set accordingly. If init_clip is 'sawb', the symm_wts parameter has no effect.
        :param kwargs: passed to Conv1d constructor
        TODO: implement quantized bias?
        """

        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb'])

        super(PACTConv1d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.n_levels = n_levels
        self.quantize = quantize
        self.init_clip = init_clip
        self.learn_clip = learn_clip
        self.symm_wts = symm_wts
        self.nb_std = nb_std
        # this member indicates that quantization is enabled
        self.started = False

        clip_lo = torch.tensor(-1.)
        # clip_lo & clip_hi should have dimension (out_channels, 1, 1) to in the case of per-channel quantization.
        # The PACTController will take care of managing them according to the configuration (per-channel, per-layer)
        clip_lo = self.expand_bounds(clip_lo)
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip)
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        # in the case when learn_clip and symm_wts are both True, clip_hi is not actually used;
        # instead the upper clipping bound is calculated from clip_lo with AlmostSymmQuantFunc.
        # This way, only the lower clip bound is
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=(learn_clip and not symm_wts))
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        # this member indicates that the module's clipping bounds should not be touched. it is set by the controller
        self.frozen = False

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_channels*[t])
            t = torch.reshape(t, (self.out_channels, 1, 1))
        return t

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return (self.clip_hi-self.clip_lo)/(self.n_levels-1)

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

    def forward(self, x):
        if self.started:
            if self.learn_clip and self.symm_wts:
                clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
            else:
                clip_upper = self.clip_hi
            w = PACTQuantFunc(self.weight, self.get_eps_w(), self.clip_lo, clip_upper, clip_gradient=torch.tensor(True, device=self.clip_lo.device))
        else:
            w = self.weight
        return nn.functional.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_conv1d(cls, c : nn.Conv1d, **kwargs):
        # kwargs should be arguments to PACTConv2d
        return cls(in_channels=c.in_channels,
                   out_channels=c.out_channels,
                   kernel_size=c.kernel_size,
                   stride=c.stride,
                   padding=c.padding,
                   dilation=c.dilation,
                   groups=c.groups,
                   bias=(c.bias is not None),
                   padding_mode=c.padding_mode,
                   **kwargs)


class PACTLinear(nn.Linear):
    def __init__(self,
                 in_features : int,
                 out_features : int,
                 n_levels : int = 256,
                 quantize : str = 'per_layer',
                 init_clip : str = 'sawb',
                 learn_clip : bool = False,
                 symm_wts : bool = True,
                 nb_std : int = 3,
                 **kwargs):
        """
        :param in_features:   see nn.Linear
        :param out_features:  see nn.Linear
        :param n_levels:      Number of quantization levels
        :param quantize:      quantization type: 'per_layer' or 'per_channel'
        :param init_clip:     how to initialize clipping bounds: 'max', 'std' or 'sawb'
        :param learn_clip:    Whether clipping bound(s) should be learned
        :param symm_wts:      If weights should be forced to be (almost) symmetric around 0 so they map without offset to integers
        :param nb_std:        # of standard deviations from mean to initialize clipping bounds to if init_clip=='std'
        :param kwargs:        passed to nn.Linear constructor
        """

        quantize = quantize.lower()
        init_clip = init_clip.lower()
        assert_param_valid(self, quantize, 'quantize', ['per_layer', 'per_channel'])
        assert_param_valid(self, init_clip, 'init_clip', ['max', 'std', 'sawb'])

        super(PACTLinear, self).__init__(in_features, out_features, **kwargs)
        self.n_levels = n_levels
        self.quantize = quantize
        self.init_clip = init_clip
        self.learn_clip = learn_clip
        self.symm_wts = symm_wts
        self.nb_std = nb_std
        # this member indicates that quantization is enabled
        self.started = False

        clip_lo = torch.tensor(-1.)
        clip_lo = self.expand_bounds(clip_lo)
        self.clip_lo = nn.Parameter(clip_lo, requires_grad=learn_clip)
        clip_hi = torch.tensor(1.)
        clip_hi = self.expand_bounds(clip_hi)
        self.clip_hi = nn.Parameter(clip_hi, requires_grad=learn_clip and not symm_wts)
        # to provide convenient access for the controller to the clipping params, store them in a dict.
        self.clipping_params = {'low':self.clip_lo, 'high':self.clip_hi}

        # this member indicates that the module's clipping bounds should not be touched. it is set by the controller
        self.frozen = False

    def expand_bounds(self, t):
        if self.quantize == 'per_channel':
            if t.numel() == 1:
                t = torch.reshape(t, (1,))
                t = torch.cat(self.out_features * [t])
            t = t.reshape((self.out_features, 1))
        return t

    def get_eps_w(self):
        """
        :return: epsilon of the weight quantization.
        """
        return (self.clip_hi-self.clip_lo)/(self.n_levels-1)

    def get_eps_out(self, eps_in, *args, **kwargs):
        """
        :return: epsilons of the output pre-activations
        """
        return self.get_eps_w()*eps_in

    def forward(self, x):
        if self.started:
            if self.learn_clip and self.symm_wts:
                clip_upper = AlmostSymmQuantFunc.apply(self.clip_lo, self.n_levels)
            else:
                clip_upper = self.clip_hi
            w = PACTQuantize(self.weight, self.get_eps_w(), self.clip_lo, clip_upper, clip_gradient=torch.tensor(True, device=self.clip_lo.device))
        else:
            w = self.weight
        return nn.functional.linear(x, w, self.bias)

    @classmethod
    def from_linear(cls, l : nn.Linear, **kwargs):
        return cls(in_features=l.in_features,
                   out_features=l.out_features,
                   bias=(l.bias is not None),
                   **kwargs)
