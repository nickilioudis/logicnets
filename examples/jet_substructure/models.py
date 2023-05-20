#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import reduce
from os.path import realpath

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.nn import QuantHardTanh, QuantReLU

from pyverilator import PyVerilator

from src.logicnets.quant import QuantBrevitasActivation
from src.logicnets.nn import SparseLinearNeq, ScalarBiasScale, RandomFixedSparsityMask2D, SparseConvNeq # N added
from src.logicnets.init import random_restrict_fanin

class JetSubstructureNeqModel(nn.Module):
    def __init__(self, model_config):
        super(JetSubstructureNeqModel, self).__init__()
        self.model_config = model_config
        self.num_neurons = [model_config["input_length"]] + model_config["hidden_layers"] + [model_config["output_length"]]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i-1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(QuantHardTanh(model_config["input_bitwidth"], max_val=1., narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn_in, input_bias])
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["input_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=input_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask})
                layer_list.append(layer)
            elif i == len(self.num_neurons)-1:
                output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=1.33, narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=[output_bias_scale])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
            else:
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    def verilog_inference(self, verilog_dir, top_module_filename, logfile: bool = False, add_registers: bool = False):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(f"{self.verilog_dir}/{self.top_module_filename}", verilog_path=[self.verilog_dir], build_dir=f"{self.verilog_dir}/verilator")
        self.is_verilog_inference = True
        self.logfile = logfile
        if add_registers:
            self.latency = len(self.num_neurons)

    def pytorch_inference(self):
        self.is_verilog_inference = False

    def verilog_forward(self, x):
        # Get integer output from the first layer
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[-1].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features*input_bitwidth
        total_output_bits = self.module_list[-1].out_features*output_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i,:]
            y_i = self.pytorch_forward(x[i:i+1,:])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str(z), y_i))
            xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
            ysc_i = reduce(lambda a,b: a+b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                #print(self.dut.io.M5)
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert(expected == result)
            res_split = [result[i:i+output_bitwidth] for i in range(0, len(result), output_bitwidth)][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))
            y[i,:] = yv_i
            # Dump the I/O pairs
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n")
        return y

    def pytorch_forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)

class JetSubstructureLutModel(JetSubstructureNeqModel):
    pass

class JetSubstructureVerilogModel(JetSubstructureNeqModel):
    pass


class JetSubstructureConvNeqModel(nn.Module):
    def __init__(self, model_config):
        super(JetSubstructureConvNeqModel, self).__init__()
        self.model_config = model_config
        self.in_channels = model_config["conv_params"]["in_channels"]
        self.kernel_height = model_config["conv_params"]["kernel_height"]
        self.kernel_width = model_config["conv_params"]["kernel_width"]
        self.in_height = model_config["conv_params"]["in_height"]
        self.in_width = model_config["conv_params"]["in_width"]
        self.in_features = self.in_channels * self.kernel_height * self.kernel_width #currently keep same kernel size for all layers
        # num_neurons is array where each element is num of outputs (hence previous value is num of inputs) in the corresponding layer
        # this is equivalent to num of neurons in that layer, with the exception of first and last element which are just num inputs/output, and don't have corresponding neurons
       
        # each conv hidden layer defined as num output channels, and since will do serial patch calculations for now, each output channel is one NEQ
        # each fc hidden layer just num neurons in that layer
        # TODO: currently only support simplest convolution layers; need to add sparse_conv_kws like stride, padding... to pass when instantiating layer as SparseConvNeq; must add these and kernel size to config explicitly?
        self.conv_out_flat_length = 24*24*32 # self.in_height * self.in_width * model_config["hidden_layers"]["conv"][-1]  # assumes padding same: otherwise calc using input dims, kernel size, stride, padding? See pytorhc CONV2D page
        self.num_neurons_conv = [self.in_channels] + model_config["hidden_layers"]["conv"]
        self.num_neurons_fc = [self.conv_out_flat_length] + model_config["hidden_layers"]["fc"] + [model_config["output_length"]]
        layer_list = []
        for i in range(1, len(self.num_neurons_conv)):
            in_channels = self.num_neurons_conv[i-1]
            out_channels = self.num_neurons_conv[i]
            in_features = in_channels * self.kernel_height * self.kernel_width
            out_features = out_channels # for each 'patch' in any layer's input image (i.e. in_channels * kernel_height * kernel_width), we produce out_channels output values that together form a single pixel of the output image
            bn = nn.BatchNorm2d(out_channels) # N - changed from 1D to 2D
            if i == 1: # first layer
                bn_in = nn.BatchNorm2d(in_channels)
                input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(QuantHardTanh(model_config["input_bitwidth"], max_val=1., narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn_in, input_bias])
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["input_fanin"])
                layer = SparseConvNeq(in_channels=in_channels, out_channels=out_channels, kernel_size=(self.kernel_height, self.kernel_width), input_quant=input_quant, output_quant=output_quant, sparse_conv_kws={'mask': mask})
                layer_list.append(layer)
            elif i == len(self.num_neurons_conv)-1: # in between layers
                output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=1.33, narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=[output_bias_scale])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
                layer = SparseConvNeq(in_channels=in_channels, out_channels=out_channels, kernel_size=(self.kernel_height, self.kernel_width), input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_conv_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
            else: # last layer
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
                layer = SparseConvNeq(in_channels=in_channels, out_channels=out_channels, kernel_size=(self.kernel_height, self.kernel_width), input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_conv_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
        for i in range(1, len(self.num_neurons_fc)):
            in_features = self.num_neurons_fc[i-1]
            out_features = self.num_neurons_fc[i]
            bn = nn.BatchNorm1d(out_features)
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(QuantHardTanh(model_config["input_bitwidth"], max_val=1., narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn_in, input_bias])
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["input_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=input_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask})
                layer_list.append(layer)
            elif i == len(self.num_neurons_fc)-1:
                output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=1.33, narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=[output_bias_scale])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
            else:
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    def verilog_inference(self, verilog_dir, top_module_filename, logfile: bool = False, add_registers: bool = False):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(f"{self.verilog_dir}/{self.top_module_filename}", verilog_path=[self.verilog_dir], build_dir=f"{self.verilog_dir}/verilator")
        self.is_verilog_inference = True
        self.logfile = logfile
        if add_registers:
            self.latency = len(self.num_neurons_conv) + len(self.num_neurons_fc)

    def pytorch_inference(self):
        self.is_verilog_inference = False

    def verilog_forward(self, x): # N - NEED TO CHANGE THIS?
        # Get integer output from the first layer
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[-1].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features*input_bitwidth # N - in_features is attribute of SparseConvNeq class
        total_output_bits = self.module_list[-1].out_features*output_bitwidth # N - same
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        x = torch.flatten(x, 1) # N - added this; is 1 needed to flatten all dims except batch?
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i,:]
            y_i = self.pytorch_forward(x[i:i+1,:])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str(z), y_i))
            xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
            ysc_i = reduce(lambda a,b: a+b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                #print(self.dut.io.M5)
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert(expected == result)
            res_split = [result[i:i+output_bitwidth] for i in range(0, len(result), output_bitwidth)][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))
            y[i,:] = yv_i
            # Dump the I/O pairs
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n")
        return y

    def pytorch_forward(self, x):
        for l in self.module_list:
            # N - added
            # print("x_size:", x.size())
            if type(l)==SparseLinearNeq:
                x = torch.flatten(x, 1)
            x = l(x) # N - since each elem l is a nn.module (namely, either SparseLinearNeq or SparseConvNeq), calling l(x) calls 'forward' function of that module
        return x

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)

class JetSubstructureConvLutModel(JetSubstructureConvNeqModel):
    pass

class JetSubstructureConvVerilogModel(JetSubstructureConvNeqModel):
    pass