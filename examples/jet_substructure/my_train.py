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

import os
from argparse import ArgumentParser
from functools import reduce
import random

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel, JetSubstructureConvNeqModel # N added

# N - added
import torchvision
import torchvision.transforms as transforms
# from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torchmetrics
import pytorch_lightning as pl
import os


######################################### start N Added ###############################################
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, (3,3), stride=(1, 1)) #channel input number also changes with dataset
        self.batchnorm2d1 = nn.BatchNorm2d(64)
        # self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride=(1, 1))
        self.batchnorm2d2 = nn.BatchNorm2d(64)
        # self.act2 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv3 = nn.Conv2d(64, 128, (3,3), stride=(1, 1))
        self.batchnorm2d3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, (3,3), stride=(1, 1))
        self.batchnorm2d4 = nn.BatchNorm2d(128)
        # self.act3 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv5 = nn.Conv2d(128, 256, (3,3), stride=(1, 1))
        self.batchnorm2d5 = nn.BatchNorm2d(256)
        # self.act4 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, (3,3), stride=(1, 1))
        self.batchnorm2d6 = nn.BatchNorm2d(256)

        # self.act5 = nn.ReLU()
        self.fc1 = nn.Linear(256*18*18, 512) # need to change this depending on dataset!
        self.batchnorm1d1 = nn.BatchNorm1d(512)
        # self.act6 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.batchnorm1d2 = nn.BatchNorm1d(512)
        # self.act7 = nn.ReLU()
        self.fc_final = nn.Linear(512, 10)
        self.batchnorm1d_final = nn.BatchNorm1d(10)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

# Conv64,3,1,
# BatchNorm,
# Conv64,3,1,
# BatchNorm,
# MaxPool2,
# Conv128,3,1,
# BatchNorm,
# Conv128,3,1,
# BatchNorm, 
# MaxPool2,
# Conv256,3,1,
# BatchNorm,
# Conv256,3,1,
# BatchNorm,

# FConn512,
# BatchNorm,
# FConn512,
# BatchNorm, 
# FConn10,
# BatchNorm,
# SoftMax

    # sooo - discovered Keras mis-documented padding=same (https://github.com/keras-team/keras/issues/15703), bc setting stride>1 yields output size diff to input
    # which means output of e.g. conv1 was 14x14 instead of 28x28
    # Hence, after looking at model summary of ref KD and checking dimensions, basically did educated trial and error to get padding here in PyTorch Lightning to yield same output sizes at each layer as Keras, to compare
    # Many key points, incl: 'padding' arg of nn.Conv2D does not allow diff left/right or top/bottom padding, only one tuple of (left/right, top/bot), so odd kernel size screws things up
    # Hence have to use F.pad to explicitly pad left and right, top and bottom separately
    # Also, padding='same' only supported for stride=(1,1), in nn.Conv2D
    def forward(self, x):
        # print(x.size())
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.conv1(x)
        x = self.batchnorm2d1(x)
        x = self.act(x)
        # print(x.size())
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.conv2(x)
        x = self.batchnorm2d2(x)
        x = self.act(x)
        # print(x.size())
        # x = F.pad(x, (1, 0, 1, 0))
        x = self.pool(x)
        # print(x.size())
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.conv3(x)
        x = self.batchnorm2d3(x)
        x = self.act(x)
        # print(x.size())
        # x = F.pad(x, (1, 0, 1, 0))
        x = self.pool(x)
        # print(x.size())
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.conv4(x)
        x = self.batchnorm2d4(x)
        x = self.act(x)
        # print(x.size())
        # x = F.pad(x, (1, 1, 1, 1))
        x = self.conv5(x)
        x = self.batchnorm2d5(x)
        x = self.act(x)
        # print(x.size())
        x = self.conv6(x)
        x = self.batchnorm2d6(x)
        x = self.act(x)
        # print(x.size())

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batchnorm1d1(x)
        x = self.act(x)
        # print(x.size())
        x = self.fc2(x)
        x = self.batchnorm1d2(x)
        x = self.act(x)
        # print(x.size())
        x = self.fc_final(x)
        x = self.batchnorm1d_final(x)
        return x
    

class Teacher_pl(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.teacher = Teacher()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.teacher.forward(x)

    def teacher_loss(self, teacher_predictions, labels):
      criterion = nn.CrossEntropyLoss()
      teacher_loss = criterion(teacher_predictions, labels)
      return teacher_loss

    def training_step(self, train_batch, batch_idx):
      data, labels = train_batch
      teacher_predictions = self.teacher.forward(data)
      teacher_loss = self.teacher_loss(teacher_predictions, labels)
      self.log('train_loss', teacher_loss)
      self.train_acc(teacher_predictions, labels)
      self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
      return teacher_loss

    def test_step(self, test_batch, batch_idx):
      data, labels = test_batch
      teacher_predictions = self.teacher.forward(data)
      teacher_loss = self.teacher_loss(teacher_predictions, labels)
      self.log('test_loss', teacher_loss)
      self.test_acc(teacher_predictions, labels)
      self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
      optimizer = optim.Adam(self.teacher.parameters(), lr=1e-3)
      return optimizer
    

# class MNISTDataModule(pl.LightningDataModule):

#   def setup(self, stage):
#     # transforms for images
#     transform=transforms.Compose([transforms.ToTensor(), 
#                                   transforms.Normalize((0.5,), (0.5,))])
      
#     # prepare transforms standard to MNIST
#     self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
#     self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

#   def train_dataloader(self):
#     return DataLoader(self.mnist_train, batch_size=64, num_workers=0)

#   def test_dataloader(self):
#     return DataLoader(self.mnist_test, batch_size=64, num_workers=0)
  
class CIFAR10DataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.5,), (0.5,), (0.5,))])
      
    # prepare transforms standard to MNIST
    self.cifar10_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    self.cifar10_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.cifar10_train, batch_size=32, num_workers=10)

  def test_dataloader(self):
    return DataLoader(self.cifar10_test, batch_size=32, num_workers=10)

######################################### end N Added ###############################################



# TODO: Replace default configs with YAML files.
configs = {
    "jsc-s": {
        "hidden_layers": [64, 32, 32, 32],
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 3,
        "hidden_fanin": 3,
        "output_fanin": 3,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate": 1e-3,
        "seed": 2,
        "checkpoint": None,
    },
    "jsc-m": {
        "hidden_layers": [64, 32, 32, 32],
        "input_bitwidth": 3,
        "hidden_bitwidth": 3,
        "output_bitwidth": 3,
        "input_fanin": 4,
        "hidden_fanin": 4,
        "output_fanin": 4,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate": 1e-3,
        "seed": 3,
        "checkpoint": None,
    },
    "jsc-l": {
        "hidden_layers": [32, 64, 192, 192, 16],
        "input_bitwidth": 4,
        "hidden_bitwidth": 3,
        "output_bitwidth": 7,
        "input_fanin": 3,
        "hidden_fanin": 4,
        "output_fanin": 5,
        "weight_decay": 1e-3,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate": 1e-3,
        "seed": 16,
        "checkpoint": None,
    },
    ########################### start N Added #######################
    "LFC-student": {
        "hidden_layers": [64, 32, 32, 32], # treat number as FC so don't have to change code, but introduce e.g. ('conv', 32)?
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 3,
        "hidden_fanin": 3,
        "output_fanin": 3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "epochs": 1000,
        "learning_rate": 1e-3,
        "seed": 2,
        "checkpoint": None,
    },
    # "conv-student": {
    #     "hidden_layers": {
    #         "conv": [16, 32],
    #         "fc": [10],
    #     },
    #     "conv_params": {
    #         "in_height": 28,
    #         "in_width": 28,
    #         "in_channels": 1,
    #         "kernel_height": 3,
    #         "kernel_width": 3,
    #     },
    #     "input_bitwidth": 2,
    #     "hidden_bitwidth": 2,
    #     "output_bitwidth": 2,
    #     "input_fanin": 3,
    #     "hidden_fanin": 3,
    #     "output_fanin": 3,
    #     "weight_decay": 1e-3,
    #     "batch_size": 64,
    #     "epochs": 100,
    #     "learning_rate": 1e-3,
    #     "seed": 2,
    #     "checkpoint": None,
    # },
    "conv-student": {
        "hidden_layers": {
            "conv": [16, 32, 32],
            "fc": [32, 10],
        },
        "conv_params": {
            "in_height": 32,
            "in_width": 32,
            "in_channels": 3,
            "kernel_height": 3,
            "kernel_width": 3,
        },
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 3,
        "hidden_fanin": 3,
        "output_fanin": 3,
        "weight_decay": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 1e-3,
        "seed": 2,
        "checkpoint": None,
    },
     ########################### end N Added #######################
}

# A dictionary, so we can set some defaults if necessary
model_config = {
    "hidden_layers": None,
    "conv_params": None, # N added
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "input_fanin": None,
    "hidden_fanin": None,
    "output_fanin": None,
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
}

dataset_config = {
    "dataset_file": None,
    "dataset_config": None,
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
}


 ########################### start N Added #######################

def student_loss(student_predictions, labels):
      criterion = nn.CrossEntropyLoss()
      student_loss = criterion(student_predictions, labels)
      return student_loss


# notes: removed self keyword bc no longer method of a class
def total_loss(teacher_predictions, student_predictions, labels):
      distillation_criterion = nn.KLDivLoss()
      alpha=0.1
      temperature=10
      student_loss_m = student_loss(student_predictions, labels) # gave _m subscript since no longer have class and would confuse student_loss var with function
      distillation_loss = (
          distillation_criterion(
              torch.nn.functional.softmax(teacher_predictions / temperature, dim=1),
              torch.nn.functional.softmax(student_predictions / temperature, dim=1),
          )
          * temperature**2
      )
      return alpha*student_loss_m + (1-alpha)*distillation_loss

 ########################### end N Added #######################


def train(model, datasets, train_cfg, options, teacher_model): # N - added teacher model
    # Create data loaders for training and inference:
    train_loader = DataLoader(datasets["train"], batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(datasets["valid"], batch_size=train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=train_cfg['batch_size'], shuffle=False)
  
    # Configure optimizer
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = ["bn", "bias", "learned_value"] # Make a list of parameters name fragments which will ignore weight decay TODO: make this list part of the train_cfg
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(lambda a,b: a or b, map(lambda x: x in pname, decay_exclusions)): # check if the current label should be excluded from weight decay
                #print("Disabling weight decay for %s" % (pname))
                no_decay_params.append(params)
            else:
                #print("Enabling weight decay for %s" % (pname))
                decay_params.append(params)
        #else:
            #print("Ignoring %s" % (pname))
    params =    [{'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}]
    optimizer = optim.AdamW(params, lr=train_cfg['learning_rate'], betas=(0.5, 0.999), weight_decay=weight_decay)

    # Configure scheduler
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps*100, T_mult=1)

    # Configure criterion
    # criterion = nn.CrossEntropyLoss() # N - commented out bc define differently, below

    # Push the model to the GPU, if necessary
    if options["cuda"]:
        model.cuda()

    # Setup tensorboard
    writer = SummaryWriter(options["log_dir"])

    # Main training loop
    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]
    for epoch in range(0, num_epochs):
        # Train for this epoch
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)


            ########################### start N changed #######################

            # print("output, target:", len(output), len(target)) # N - added

            # N - changed
            # loss = criterion(output, torch.max(target, 1)[1])
            # pred = output.detach().max(1, keepdim=True)[1]
            # target_label = torch.max(target.detach(), 1, keepdim=True)[1] 

            # loss = criterion(output, target) # when training with MNIST, prior to KD integration

            # teacher_model passed into arg of train() is pl module but has simple, well-defined forward, so can do this
            teacher_predictions = teacher_model.forward(data) #NOTE: reason why accuracy might be bad is bc distillation loss thinks predictions are PRE-softmax probabilities whereas here they are actual labels??
            loss = total_loss(teacher_predictions, output, target)

            # .detach() makes separate copy of tensor that won't keep metadata like grad propagation?
            # .max(1, keepdim=True)[1] returns indices max values along dim 1 (cols) of output vec; this convert each softmax prob vector to tensor of class indices
            # this means pred is tensor of one-element tensors; if problem was multi-label, could have multiple indices in each label tensor
            pred = output.detach().max(1, keepdim=True)[1]  
            # after a lot of ChatGPT prompts lol, found that .cat and .stack can concatenate tensors - but input needs to be tuple of tensors, not tensor of tensors
            pred = torch.cat(tuple(pred))
            target_label = target.detach()
            
            # prints used to rebug, bc had accuracy % smth like 600, so obv wrong
            # print("pred, target_label, data:", len(pred), len(target_label), len(data))
            # print("pred, target_label", pred, target_label)
            # print("long:", pred.eq(target_label))

            # .eq returns bool tensor with True at indices where pred matched target_label; .long converts to ints - True 1, False 0; sum gives total 1s hence correct preds
            curCorrect = pred.eq(target_label).long().sum()

            # print("teacher_predictions:", teacher_predictions)
            # print("output:", output)
            # print("target:", target)
            # print("pred:", pred)
            # print("target_label:", target_label)
            # print("currCorrect:", curCorrect)
            # print("batch_idx:", batch_idx)

            ########################### end N changed #######################


            curAcc = 100.0*curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach()*len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log stats to tensorboard
            #writer.add_scalar('train_loss', loss.detach().cpu().numpy(), epoch*steps + batch_idx)
            #writer.add_scalar('train_accuracy', curAcc.detach().cpu().numpy(), epoch*steps + batch_idx)
            #g = optimizer.param_groups[0]
            #writer.add_scalar('LR', g['lr'], epoch*steps + batch_idx)

        print("correct, dataset:", correct,len(train_loader.dataset))
        accLoss /= len(train_loader.dataset)
        accuracy = 100.0*correct / len(train_loader.dataset)
        print(f"Epoch: {epoch}/{num_epochs}\tTrain Acc (%): {accuracy.detach().cpu().numpy():.2f}\tTrain Loss: {accLoss.detach().cpu().numpy():.3e}")
        #for g in optimizer.param_groups:
        #        print("LR: {:.6f} ".format(g['lr']))
        #        print("LR: {:.6f} ".format(g['weight_decay']))
        writer.add_scalar('avg_train_loss', accLoss.detach().cpu().numpy(), (epoch+1)*steps)
        writer.add_scalar('avg_train_accuracy', accuracy.detach().cpu().numpy(), (epoch+1)*steps)
        val_accuracy, val_avg_roc_auc = test(model, val_loader, options["cuda"])
        test_accuracy, test_avg_roc_auc = test(model, test_loader, options["cuda"])
        modelSave = {   'model_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'val_avg_roc_auc': val_avg_roc_auc,
                        'test_avg_roc_auc': test_avg_roc_auc,
                        'epoch': epoch}
        torch.save(modelSave, options["log_dir"] + "/checkpoint.pth")
        if(maxAcc<val_accuracy):
            torch.save(modelSave, options["log_dir"] + "/best_accuracy.pth")
            maxAcc = val_accuracy
        writer.add_scalar('val_accuracy', val_accuracy, (epoch+1)*steps)
        writer.add_scalar('test_accuracy', test_accuracy, (epoch+1)*steps)
        writer.add_scalar('val_avg_roc_auc', val_avg_roc_auc, (epoch+1)*steps)
        writer.add_scalar('test_avg_roc_auc', test_avg_roc_auc, (epoch+1)*steps)
        print(f"Epoch: {epoch}/{num_epochs}\tValid Acc (%): {val_accuracy:.2f}\tTest Acc: {test_accuracy:.2f}")

def test(model, dataset_loader, cuda):
    with torch.no_grad():
        model.eval()
        entire_prob = None
        golden_ref = None
        correct = 0
        accLoss = 0.0
        for batch_idx, (data, target) in enumerate(dataset_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            prob = F.softmax(output, dim=1)
            pred = output.detach().max(1, keepdim=True)[1]


            ############### start N changed ################

            pred = torch.cat(tuple(pred))
            # target_label = torch.max(target.detach(), 1, keepdim=True)[1]
            target_label = target.detach()

            ############### end N changed ################

            curCorrect = pred.eq(target_label).long().sum()
            curAcc = 100.0*curCorrect / len(data)
            correct += curCorrect
            if batch_idx == 0:
                entire_prob = prob
                golden_ref = target_label
            else:
                entire_prob = torch.cat((entire_prob, prob), dim=0)
                golden_ref = torch.cat((golden_ref, target_label))
        accuracy = 100*float(correct) / len(dataset_loader.dataset)
        avg_roc_auc = roc_auc_score(golden_ref.detach().cpu().numpy(), entire_prob.detach().cpu().numpy(), average='macro', multi_class='ovr')
        return accuracy, avg_roc_auc

if __name__ == "__main__":
    parser = ArgumentParser(description="LogicNets Jet Substructure Classification Example")
    parser.add_argument('--arch', type=str, choices=configs.keys(), default="jsc-s",
        help="Specific the neural network model to use (default: %(default)s)")
    parser.add_argument('--weight-decay', type=float, default=None, metavar='D',
        help="Weight decay (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
        help="Batch size for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
        help="Number of epochs to train (default: %(default)s)")
    parser.add_argument('--learning-rate', type=float, default=None, metavar='LR',
        help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--cuda', action='store_true', default=False,
        help="Train on a GPU (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=None,
        help="Seed to use for RNG (default: %(default)s)")
    parser.add_argument('--input-bitwidth', type=int, default=None,
        help="Bitwidth to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-bitwidth', type=int, default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)")
    parser.add_argument('--output-bitwidth', type=int, default=None,
        help="Bitwidth to use at the output (default: %(default)s)")
    parser.add_argument('--input-fanin', type=int, default=None,
        help="Fanin to use at the input (default: %(default)s)")
    parser.add_argument('--hidden-fanin', type=int, default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)")
    parser.add_argument('--output-fanin', type=int, default=None,
        help="Fanin to use at the output (default: %(default)s)")
    parser.add_argument('--hidden-layers', nargs='+', type=int, default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)")
    parser.add_argument('--log-dir', type=str, default='./log',
        help="A location to store the log output of the training run and the output model (default: %(default)s)")
    parser.add_argument('--dataset-file', type=str, default='data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z',
        help="The file to use as the dataset input (default: %(default)s)")
    parser.add_argument('--dataset-config', type=str, default='config/yaml_IP_OP_config.yml',
        help="The file to use to configure the input dataset (default: %(default)s)")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="Retrain the model from a previous checkpoint (default: %(default)s)")
    ########## start N added #########
    parser.add_argument('--conv-params', nargs='+', type=int, default=None,
        help="A dictionary of convolution parameters like kernel size (default: %(default)s)")
    ########## end N added #########
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options['arch']
    config = {}
    for k in options.keys():
        config[k] = options[k] if options[k] is not None else defaults[k] # Override defaults, if specified.

    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]

    # Set random seeds
    random.seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])
    torch.manual_seed(train_cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(train_cfg['seed'])
    if options["cuda"]:
        torch.cuda.manual_seed_all(train_cfg['seed'])
        torch.backends.cudnn.deterministic = True



    ########################## start N changed ###########################

    # data_module = MNISTDataModule()
    data_module = CIFAR10DataModule()

    # uncomment to train afresh
    # train teacher as pl module
    teacher_model = Teacher_pl() #NOTE this will be passed to train(), but only use its .forward there
    teacher_trainer = pl.Trainer(accelerator="auto", max_epochs=5)
    teacher_trainer.fit(teacher_model, data_module)
    teacher_trainer.test(teacher_model, data_module)
    teacher_trainer.save_checkpoint("teacher.ckpt")

    # comment out, when training afresh
    # teacher_model = Teacher_pl.load_from_checkpoint(checkpoint_path="teacher.ckpt")
    # teacher_trainer = pl.Trainer(accelerator="auto", max_epochs=5)
    # teacher_trainer.test(teacher_model, data_module)

    # Fetch the datasets
    dataset = {}
    # dataset['train'] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="train")
    # dataset['valid'] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="train") # This dataset is so small, we'll just use the training set as the validation set, otherwise we may have too few trainings examples to converge.
    # dataset['test'] = JetSubstructureDataset(dataset_cfg['dataset_file'], dataset_cfg['dataset_config'], split="test")

    # transforms for images
    # transform=transforms.Compose([transforms.ToTensor(), 
    #                               transforms.Normalize((0.5,), (0.5,)),
    #                               transforms.Lambda(torch.flatten)]) # need to add torch.flatten bc student model is currently same as jsc and accepts only 1D vec as input data, not 2D MNIST images
    
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.5,), (0.5,), (0.5,))])
    
    # prepare transforms standard to dataset
    dataset['train'] = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    dataset['valid'] = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    dataset['test'] = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

    ########################## end N changed ###########################



    # Instantiate model
    x, y = dataset['train'][0]
    # model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = 10 # N - changed from len(y)
    model = JetSubstructureConvNeqModel(model_cfg)
    if options_cfg['checkpoint'] is not None:
        print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
        checkpoint = torch.load(options_cfg['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_dict'])

    # train(model, dataset, train_cfg, options_cfg, teacher_model) # N - added teacher_model arg

