#@title PyTorch Lightning - CIFAR10
# PyTorch Lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

!pip install torchmetrics -qqq
import torchmetrics

!pip install pytorch_lightning -qqq
import pytorch_lightning as pl
import os

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# put cifar10_models dir (with state_dict folder that has weights) in same dir as this script; can select model and weights file from just one arch if want
from cifar10_models.mobilenetv2 import mobilenet_v2

import functools

BatchNorm2d = functools.partial(nn.BatchNorm2d)


class Sequential(nn.ModuleList):
    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, *args):
        for m in self:
            args = m(*args)
            if not isinstance(args, tuple):
                args = (args, )
        if len(args) == 1:
            return args[0]
        return args

    def estimate(self, inputs, outputs):
        return {'#macs': 0}


class Linear(nn.Linear):
    def estimate(self, inputs, outputs):
        cin = inputs[0].size(1)
        cout = outputs.size(1)
        return {'#macs': int(cin * cout)}


class Conv2d(nn.Conv2d):
    def estimate(self, inputs, outputs):
        kh, kw = self.kernel_size
        cin = inputs[0].size(1)
        _, cout, hout, wout = outputs.shape
        return {'#macs': int(np.product([kh, kw, cin, cout, hout, wout]))}


class Conv2dSame(Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            stride=1, padding=None, dilation=1, groups=1,
            bias=True):
        if padding is None:
            try:
                padding = {1: 0, 3: 1, 5: 2, 7: 3}[kernel_size]
            except KeyError:
                raise ValueError(
                    f'Unsupported padding for kernel size {kernel_size}.')
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)


class ModelBase(nn.Module):
    name = None
    input_size = None

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def load_state(self, state):
        saved = {n: p.shape for n, p in state.items()}
        ours = {n: p.shape for n, p in self.state_dict().items()}
        missing = [n for n in ours if n not in saved]

        summarize('Missing parameters', missing, level='warn')
        unexpected = [n for n in saved if n not in ours]
        summarize('Unused parameters', unexpected, level='warn')
        diffshape = [
            n for n, s in ours.items()
            if n in saved and tuple(s) != tuple(saved[n])]
        summarize('Parameters with mismatched shapes', diffshape, level='warn')
        self.load_state_dict(state, strict=False)

    def save(self, path, info):
        state = self.state_dict()
        state['.info'] = info
        torch.save(state, path)
    
    def get_num_params_flops(self, inputs):
        pass
        # macs, params = profile(self, inputs=(inputs, ))


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()

        ######### cifarnet arch #######

        self.conv1 = Conv2dSame(3, 64, 3, stride=(1, 1)) #channel input number also changes with dataset
        self.batchnorm2d1 = nn.BatchNorm2d(64)
        self.conv2 = Conv2dSame(64, 64, 3, stride=(1, 1))
        self.batchnorm2d2 = nn.BatchNorm2d(64)

        self.conv3 = Conv2dSame(64, 128, 3, stride=(1, 1))
        self.batchnorm2d3 = nn.BatchNorm2d(128)
        self.conv4 = Conv2dSame(128, 128, 3, stride=(1, 1))
        self.batchnorm2d4 = nn.BatchNorm2d(128)
        self.conv_dropout4 = nn.Dropout(p=0.5)
        self.conv5 = Conv2dSame(128, 128, 3, stride=(1, 1))
        self.batchnorm2d5 = nn.BatchNorm2d(128)

        self.conv6 = Conv2dSame(128, 192, 3, stride=(1, 1))
        self.batchnorm2d6 = nn.BatchNorm2d(192)
        self.conv7 = Conv2dSame(192, 192, 3, stride=(1, 1))
        self.batchnorm2d7 = nn.BatchNorm2d(192)
        self.conv_dropout7 = nn.Dropout(p=0.5)
        self.conv8 = Conv2dSame(192, 192, 3, stride=(1, 1))
        self.batchnorm2d8 = nn.BatchNorm2d(192)
        
        self.fc_final = nn.Linear(192*1*1, 10)
        self.batchnorm1d_final = nn.BatchNorm1d(10)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool_alt = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        ######### cifarnet arch #######

        # print(x.size())
        x = self.conv1(x)
        x = self.batchnorm2d1(x)
        x = self.act(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.batchnorm2d2(x)
        x = self.act(x)

        # print(x.size())
        x = self.conv3(x)
        x = self.batchnorm2d3(x)
        x = self.act(x)
        x = self.pool(x)
        # print(x.size())
        x = self.conv4(x)
        x = self.batchnorm2d4(x)
        x = self.act(x)
        x = self.conv_dropout4(x)

        # print(x.size())
        x = self.conv5(x)
        x = self.batchnorm2d5(x)
        x = self.act(x)
        # print(x.size())
        x = self.conv6(x)
        x = self.batchnorm2d6(x)
        x = self.act(x)
        x = self.pool(x)

        # print(x.size())
        x = self.conv7(x)
        x = self.batchnorm2d7(x)
        x = self.act(x)
        x = self.conv_dropout7(x)
        # print(x.size())
        x = self.conv8(x)
        x = self.batchnorm2d8(x)
        x = self.act(x)

        x = self.pool_alt(x)

        x = torch.flatten(x, 1)

        x = self.fc_final(x)
        x = self.batchnorm1d_final(x)

        return x


        

class Student(nn.Module):
        
    def __init__(self):
        super().__init__()

        self.num_channels = 32

        self.conv1 = Conv2dSame(3, self.num_channels, 3, stride=(1, 1)) #channel input number also changes with dataset
        self.batchnorm2d1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = Conv2dSame(self.num_channels, self.num_channels, 3, stride=(1, 1))
        self.batchnorm2d2 = nn.BatchNorm2d(self.num_channels)

        self.conv3 = Conv2dSame(self.num_channels, self.num_channels*2, 3, stride=(1, 1))
        self.batchnorm2d3 = nn.BatchNorm2d(self.num_channels*2)
        self.conv4 = Conv2dSame(self.num_channels*2, self.num_channels*2, 3, stride=(1, 1))
        self.batchnorm2d4 = nn.BatchNorm2d(self.num_channels*2)
        # self.conv_dropout4 = nn.Dropout(p=0.5)
        self.conv5 = Conv2dSame(self.num_channels*2, self.num_channels*2, 3, stride=(1, 1))
        self.batchnorm2d5 = nn.BatchNorm2d(self.num_channels*2)

        self.conv6 = Conv2dSame(self.num_channels*2, self.num_channels*3, 3, stride=(1, 1))
        self.batchnorm2d6 = nn.BatchNorm2d(self.num_channels*3)
        self.conv7 = Conv2dSame(self.num_channels*3, self.num_channels*3, 3, stride=(1, 1))
        self.batchnorm2d7 = nn.BatchNorm2d(self.num_channels*3)
        # self.conv_dropout7 = nn.Dropout(p=0.5)
        self.conv8 = Conv2dSame(self.num_channels*3, self.num_channels*3, 3, stride=(1, 1))
        self.batchnorm2d8 = nn.BatchNorm2d(self.num_channels*3)
        
        self.fc_final = nn.Linear(self.num_channels*3*1*1, 10)
        self.batchnorm1d_final = nn.BatchNorm1d(10)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool_alt = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm2d1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batchnorm2d2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.batchnorm2d3(x)
        x = self.act(x)

        x = self.pool(x)

        x = self.conv4(x)
        x = self.batchnorm2d4(x)
        x = self.act(x)
        # x = self.conv_dropout4(x)
        x = self.conv5(x)
        x = self.batchnorm2d5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.batchnorm2d6(x)
        x = self.act(x)

        x = self.pool(x)

        x = self.conv7(x)
        x = self.batchnorm2d7(x)
        x = self.act(x)
        # x = self.conv_dropout7(x)
        x = self.conv8(x)
        x = self.batchnorm2d8(x)
        x = self.act(x)

        x = self.pool_alt(x)

        x = torch.flatten(x, 1)

        x = self.fc_final(x)
        x = self.batchnorm1d_final(x)

        return x

class Teacher_pl(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.teacher = Teacher(num_classes=10) # when using Teacher based on skeleton (one that extends ModelBase)
        # self.teacher = Teacher() # to use og teacher arch defined above
        self.teacher = mobilenet_v2(pretrained=True) # to use pre-trained other arch
        # self.teacher = torch.load('group22_pretrained_model.h5')
        # self.teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        return self.teacher(x)

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


class Distiller_pl(pl.LightningModule):
    def __init__(self, trained_teacher):
        super().__init__()
        self.teacher = trained_teacher
        for param in self.teacher.parameters():
          param.requires_grad = False
        self.student = Student()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x): # what happens if don't define this? Where is in called under the hood?
        return self.student(x)

    def student_loss(self, student_predictions, labels):
      criterion = nn.CrossEntropyLoss()
      student_loss = criterion(student_predictions, labels)
      return student_loss
      
    def total_loss(self, teacher_predictions, student_predictions, labels):
      distillation_criterion = nn.KLDivLoss()
      alpha=0.1
      temperature=10
      student_loss = self.student_loss(student_predictions, labels)
      distillation_loss = (
          distillation_criterion(
              torch.nn.functional.log_softmax(student_predictions / temperature, dim=1),
              torch.nn.functional.softmax(teacher_predictions / temperature, dim=1)
          )
          *temperature*temperature
      )
      # print("student loss:", student_loss.item())
      # print("distillation loss:", distillation_loss.item())
      return alpha*student_loss + (1 -alpha)*distillation_loss

    def training_step(self, train_batch, batch_idx):
      data, labels = train_batch
      teacher_predictions = self.teacher.forward(data)
      student_predictions = self.student.forward(data)
      total_loss = self.total_loss(teacher_predictions, student_predictions, labels)
      self.log('train_loss', total_loss)
      self.train_acc(student_predictions, labels)
      self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
      return total_loss

    def test_step(self, val_batch, batch_idx):
      data, labels = val_batch
      student_predictions = self.student.forward(data)
      student_loss = self.student_loss(student_predictions, labels)
      self.log('test_loss', student_loss)
      self.test_acc(student_predictions, labels)
      self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
      optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
      return optimizer


class CIFAR10DataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize(mean, std)])
      
    # prepare standard transforms
    self.cifar10_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    self.cifar10_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.cifar10_train, batch_size=64, num_workers=10)

  def test_dataloader(self):
    return DataLoader(self.cifar10_test, batch_size=64, num_workers=10)


class CIFAR100DataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    mean = [0.507, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2761]
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize(mean, std)])
      
    # prepare standard transforms
    self.cifar100_train = CIFAR100(os.getcwd(), train=True, download=True, transform=transform)
    self.cifar100_test = CIFAR100(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.cifar100_train, batch_size=64, num_workers=10)

  def test_dataloader(self):
    return DataLoader(self.cifar100_test, batch_size=64, num_workers=10)


class CIFAR10FSLDataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize(mean, std)])
      
    # prepare standard transforms
    self.cifar10fsl_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    # self.train_sampler = self._create_random_subset_sampler( CIFAR10(os.getcwd(), train=True, download=True, transform=transform) )
    self.cifar10fsl_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

  # def _create_random_subset_sampler(self, dataset):
  #   data_per_class_index = [[] for _ in range(10)] # each elem is array of data samples that belong to that class
  #   for datum_idx, (_, label) in enumerate(dataset):
  #     class_idx = label
  #     data_per_class_index[class_idx].append(datum_idx)
  #   subset_indices = []
  #   # print(class_indices)
  #   for class_index_samples in data_per_class_index:
  #     subset_indices.extend(torch.randperm(len(class_index_samples))[:5])
  #   # print(subset_indices)
  #   return SubsetRandomSampler(subset_indices) 

  # def train_dataloader(self):
  #   return DataLoader(self.cifar10fsl_train, batch_size=64, num_workers=10, sampler=self.train_sampler)

  def train_dataloader(self):
    class_data_indices = [[] for _ in range(10)] # each elem is list of data samples that belong to that class
    for datum_idx, (_, label) in enumerate(self.cifar10fsl_train): # assign id to every sample in dataset; iterate through, and append the ids of sample belonging to the same class to our new 2D list
        class_idx = label
        class_data_indices[class_idx].append(datum_idx)

    fsl_data_indices = [] # indices of data samples that will be part of the new fsl subset of the dataset

    # for each class, randomly pick 5 data samples and append their ids to fsl data indices list
    for class_ in class_data_indices:
      rand_indices = (torch.randperm(len(class_))[:5]).tolist() # returns first 5 of a random permutation of all digits up to length of class (i.e. num of data samples in that class); tolist bc returns tensor
      print(rand_indices)
      for rand_index in rand_indices:
        fsl_data_indices.append(class_[rand_index])
    print(fsl_data_indices)

    # uses fsl indices to create subset of full train set
    fsl_dataset = torch.utils.data.Subset(self.cifar10fsl_train, fsl_data_indices)
    return DataLoader(fsl_dataset, batch_size=64, num_workers=10, shuffle=True)

  def test_dataloader(self):
    return DataLoader(self.cifar10fsl_test, batch_size=64, num_workers=10)


# data_module = CIFAR10DataModule()
# data_module = CIFAR100DataModule()
data_module = CIFAR10FSLDataModule()

early_stop_callback = EarlyStopping(
   monitor='train_acc',
   min_delta=0.0005,
   patience=3,
   verbose=False,
   mode='max'
)

# run 200 epochs with lr 5e-5 w Adam, read JADE documentation, use FLOP count tool to get count for resnet50 and CNV - can use 

# uncomment when training teacher from scratch
# train teacher
# teacher_model = Teacher_pl()
# teacher_trainer = pl.Trainer(accelerator="gpu", max_epochs=10)
# teacher_trainer.fit(teacher_model, data_module)
# teacher_trainer.test(teacher_model, data_module)
# teacher_trainer.save_checkpoint("teacher_cifarnet.ckpt")

# uncomment when using-pretrained other arch
teacher_model = Teacher_pl()
teacher_trainer = pl.Trainer(accelerator="gpu", max_epochs=10)
teacher_trainer.test(teacher_model, data_module)
# # teacher_trainer.save_checkpoint("teacher_mobilenetv2.ckpt")
# # teacher_trainer.save_checkpoint("teacher_cifar100.ckpt")

# uncomment when using checkpoint from arch defined here already trained
# teacher_model = Teacher_pl.load_from_checkpoint(checkpoint_path="teacher_cifarnet.ckpt")
# teacher_trainer = pl.Trainer(accelerator="auto", max_epochs=200)
# teacher_trainer.test(teacher_model, data_module)

# distill to student
# distiller_model = Distiller_pl(teacher_model)
# distiller_trainer = pl.Trainer(accelerator="gpu", max_epochs=10) #callbacks=[early_stop_callback]
# distiller_trainer.fit(distiller_model, data_module)
# distiller_trainer.test(distiller_model, data_module)
# distiller_trainer.save_checkpoint("distilled.ckpt")

# Can clearly see, if comment out teacher_trainer.fit, test accuracy becomes ~0.1 (i.e. basically random for 10 classes). But if train distillation, still get good test accuracy
# This is because student_loss still taken into account to train; setting alpha=0.0 (i.e. only distillation loss considered) and then training distiller results in ~0.1 test accuracy of student, as expected, if teacher untrained    