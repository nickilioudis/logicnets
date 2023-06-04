#@title PyTorch Lightning - CIFAR100, CIFAR10 FSL
# PyTorch Lightning
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

# !pip install torchmetrics -qqq
import torchmetrics

# !pip install pytorch_lightning -qqq
import pytorch_lightning as pl
import os

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# put cifar10_models dir (with state_dict folder that has weights) in same dir as this script; can select model and weights file from just one arch if want
# from cifar10_models.mobilenetv2 import mobilenet_v2

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

        

class Student(nn.Module):
        
    def __init__(self, num_classes):
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

        self.fc_final_input = self.num_channels*3*1*1
        self.fc_final_output = num_classes
        
        self.fc_final = nn.Linear(self.fc_final_input, self.fc_final_output)
        self.batchnorm1d_final = nn.BatchNorm1d(self.fc_final_output)

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
    def __init__(self, num_classes=100):
        super().__init__()
        # self.teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
        self.teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
        # for param in self.teacher.parameters():
        #   param.requires_grad = False
        if num_classes != 100:
          self.teacher.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.teacher.last_channel, num_classes),
          )
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

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
      optimizer = optim.Adam(self.teacher.parameters(), lr=5e-4)
      return optimizer


class Distiller_pl(pl.LightningModule):
    def __init__(self, num_classes, trained_teacher, student=None):
        super().__init__()
        self.teacher = trained_teacher
        for param in self.teacher.parameters():
          param.requires_grad = False
        if student is None: # if no student passed, then create new student
          self.student=Student(num_classes=num_classes)
        else: # if student is passed, if output layer diff num_classes, replace
          self.student = student
          if self.student.fc_final_output != num_classes:
            self.student.fc_final_output = num_classes
            self.student.fc_final = nn.Linear(self.student.fc_final_input, num_classes)
            self.student.batchnorm1d_final = nn.BatchNorm1d(num_classes)
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

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
      optimizer = optim.Adam(self.student.parameters(), lr=5e-4)
      return optimizer

    def get_student(self):
      return self.student


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
    self.cifar10fsl_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    class_data_indices = [[] for _ in range(10)] # each elem is list of data samples that belong to that class
    for datum_idx, (_, label) in enumerate(self.cifar10fsl_train): # assign id to every sample in dataset; iterate through, and append the ids of sample belonging to the same class to our new 2D list
        class_idx = label
        class_data_indices[class_idx].append(datum_idx)

    fsl_data_indices = [] # indices of data samples that will be part of the new fsl subset of the dataset

    # for each class, randomly pick 5 data samples and append their ids to fsl data indices list
    for class_ in class_data_indices:
      rand_indices = (torch.randperm(len(class_))[:10]).tolist() # returns first 5 of a random permutation of all digits up to length of class (i.e. num of data samples in that class); tolist bc returns tensor
      print(rand_indices)
      for rand_index in rand_indices:
        fsl_data_indices.append(class_[rand_index])
    print(fsl_data_indices)

    # uses fsl indices to create subset of full train set
    fsl_dataset = torch.utils.data.Subset(self.cifar10fsl_train, fsl_data_indices)
    return DataLoader(fsl_dataset, batch_size=10*10, num_workers=10, shuffle=True)
    # return DataLoader(self.cifar10fsl_train, batch_size=64, num_workers=10)

  def test_dataloader(self):
    return DataLoader(self.cifar10fsl_test, batch_size=64, num_workers=10)

if __name__ == '__main__':
    cifar100_data_module = CIFAR100DataModule()
    cifar10fsl_data_module = CIFAR10FSLDataModule()
    cifar10_data_module = CIFAR10DataModule()
    
    general_teacher_model = Teacher_pl(num_classes=100)
    general_teacher_trainer = pl.Trainer(accelerator="gpu", max_epochs=10)
    general_teacher_trainer.test(general_teacher_model, cifar100_data_module)
    
    general_distiller_model = Distiller_pl(num_classes=100, trained_teacher=general_teacher_model)
    general_distiller_trainer = pl.Trainer(accelerator="gpu", max_epochs=10)
    general_distiller_trainer.fit(general_distiller_model, cifar100_data_module)
    general_distiller_trainer.test(general_distiller_model, cifar100_data_module)
    general_distiller_trainer.save_checkpoint("general_distilled.ckpt")
    
    general_trained_student = general_distiller_model.get_student()
    
    # teacher fine-tuning for cifar10fsl
    specific_teacher_model = Teacher_pl(num_classes=10)
    specific_teacher_trainer = pl.Trainer(accelerator="gpu", max_epochs=10)
    specific_teacher_trainer.fit(specific_teacher_model, cifar10_data_module)
    specific_teacher_trainer.test(specific_teacher_model, cifar10_data_module)
    specific_teacher_trainer.save_checkpoint("teacher_finetuned.ckpt")
    
    specific_distiller_model = Distiller_pl(num_classes=10, trained_teacher=specific_teacher_model, student=general_trained_student)
    specific_distiller_trainer = pl.Trainer(accelerator="gpu", max_epochs=10) #callbacks=[early_stop_callback]
    specific_distiller_trainer.fit(specific_distiller_model, cifar10_data_module)
    specific_distiller_trainer.test(specific_distiller_model, cifar10_data_module)
    specific_distiller_trainer.save_checkpoint("specific_distilled.ckpt")