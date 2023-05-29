#@title PyTorch Lightning - CIFAR10
# PyTorch Lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# !pip install torchmetrics -qqq
import torchmetrics

# !pip install pytorch_lightning -qqq
import pytorch_lightning as pl
import os

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3), stride=(1, 1)) #channel input number also changes with dataset
        self.batchnorm2d1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride=(1, 1))
        self.batchnorm2d2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, (3,3), stride=(1, 1))
        self.batchnorm2d3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, (3,3), stride=(1, 1))
        self.batchnorm2d4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, (3,3), stride=(1, 1))
        self.batchnorm2d5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, (3,3), stride=(1, 1))
        self.batchnorm2d6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*18*18, 512) # need to change this depending on dataset!
        self.batchnorm1d1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.batchnorm1d2 = nn.BatchNorm1d(512)
        self.fc_final = nn.Linear(512, 10)
        self.batchnorm1d_final = nn.BatchNorm1d(10)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.batchnorm2d1(x)
        x = self.act(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.batchnorm2d2(x)
        x = self.act(x)
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.conv3(x)
        x = self.batchnorm2d3(x)
        x = self.act(x)
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.conv4(x)
        x = self.batchnorm2d4(x)
        x = self.act(x)
        # print(x.size())
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
        

class Student(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (3,3), stride=(1, 1))
        self.batchnorm2d1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3,3), stride=(1, 1))
        self.batchnorm2d2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, (3,3), stride=(1, 1))
        self.batchnorm2d3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3,3), stride=(1, 1))
        self.batchnorm2d4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, (3,3), stride=(1, 1))
        self.batchnorm2d5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, (3,3), stride=(1, 1))
        self.batchnorm2d6 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*17*17, 32)
        self.batchnorm1d1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        self.batchnorm1d2 = nn.BatchNorm1d(32)
        self.fc_final = nn.Linear(32, 10)
        self.batchnorm1d_final = nn.BatchNorm1d(10)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm2d1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batchnorm2d2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batchnorm2d3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.batchnorm2d4(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.batchnorm2d5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.batchnorm2d6(x)
        x = self.act(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batchnorm1d1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.batchnorm1d2(x)
        x = self.act(x)
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
      optimizer = optim.Adam(self.teacher.parameters(), lr=5e-5)
      return optimizer


class Distiller_pl(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.teacher = Teacher()
        self.student = Student()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def forward(self, x): # what happens if don't define this? Where is in called under the hood?
        return self.student.forward(x)

    def student_loss(self, student_predictions, labels):
      criterion = nn.CrossEntropyLoss()
      student_loss = criterion(student_predictions, labels)
      return student_loss
      
    def total_loss(self, teacher_predictions, student_predictions, labels):
      distillation_criterion = nn.KLDivLoss()
      alpha=0.1
      temperature=1
      student_loss = self.student_loss(student_predictions, labels)
      distillation_loss = (
          distillation_criterion(
              torch.nn.functional.softmax(teacher_predictions / temperature, dim=1),
              torch.nn.functional.softmax(student_predictions / temperature, dim=1),
          )
          * temperature**2
      )
      return alpha*student_loss + (1-alpha)*distillation_loss

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
      optimizer = optim.Adam(self.student.parameters(), lr=5e-5)
      return optimizer


class CIFAR10DataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.5,), (0.5,), (0.5,))])
      
    # prepare transforms standard to MNIST
    self.cifar10_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    self.cifar10_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.cifar10_train, batch_size=64, num_workers=10)

  def test_dataloader(self):
    return DataLoader(self.cifar10_test, batch_size=64, num_workers=10)

data_module = CIFAR10DataModule()

early_stop_callback = EarlyStopping(
   monitor='train_acc',
   min_delta=0.005,
   patience=3,
   verbose=False,
   mode='max'
)

# run 200 epochs with lr 5e-5 w Adam, read JADE documentation, use FLOP count tool to get count for resnet50 and CNV - can use 

# train teacher
teacher_model = Teacher_pl()
teacher_trainer = pl.Trainer(accelerator="gpu", max_epochs=200)
teacher_trainer.fit(teacher_model, data_module)
teacher_trainer.test(teacher_model, data_module)
teacher_trainer.save_checkpoint("teacher.ckpt")

# teacher_model = Teacher_pl.load_from_checkpoint(checkpoint_path="teacher.ckpt")
# teacher_trainer = pl.Trainer(accelerator="auto", max_epochs=20)
# teacher_trainer.test(teacher_model, data_module)

# distill to student
distiller_model = Distiller_pl()
distiller_trainer = pl.Trainer(accelerator="gpu", max_epochs=20, callbacks=[early_stop_callback])
distiller_trainer.fit(distiller_model, data_module)
distiller_trainer.test(distiller_model, data_module)

# Can clearly see, if comment out teacher_trainer.fit, test accuracy becomes ~0.1 (i.e. basically random for 10 classes). But if train distillation, still get good test accuracy
# This is because student_loss still taken into account to train; setting alpha=0.0 (i.e. only distillation loss considered) and then training distiller results in ~0.1 test accuracy of student, as expected, if teacher untrained    