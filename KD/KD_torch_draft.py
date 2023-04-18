# PyTorch Lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import torchmetrics

import pytorch_lightning as pl
import os


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(256, 256)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(256, 256)
        self.act4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc_final = nn.Linear(256, 10)

    # sooo - discovered Keras mis-documented padding=same (https://github.com/keras-team/keras/issues/15703), bc setting stride>1 yields output size diff to input
    # which means output of e.g. conv1 was 14x14 instead of 28x28
    # Hence, after looking at model summary of ref KD and checking dimensions, basically did educated trial and error to get padding here in PyTorch Lightning to yield same output sizes at each layer, to compare
    # Many key points, incl: 'padding' arg of nn.Conv2D does not allow diff left/right or top/bottom padding, only one tuple of (left/right, top/bot), so odd kernel size screws things up
    # Hence have to use F.pad to explicitly pad left and right, top and bottom separately
    # Also, padding='same' only supported for stride=(1,1), in nn.Conv2D
    def forward(self, x):
        # print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.fc_final(x)
        return x

# Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
# stride_height * (out_height - 1) - in_height + kernel_height = pad_top + pad_bottom  # set out_height=in_height for padding=same
# 2*(28-1) - 28 + 3 = 29 ; so do pad_top=15, pad_bottom=14; identical for width, so do pad_left=15, pad_right=14
class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(64, 64)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(64, 64)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(64, 64)
        self.act4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc_final = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.fc_final(x)
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
      temperature=10
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
      optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
      return optimizer


class MNISTDataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.5,), (0.5,))])
      
    # prepare transforms standard to MNIST
    self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=64, num_workers=10)

  def test_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=64, num_workers=10)


if __name__ == "__main__":

  data_module = MNISTDataModule()

  # train teacher
  teacher_model = Teacher_pl()
  teacher_trainer = pl.Trainer(accelerator="auto", max_epochs=5)
  teacher_trainer.fit(teacher_model, data_module)
  teacher_trainer.test(teacher_model, data_module)

  # distill to student
  distiller_model = Distiller_pl()
  distiller_trainer = pl.Trainer(accelerator="auto", max_epochs=3)
  distiller_trainer.fit(distiller_model, data_module)
  distiller_trainer.test(distiller_model, data_module)

# Can clearly see, if comment out teacher_trainer.fit, test accuracy becomes ~0.1 (i.e. basically random for 10 classes). But if train distillation, still get good test accuracy
# This is because student_loss still taken into account to train; setting alpha=0.0 (i.e. only distillation loss considered) and then training distiller results in ~0.1 test accuracy of student, as expected, if teacher untrained    