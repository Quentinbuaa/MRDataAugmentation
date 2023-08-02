from __future__ import print_function
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR
from MT import *

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss(size_average=False)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def GetOurTrainDataset(dataset, length,target_mr_transform):
    s_0, s_1= torch.utils.data.random_split(dataset, lengths = [length, len(dataset) - length])
    s_2, s_3 = torch.utils.data.random_split(s_0, lengths = [int(length/2), int(length/2)])
    s_2_dash = transformed_dataset(s_2, target_mr_transform)
    s = torch.utils.data.ConcatDataset([s_2, s_2_dash, s_1])
    return s

def GetRandomTrainDataset(dataset, length,target_mr_transform):
    s_0, s_1= torch.utils.data.random_split(dataset, lengths = [length, len(dataset) - length])
    s_2, s_3 = torch.utils.data.random_split(s_0, lengths = [int(length/2), int(length/2)])
    s_2_dash = transformed_dataset(s_2, target_mr_transform)
    s = torch.utils.data.ConcatDataset([s_2_dash, s_3, s_1])
    return s

def GetDataset(strategy, dataset, length, target_mr_transform):
    if strategy == 'ours':
        return GetOurTrainDataset(dataset, length, target_mr_transform)
    if strategy == 'none':
        return dataset
    if strategy == 'replace':
        return GetRandomTrainDataset(dataset, length, target_mr_transform)


def getFileName(strategy,name):
    return 'result-{}-{}.csv'.format(name,strategy)

def save_info(results, file):
    result = pd.DataFrame(results, columns=['m', 'MR', 'ACC', 'MTS'])
    result.to_csv(file)

def prepare_env(checkpoint_file):
    model = LeNet5()
    torch.save(model.state_dict(),checkpoint_file )

def GetMRs():
    invert_mr = Invert()
    invert_mr.SetM([5000])
    shift_mr = HorizontalTranslation(5)
    shift_mr.SetM([15000])
    return [shift_mr]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--m', type=int, default=100,
                        help='Set the number of M')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
   # m = args.m
    #m_list = [20000]
    m_list = range(1000,20000,5000)
    #target_mr_transform_list = [TF.invert,Rotate(),Scale(0.9)]
    #target_mr_transform_list = [Invert(), Rotate(10), Scale(0.4), HorizontalTranslation(5), VerticalTranslation(5), Shear(30)]
    #target_mr_transform_list = [Scale(0.4), HorizontalTranslation(5), VerticalTranslation(5), Shear(30)]
    #target_mr_transform_list = [Shear(30)]
    checkpoint_file = "mnist_original.pt"
    prepare_env(checkpoint_file)
    target_mr_transform_list = GetMRs()
    model = LeNet5()
    for strategy in ['none', 'ours']:
        result_list = []
        for tf_index in range(len(target_mr_transform_list)):
            for m in target_mr_transform_list[tf_index].GetM():
                target_mr_transform = target_mr_transform_list[tf_index]
                dataset1 =GetDataset(strategy, dataset1, m, target_mr_transform)
                train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
                test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
                #model = Net().to(device) # Some net.
                model.load_state_dict(torch.load(checkpoint_file))
                model.to(device)
                #model= LeNet5().to(device) # The LeNet5
                #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
                scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
                for epoch in range(1, args.epochs + 1):
                    train(args, model, device, train_loader, optimizer, epoch)
                    scheduler.step()
                acc = test(model, device, test_loader)
                mts = mt(model, device, test_loader, target_mr_transform)
                result = [m, tf_index, acc, mts]
                result_list.append(result)
                file = getFileName(strategy, target_mr_transform.name)
        save_info(result_list, file)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()