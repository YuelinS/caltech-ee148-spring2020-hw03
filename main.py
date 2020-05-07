from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
Run with:
python main.py --batch-size 32 --epochs 1 --log-interval 100 --model-number 1
python main.py --evaluate --load-model mnist_model.pt
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''   
    def __init__(self):
        super(Net, self).__init__()
        
        out1, out2 = 8, 16        
        lin_in = self.calculate_size(28)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=(2,2), stride=1)
        self.conv2 = nn.Conv2d(out1, out2, 4, 1)
        # self.conv3 = nn.Conv2d(out2, out3, 4, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(out2*lin_in**2, 64)
        self.fc2 = nn.Linear(64, 10)
        self.batchnorm1 = nn.BatchNorm2d(out1)
        self.batchnorm2 = nn.BatchNorm2d(out2)
        # self.batchnorm3 = nn.BatchNorm2d(out3)
        
        
    def calculate_size(self, size_in):
    
        size_out = size_in - 2
        size_out = np.floor((size_out - 2) / 2)
        return int(size_out)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout2(x)
        x = self.batchnorm2(x)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout3(x)
        # x = self.batchnorm3(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, *args):
    model.eval()    # Set the model to inference mode
        
    test_loss = 0
    test_correct = 0
    test_num = 0   
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_correct, test_num,
        100. * test_correct / test_num))
    
    if len(args)>0:       
        train_eval_loader = args[0]
        train_loss = 0
        train_correct = 0
        train_num = 0 
        with torch.no_grad():  
            for data, target in train_eval_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    train_correct += pred.eq(target.view_as(pred)).sum().item()
                    train_num += len(data)
        train_loss /= train_num
    
        print('Full Training Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, train_correct, train_num,
            100. * train_correct / train_num))
        
        return train_loss, test_loss
        
        
        

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model-number', type=int, default=2, metavar='N',
                        help='select from fcNet, ConvNet, Net (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    models = [fcNet, ConvNet, Net]
    model_sel = args.model_number
    
    version = ''
    
    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)
        
        # Set the test model
        model = models[model_sel]().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../MNIST', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Data augmentation
    
    # version = '_augmented'
    img_trasfomrm = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
        ])

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../MNIST', train=True, download=True, transform = img_trasfomrm)

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
       
    val_split = .15
    
    targets = train_dataset.targets.data.numpy()
    target_indices = [(targets == k).nonzero() for k in range(10)]
    
    subset_indices_train, subset_indices_valid = [],[]
    
    for k in range(10):
        target_indice = target_indices[k][0]               
        
        split = int(np.floor(val_split * len(target_indice)))
        
        np.random.seed(42)
        np.random.shuffle(target_indice)
        
        subset_indices_train = np.append(subset_indices_train,target_indice[split:])
        subset_indices_valid = np.append(subset_indices_valid,target_indice[:split])

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train.astype(int).tolist())
    )
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_train.astype(int).tolist())
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid.astype(int).tolist())
    )

    # Load your model 
    model = models[model_sel]().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_losses, test_losses = [],[]

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss, test_loss = test(model, device, val_loader, train_eval_loader)
            
        scheduler.step()    # learning rate scheduler

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # You may optionally save your model at each epoch here

    if args.save_model:        
        torch.save(model.state_dict(), "mnist_model" + str(model_sel) + version + ".pt")      
        np.save('../results/losses_across_epoche' + str(model_sel) + version + '.npy',[train_losses,test_losses]);


if __name__ == '__main__':
    main()
