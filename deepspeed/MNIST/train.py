from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.distributed as dist
import deepspeed


def init_distributed_mode(args):
    deepspeed.init_distributed()



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


def train(args, model_engine, train_loader, fp16, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)
        if fp16:
            data = data.half()
        output = model_engine(data)
        loss = F.nll_loss(output, target)
        model_engine.backward(loss)
        model_engine.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, dist.get_world_size() * batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, model_engine, test_loader, fp16):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)
            if fp16:
                data = data.half()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument("--save_path", type=str, default="",
                        help="path used to save model")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    init_distributed_mode(args)

    torch.manual_seed(args.seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('/TData/data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('/TData/data', train=False,
                       transform=transform)
    test_kwargs = {'batch_size': args.test_batch_size,
                   'num_workers': 2}
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler = test_sampler, **test_kwargs)

    model = Net()
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_dataset)

    fp16 = model_engine.fp16_enabled()
    print(f'fp16={fp16}')


    for epoch in range(1, args.epochs + 1):
        train(args, model_engine, train_loader, fp16, epoch)
        if dist.get_rank() == 0:
            test(model, model_engine, test_loader, fp16)

    if args.save_model:
        if dist.get_rank() == 0:
            # only save model on RANK0 process.
            print(f'Save the model to {os.path.join(args.save_path, "mnist_cnn.pt")}')
            torch.save(model.state_dict(), os.path.join(args.save_path, "mnist_cnn.pt"))


if __name__ == '__main__':
    main()
