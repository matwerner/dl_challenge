import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from helper import DeepEquationDataset
from model import DeepEquationNet

# Reference:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (instance, target) in enumerate(train_loader):
        image_a, image_b, operator = instance
        target_a, target_b, target_eq = target
        image_a, image_b, operator = image_a.to(device), image_b.to(device), operator.to(device)
        target_a, target_b, target_eq = target_a.to(device), target_b.to(device), target_eq.to(device)        

        optimizer.zero_grad()
        output_a, output_b, output_eq = model(image_a, image_b, operator)
        loss_a = F.nll_loss(output_a, target_a)
        loss_b = F.nll_loss(output_b, target_b)
        loss_eq = F.nll_loss(output_eq, target_eq)
        loss = loss_a + loss_b + loss_eq
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image_a), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_a = 0
    correct_b = 0
    correct_eq = 0
    with torch.no_grad():
        for batch_idx, (instance, target) in enumerate(test_loader):
            image_a, image_b, operator = instance
            target_a, target_b, target_eq = target
            image_a, image_b, operator = image_a.to(device), image_b.to(device), operator.to(device)
            target_a, target_b, target_eq = target_a.to(device), target_b.to(device), target_eq.to(device)

            output_a, output_b, output_eq = model(image_a, image_b, operator)

            test_loss += F.nll_loss(output_eq, target_eq, reduction='sum').item()  # sum up batch loss
            pred_a = output_a.argmax(dim=1, keepdim=True)
            pred_b = output_b.argmax(dim=1, keepdim=True)
            pred_eq = output_eq.argmax(dim=1, keepdim=True)
            correct_a += pred_a.eq(target_a.view_as(pred_a)).sum().item()
            correct_b += pred_b.eq(target_b.view_as(pred_b)).sum().item()
            correct_eq += pred_eq.eq(target_eq.view_as(pred_eq)).sum().item()
    test_loss /= len(test_loader.dataset)

    test_size = len(test_loader.dataset)
    acurracy_a_str  = 'Image A: Accuracy: {}/{} ({:.0f}%)'.format(correct_a, test_size, 100. * correct_a / test_size)
    acurracy_b_str  = 'Image B: Accuracy: {}/{} ({:.0f}%)'.format(correct_b, test_size, 100. * correct_b / test_size)
    acurracy_eq_str = 'Equation: Accuracy: {}/{} ({:.0f}%)'.format(correct_eq, test_size, 100. * correct_eq / test_size)
    print('\nTest set: Average loss: {:.4f}, \n{}\n{}\n{}\n'.format(
        test_loss, acurracy_a_str, acurracy_b_str, acurracy_eq_str))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([transforms.ToTensor()])
    data_filepath = '../../resources/'
    train_dataset = datasets.MNIST(data_filepath, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_filepath, train=False, transform=transform)

    # Split into train-validation
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataset = DeepEquationDataset(train_dataset, int(6 * len(train_dataset)))
    valid_dataset = DeepEquationDataset(valid_dataset, int(6 * len(valid_dataset)))    

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **test_kwargs)    

    model = DeepEquationNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, valid_loader)
        scheduler.step()
    
    # test_dataset = DeepEquationDataset(test_dataset, int(3 * len(test_dataset)))
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    # test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "deep_equation_net.pt")


if __name__ == '__main__':
    main()