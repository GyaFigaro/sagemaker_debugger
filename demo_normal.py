from __future__ import print_function

# Standard Library
import argparse

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# First Party
import time
from tqdm import tqdm
import numpy as np
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig

from rule.rule_run import dataset_debug, epoch_debug, classfier_debug

Losses1 = []
Losses2 = []
predictions = []
labels = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
        self.add_module("sigmoid", nn.Sigmoid())
        self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
        self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
        self.add_module("tanh", nn.Tanh())
        self.add_module("max_pool2", nn.MaxPool2d(2, stride=2))
        self.add_module("fc1", nn.Linear(4 * 4 * 50, 500))
        self.add_module("relu", nn.ReLU())
        self.add_module("fc2", nn.Linear(500, 10))

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.max_pool(x)
        x = self.tanh(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    global Losses
    model.train()
    count = 0
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave = True):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad=True))
        loss = criterion(output, target)
        loss.backward()
        count += 1
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        # # ?????????????????????
        # if batch_idx % args.log_interval == 0:
        #     print(
        #         '[epoch %d] train_loss: %.4f  test_accuracy: %.4f  train_time: %f s\n' % (
        #             epoch + 1, 
        #             loss.item(), 
        #             correct, 
        #             (time.perf_counter() - time_start)
        #         )
        #     )
    
    return 100.0 * correct / len(train_loader.dataset)

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions.append(pred)
            labels.append(target)
            Losses2.append(criterion(output, target).item())

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    
    return 100.0 * correct / len(test_loader.dataset)

# Create a hook. The initilization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initilization that enables logging of
# weights, biases and gradients in the model.
def create_hook(output_dir, module=None, hook_type="saveall"):
    # Create a hook that logs weights, biases, gradients and inputs/ouputs of model every 10 steps while training.
    if hook_type == "saveall":
        hook = Hook(
            out_dir=output_dir,
            save_config=SaveConfig(save_interval=100),
            save_all=True,
        )
    elif hook_type == "module-input-output":
        # The names of input and output tensors of a module are in following format
        # Inputs :  <module_name>_input_<input_index>, and
        # Output :  <module_name>_output
        # In order to log the inputs and output of a module, we will create a collection as follows:
        assert module is not None

        # Create a hook that logs weights, biases, gradients and inputs/outputs of model every 5 steps from steps 0-100 while training.
        hook = Hook(
            out_dir=output_dir,
            save_config=SaveConfig(save_interval=100),
            include_collections=["weights", "gradients", "biases", "l_mod"],
        )
        hook.get_collection("l_mod").add_module_tensors(module, inputs=True, outputs=True)
    elif hook_type == "weights-bias-gradients":
        save_config = SaveConfig(save_interval=100)
        # Create a hook that logs ONLY weights, biases, and gradients every 5 steps (from steps 0-100) while training the model.
        hook = Hook(out_dir=output_dir, save_config=save_config)
    return hook


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        help="output directory to save data in",
        default="./tmp/testing/demo",
    )
    parser.add_argument(
        "--hook-type",
        type=str,
        choices=["saveall", "module-input-output", "weights-bias-gradients"],
        default="saveall",
    )
    parser.add_argument("--mode", action="store_true")
    parser.add_argument(
        "--rule_type", choices=["vanishing_grad", "exploding_tensor", "none"], default="none"
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
    )

    print("Input_Balance Result:", end=" ")
    print(dataset_debug('balance', train_loader))
    print("Not_Normalized_Data Result:", end=" ")
    print(dataset_debug('normalize', train_loader))

    model = Net().to(device)

    if args.rule_type == "vanishing_grad":
        lr, momentum = 1.0, 0.9
    elif args.rule_type == "exploding_tensor":
        lr, momentum = 1000000.0, 0.9
    else:
        lr, momentum = args.lr, args.momentum

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.modules.NLLLoss()
    criterion_test = torch.nn.modules.NLLLoss(reduction="sum")

    hook = create_hook(output_dir=args.output_uri, module=model, hook_type=args.hook_type)
    hook.register_hook(model)
    hook.register_loss(criterion)

    accuracy = []

    for epoch in range(1):
        hook.set_mode(smd.modes.TRAIN)
        print("THIS IS EPOCH:  ", epoch)
        accuracy.append(train(args, model, device, train_loader, optimizer, epoch, criterion))
        hook.set_mode(smd.modes.EVAL)
        accuracy.append(test(args, model, device, test_loader, criterion))
        epoch_debug(epoch, 938)

    filename = 'accuracy'
    np.save(filename,accuracy)    
    print("Classifier_Confusion Result:", end=" ")
    print(classfier_debug(10, labels, predictions))
        

if __name__ == "__main__":
    main()
