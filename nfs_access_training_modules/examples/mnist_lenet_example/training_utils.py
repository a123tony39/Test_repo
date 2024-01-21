import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import Subset
import numpy as np
import os
import importlib
from collections import OrderedDict
from ray.air import session, Checkpoint

def setDevice(device_name = None):
    device = torch.device('cuda' if device_name == 'GPU' and torch.cuda.is_available() else 'cpu')
    return device

def setDataLoader(cid, batch_size, datasetName, datasetPath=None, transform=None):
    path = '/opt/nfsfl/{}'.format(datasetName)
    # transformer
    if transform is None:
        transform = transforms.ToTensor()
    else:
        transform_list = [eval(t) for t in transform]
        transform = transforms.Compose(transform_list)
    # create dataset # TODO: user edit
    if datasetName == "MNIST":
        train_dataset = datasets.MNIST(root = os.path.join(path,"data"), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root = os.path.join(path,"data"), train=False, download=True, transform=transform)
    elif datasetName == "CIFAR10":
        train_dataset = datasets.CIFAR10(root = os.path.join(path,"data"), train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root = os.path.join(path,"data"), train=False, download=True, transform=transform)
    else:
        raise ValueError('dataset name is None')
    # load dataset
    if datasetPath is not None and os.path.exists(datasetPath):
        if os.path.isfile(datasetPath):
            client_indices =  None
            print("Client {} loads the dataset {}".format(cid, datasetPath))
            F = open(datasetPath, 'rb')
            client_indices=torch.load(F)
            if client_indices is not None:
                train_dataset = Subset(train_dataset, client_indices)
    else:
        pass
        # print("There is no dataset to load")
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader

def setModel(cid, localModelPath, model_name=None):
    # assert False, f"\n===\n{localModelPath}\n{model_name}\n===\n"
    try:
        model_module = importlib.import_module(model_name)
        create_model_instr = "model_module.construct_model()"
        model = eval(create_model_instr)
    except:
        raise AssertionError('Import model:{} error, check your model file.'.format(model_name))

    checkpoint_epochs = 0
    if localModelPath is not None and os.path.exists(localModelPath):
        # load checkpoint_epochs
        ray_checkpoint_path = os.path.join(os.path.dirname(localModelPath), "dict_checkpoint.pkl")
        if os.path.exists(ray_checkpoint_path):
            checkpoint = Checkpoint.from_directory(os.path.dirname(localModelPath)).to_dict()
            checkpoint_epochs = checkpoint.get('epochs', checkpoint_epochs)
        # load model
        model_ndarray = np.load(localModelPath, allow_pickle=True)
        loaded_model_state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), model_ndarray)})
        model.load_state_dict(loaded_model_state_dict)
        print("Client {} loads the model at {} ,this model has training {} epochs".format(cid, localModelPath,checkpoint_epochs))
    else:
        pass
        # print("There is no Model to load")
    # summary(model, self.train_dataset[0][0].shape)
    return checkpoint_epochs, model

def setLossfunc(loss_func_name):
    # Loss_function
    loss_func_dict = {
        'MSE': nn.MSELoss(),
        'L1': nn.L1Loss(),
        'CrossEntropy': nn.CrossEntropyLoss(),
        'BCE': nn.BCELoss(),
    }
    
    if loss_func_name in loss_func_dict:
        loss_func = loss_func_dict[loss_func_name]
    else:
        print("Not support this loss_function: ", loss_func_name)
    return loss_func

def setOptimizer(model, optimizer_name, lr):
    # Optimizer
    optimizer_dict = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'Adagrad': torch.optim.Adagrad,
        'RMSprop': torch.optim.RMSprop
    }
    if optimizer_name in optimizer_dict:
        optimizer_cls = optimizer_dict[optimizer_name]
        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=1e-8)
    else:
        print('Not support this optimzer: ', optimizer_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    return optimizer

def train_loop_per_worker(train_loader, model, loss_function, optimizer, device, seed):
    # print("Client Go Training.")
    model.train()
    seed_log = "Setting seed \"{}\" for epochs.".format(seed) if isinstance(seed, int) else "Not set seed for epochs."
    print(seed_log)
    if isinstance(seed, int): torch.manual_seed(seed)
    # with tqdm(train_loader, unit="batch") as loader_t:
    for batch_idx, (image, label) in enumerate(train_loader):    
        label = nn.functional.one_hot(label, 10)
        label = label.to(torch.float)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1,
            #         batch_idx * len(image),
            #         len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader),
            #         loss.item()))
            
def model_evaluate(model, test_loader, loss_function, device):
    # size = len(test_loader.dataset) // session.get_world_size()
    size = len(test_loader.dataset)
    # print("Client Go Evaluating.")
    model.eval()
    test_loss = 0
    correct = 0

    for (data, label) in test_loader:
        label_onehot = nn.functional.one_hot(label, 10)
        label_onehot = label_onehot.to(torch.float)
        label_onehot = label_onehot.to(device)
        data, label = data.to(device), label.to(device)

        with torch.no_grad():
            output = model(data)
            test_loss += loss_function(output, label_onehot).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(label.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss,
    #     correct,
    #     size,
    #     100. * correct /size)
    # accuracy = float(100. * correct / size)
    accuracy = float(correct / size)
    loss = test_loss
    return accuracy, loss
            
