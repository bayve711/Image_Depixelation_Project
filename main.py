import os
from torch.utils.data import random_split
import json
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from architectures import DualCNN
from datasets import RandomImagePixelationDataset
from utils import plot
from datasets import stack_with_padding
import pickle as pkl
from datasets import PickleDataset
from submission_serialization import serialize


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Check the length of `data` and unpack accordingly
            if len(data) == 3:
                inputs, knowns, targets = data
                targets = targets.to(device).float()
            elif len(data) == 2:
                inputs, knowns = data
                targets = None
            else:
                raise ValueError(f"Unexpected number of elements in `data`: {len(data)}")

            inputs = inputs.to(device)
            knowns = knowns.to(device).float()

            outputs = model(inputs, knowns)

            if targets is not None:
                loss += loss_fn(outputs, targets).item()
    loss /= len(dataloader)
    model.train()
    return loss




def main(results_path, network_config: dict, learning_rate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 500, device: str = "cpu"):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    device = torch.device(device)
    np.random.seed(0)
    torch.manual_seed(0)

    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)
    with open(r"/Users/bayve/Desktop/JKU Python 2/Assignment 7/test_set.pkl", "rb") as f:
        data = pkl.load(f)
        test_dataset = PickleDataset(data)

    dataset = RandomImagePixelationDataset('train_pics', (4, 32), (4, 32), (4, 16), dtype=np.uint8)


    # trainset = torch.utils.data.Subset(
    #     dataset,
    #     indices=np.arange(int(len(dataset) * (3 / 5))))
    # validationset = torch.utils.data.Subset(
    #     dataset,
    #     indices=np.arange(int(len(dataset) * (3 / 5)), int(len(dataset) * (4 / 5))))
    # testset = torch.utils.data.Subset(
    #     dataset,
    #     indices=np.arange(int(len(dataset) * (4 / 5)), len(dataset)))
    trainset_size = int(len(dataset) * 0.8)
    validset_size = len(dataset) - trainset_size

    trainset, validationset = torch.utils.data.random_split(dataset, [trainset_size, validset_size])

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        collate_fn=stack_with_padding
    )
    validloader = torch.utils.data.DataLoader(
        validationset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        collate_fn=stack_with_padding
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    net = DualCNN(n_in_channels = 2, n_hidden_layers = 9, n_kernels = 128, kernel_size= 3)
    net.to(device)

    mse = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print_stats_at = 5  # print status to tensorboard every x updates
    plot_at = 5  # plot every x updates
    validate_at = 5  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    while update < n_updates:
        for data in trainloader:
            inputs, knowns, targets = data
            inputs = inputs.to(device)
            knowns = knowns.to(device).float()
            targets = targets.to(device).float()

            # Reset gradients
            optimizer.zero_grad()
            outputs = net(inputs, knowns)

            # Get outputs of our network
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()

            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plotpath, update)

            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, dataloader=validloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(),
                                         global_step=update)
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")


    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, dataloader=validloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, dataloader=testloader, loss_fn=mse, device=device)

    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)

    predictions_list = []
    for data in testloader:
        inputs_final, knowns_final, _ = data
        inputs_final = inputs_final.to(device)
        knowns_final = knowns_final.to(device).float()
        with torch.no_grad():
            outputs = net(inputs_final, knowns_final)
        prediction = outputs.detach().cpu().numpy().astype(np.uint8)
        predictions_list.append(prediction)
    serialize(predictions_list, "/Users/bayve/Desktop/JKU Python 2/Assignment 7/my_project/predictions.txt")


with open("/Users/bayve/Desktop/JKU Python 2/Assignment 7/my_project/working_config.json") as cf:
    config = json.load(cf)
main(**config)