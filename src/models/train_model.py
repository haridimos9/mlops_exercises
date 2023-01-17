import argparse
import os
import matplotlib.pyplot as plt
import torch

from src.models.model import MyAwesomeModel
from src.data.make_dataset import CorruptMnist
import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name='config.yaml')
def training(cfg) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    hparams = cfg.experiment.hyperparameters
    hoptimizer = cfg.optimizers.Optimizer
    hdirs = cfg.experiment.dirs

    data_input_filepath = hdirs.input_path
    data_output_filepath = hdirs.output_path

    lr = hparams.lr
    batch_size = hparams.batch_size
    epochs = hparams.epochs
    seed = hparams.seed

    torch.manual_seed(seed)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = MyAwesomeModel()
    model = model.to(device)

    train_set = CorruptMnist(train=True, in_folder=data_input_filepath, out_folder=data_output_filepath)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    optimizer_dict = {
        "AdamW": torch.optim.AdamW(model.parameters(), lr=lr),
        "SGD": torch.optim.SGD(model.parameters(), lr=lr),
    }

    optimizer = optimizer_dict[hoptimizer[0].optimizer]
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epoch = epochs
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            print(x.shape)
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
    print(os.path.dirname(__file__))
    torch.save(model.state_dict(), os.path.dirname(__file__)+"/../../models/trained_model.pt")
    print("I made it up to here.")
    plt.plot(loss_tracker, "-")
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.savefig(os.path.dirname(__file__)+"/../../reports/figures/training_curve.png")


if __name__ == "__main__":
    training()
