import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path
import pytorch_lightning as pl
import wandb
import torch
import numpy as np
from pytorch_lightning.loggers import WandbLogger
sys.path.append(".")
from src.dataset.datamodule import MGZDataModule
from models.model_module import MGZSegmentation
from src.utilities import ESDConfig, PROJ_NAME
ROOT = Path.cwd()

def main(options: ESDConfig):
    datamodule = MGZDataModule(processed_dir=options.processed_dir, 
                               raw_dir=options.raw_dir, 
                               batch_size=options.batch_size,
                               seed=options.seed,
                               slice_size=options.slice_size,
                               num_workers=options.num_workers
    )
    datamodule.prepare_data()
    model = MGZSegmentation.load_from_checkpoint("models/FCNResnetTransfer/last.ckpt")
    datamodule.setup("fit")
    eval_dataset = datamodule.val_dataset
    img, mask = eval_dataset[23]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img, mask = img.unsqueeze(0).to(device), mask.to(device)
    pred = model(img)
    print(mask)
    print(pred)
    fig, ax = plt.subplots(1, 3, figsize=(4, 4), squeeze=False, tight_layout=True)
    img = img.detach().cpu()
    mask = mask.detach().cpu()
    pred = pred.detach().cpu()
    print(type(img))
    img = np.uint8(torch.squeeze(img, 0).permute(1, 2, 0))
    pred = np.uint8(torch.squeeze(pred, 0).argmax(dim=0))
    ax[0,0].imshow(img)
    ax[0,1].imshow(mask)
    ax[0,2].imshow(pred)
    plt.show()

if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        help="The model to initialize.",
        default=config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate for training model",
        default=config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.in_channels,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.out_channels,
        help="Number of output channels",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the encoders (CNN only)",
        default=config.depth,
    )
    parser.add_argument(
        "--n_encoders",
        type=int,
        help="Number of encoders (Unet only)",
        default=config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size of the neural network (CNN/Unet)",
        default=config.embedding_size,
    )
    parser.add_argument(
        "--pool_sizes",
        help="A comma separated list of pool_sizes (CNN only)",
        type=str,
        default=config.pool_sizes,
    )
    parser.add_argument(
        "--kernel_size",
        help="Kernel size of the convolutions",
        type=int,
        default=config.kernel_size,
    )

    parse_args = parser.parse_args()

    config = ESDConfig(**parse_args.__dict__)
    main(config)