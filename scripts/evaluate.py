import sys
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
import torch

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tifffile

# import relative packages
sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import restitch_eval, restitch_and_plot

ROOT = Path.cwd()


@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = ROOT / "data" / "processed" / "4x4"
    raw_dir: str | os.PathLike = ROOT / "data" / "raw" / "Train"
    results_dir: str | os.PathLike = ROOT / "data" / "predictions" / "UNet"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = ROOT / "models" / "UNet" / "last.ckpt"


def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Load datamodule
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # load model from checkpoint at options.model_path
    model = ESDSegmentation("UNet", 99, 4)
    checkpoint = torch.load(options.model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # set the model to evaluation mode (model.eval())
    # this is important because if you don't do this, some layers
    # will not evaluate properly
    model.eval()

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer(
        devices=1,
        max_epochs=1,
        logger=pl.loggers.TensorBoardLogger("logs/", name="my_model"),
    )
    trainer.fit(model=model, datamodule=datamodule)

    # run the validation loop with trainer.validate
    trainer.validate(
        model,
        ckpt_path=options.model_path,
        datamodule=datamodule,
    )

    val_parent_tiles = set(int(filename.name[4]) if filename.name[5] == '_' else int(filename.name[4:6]) for filename in list(Path(options.processed_dir / 'Val' / 'subtiles').glob('*')))

    # run restitch_and_plot

    # for every subtile in options.processed_dir/Val/subtiles
    # run restitch_eval on that tile followed by picking the best scoring class
    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib
    # tiles = ...
    for parent_tile_id in val_parent_tiles:
        restitch_and_plot(options = options, datamodule = datamodule, model = model, parent_tile_id = parent_tile_id, image_dir = options.results_dir)


if __name__ == "__main__":
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(EvalConfig(**parser.parse_args().__dict__))
