"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
import os
import numpy as np
from pathlib import Path
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from contin_vae.models import VAE, RacingcarDataset
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from contin_vae.utils import init_weights


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--experiment-name", default="tmp", type=str)
    parser.add_argument("--gpus", type=str, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast-dev-run", default=False, action="store_true")
    parser.add_argument("--log-interval-generation", default=5, type=int)
    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = VAE.add_model_specific_args(parser)
    # parse params
    hparams = parser.parse_args()
    if hparams.seed is None:
        hparams.seed = np.random.randint(1, 2**32)
    print("\n", hparams, "\n")

    # partition data, if not already done
    hparams.datadir = str(Path(hparams.datapath).parent)
    if os.path.isfile("%s/train_stack%d.pt" % (hparams.datadir, hparams.img_stack)) and \
            os.path.isfile("%s/test_stack%d.pt" % (hparams.datadir, hparams.img_stack)):
        pass
    else:
        print("Creating Train/Test-Split...")
        RacingcarDataset.save_train_test_split(hparams.datapath,
            seed=hparams.seed, img_stack=hparams.img_stack)

    # set up logger and checkpointing
    logger = TensorBoardLogger("runs", name=hparams.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        filepath="runs/%s/version_%d/checkpoints" % (hparams.experiment_name, logger.version))

    # init model and trainer
    model = VAE(hparams)
    model.apply(init_weights)
    trainer = Trainer(
        logger=logger,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        fast_dev_run=hparams.fast_dev_run,
        early_stop_callback=hparams.early_stop_callback,
        checkpoint_callback=checkpoint_callback)

    # start training
    trainer.fit(model)
