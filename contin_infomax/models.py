import pytorch_lightning as pl
import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from argparse import ArgumentParser
from contin_infomax.datasets import RacingcarMultiviewDataset
from contin_infomax.utils import Flatten
from contin_infomax.utils import LinearRegressor, NonlinearRegressor
from torch.optim.lr_scheduler import CosineAnnealingLR


class InfoMax(pl.LightningModule):

    def __init__(self, hparams):
        super(InfoMax, self).__init__()
        self.hparams = hparams

        # init encoder
        self.encoder = nn.Sequential(  # input shape (1, 96, 96)  # NOTE: img_stack only used for contrasting
            nn.Conv2d(1, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),
            Flatten(),
            nn.Linear(256, hparams.ndim)
        )  # output shape ndim

        # init contrastive map h_i -> z_i
        if hparams.nonlinear_projection:
            self.contrastive_map = nn.Sequential(
                nn.Linear(hparams.ndim, hparams.ndim),
                nn.ReLU(),
                nn.Linear(hparams.ndim, hparams.ndim_projection)
            )
        else:
            self.contrastive_map = nn.Sequential(
                nn.Linear(hparams.ndim, hparams.ndim_projection),
            )

        # init contrastive map h_i -> z_i
        # TODO: use nonlinear projection for actions?
        self.contrastive_map_action = nn.Sequential(
            nn.Linear(3, hparams.ndim_projection),
            # nn.ReLU(),
            # nn.Linear(hparams.ndim, hparams.ndim_projection)
        )

        # init action classifiers
        num_actions = [3, 2, 2]  # steering, acceleration, braking
        self.action_classifiers_linear = []
        self.action_classifiers_nonlinear = []
        for n in num_actions:
            self.action_classifiers_linear.append(LinearRegressor(num_hidden=hparams.ndim).cuda())
            self.action_classifiers_nonlinear.append(NonlinearRegressor(num_hidden=hparams.ndim).cuda())

    def forward(self, x):
        h = self.encoder(x)
        return h

    def compute_contrastive_loss(self, h1, h2):

        # compute cosine similarity matrix C of size 2N * (2N - 1), w/o diagonal elements
        z1 = self.contrastive_map(h1)
        z2 = self.contrastive_map(h2)
        z1_normalized = F.normalize(z1, dim=-1)
        z2_normalized = F.normalize(z2, dim=-1)
        z = torch.cat([z1_normalized, z2_normalized], dim=0)  # 2N * D
        C = torch.mm(z, z.T.contiguous())  # 2N * 2N
        # remove diagonal elements from C
        mask = torch.eye(2 * self.hparams.batch_size, device=C.device).bool().logical_not()
        C = C[mask].view(2 * self.hparams.batch_size, -1)  # 2N * (2N - 1)

        # compute loss
        numerator = 2 * torch.sum(z1_normalized * z2_normalized) / self.hparams.tau
        denominator = torch.logsumexp(C / self.hparams.tau, dim=-1).sum()
        loss = (denominator - numerator) / (2 * self.hparams.batch_size)
        return loss, z1, z2


    def compute_contrastive_loss_stateaction(self, h1, y):

        # compute cosine similarity matrix C of size 2N * (2N - 1), w/o diagonal elements
        z1 = self.contrastive_map(h1)
        z2 = self.contrastive_map_action(y)
        z1_normalized = F.normalize(z1, dim=-1)
        z2_normalized = F.normalize(z2, dim=-1)
        z = torch.cat([z1_normalized, z2_normalized], dim=0)  # 2N * D
        C = torch.mm(z, z.T.contiguous())  # 2N * 2N
        # remove diagonal elements from C
        mask = torch.eye(2 * self.hparams.batch_size, device=C.device).bool().logical_not()
        C = C[mask].view(2 * self.hparams.batch_size, -1)  # 2N * (2N - 1)

        # compute loss
        numerator = 2 * torch.sum(z1_normalized * z2_normalized) / self.hparams.tau
        denominator = torch.logsumexp(C / self.hparams.tau, dim=-1).sum()
        loss = (denominator - numerator) / (2 * self.hparams.batch_size)
        return loss, z1, z2

    def evaluate_classifiers(self, z, y, train=False, logname=""):
        for i, clf in enumerate(self.action_classifiers_linear):
            if train:
                out, _ = clf.train(z, y[:, i])
            else:
                out = clf(z)
            rmse = torch.sqrt(F.mse_loss(out.view(-1), y[:, i]))
            self.logger.experiment.add_scalars(
                "%slinear_regression_rmse" % logname, {"%d" % i: rmse}, self.current_epoch)
        for i, clf in enumerate(self.action_classifiers_nonlinear):
            if train:
                out, _ = clf.train(z, y[:, i])
            else:
                out = clf(z)
            rmse = torch.sqrt(F.mse_loss(out.view(-1), y[:, i]))
            self.logger.experiment.add_scalars(
                "%snonlinear_regression_rmse" % logname, {"%d" % i: rmse}, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        ix_rand = np.random.randint(self.hparams.img_stack)
        y = y[:, ix_rand]

        # spatial contrasting
        h1_spat = self.forward(x1[:, [ix_rand]])
        h2_spat = self.forward(x2[:, [ix_rand]])
        loss_spatial, z1, z2 = self.compute_contrastive_loss(h1_spat, h2_spat)
        self.logger.experiment.add_scalars("0_train/",
            {"contrastive_loss_spatial": loss_spatial}, self.current_epoch)

        # temporal contrasting
        if self.hparams.img_stack > 1:
            ix_rand = np.random.choice(self.hparams.img_stack - 1, replace=False)
            h1_temp = self.forward(x1[:, [ix_rand]])
            h2_temp = self.forward(x1[:, [-1]])  # NOTE: same view by purpose
            loss_temporal, _, _ = self.compute_contrastive_loss(h1_temp, h2_temp)
            self.logger.experiment.add_scalars("0_train/",
                {"contrastive_loss_temporal": loss_temporal}, self.current_epoch)
        else:
            loss_temporal = 0.

        # state/action contrasting
        if self.hparams.stateaction_contrasting:
            loss_stateaction, _, _ = self.compute_contrastive_loss_stateaction(h1_spat, y)
            self.logger.experiment.add_scalars("0_train/",
                {"contrastive_loss_stateaction": loss_stateaction}, self.current_epoch)
        else:
            loss_stateaction = 0.

        # train action classifiers on frozen embeddings
        self.evaluate_classifiers(h1_spat.detach(), y, train=True, logname="0_train/")

        # debug logs
        if hasattr(self, "schedule"):
            self.logger.experiment.add_scalar("debug/learning_rate", self.schedule.get_last_lr()[0], self.current_epoch)

        total_loss = loss_spatial + loss_temporal + loss_stateaction
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        ix_rand = np.random.randint(self.hparams.img_stack)
        y = y[:, ix_rand]

        # spatial contrasting
        h1_spat = self.forward(x1[:, [ix_rand]])
        h2_spat = self.forward(x2[:, [ix_rand]])
        loss_spatial, z1, z2 = self.compute_contrastive_loss(h1_spat, h2_spat)
        self.logger.experiment.add_scalars("1_val/",
            {"contrastive_loss_spatial": loss_spatial}, self.current_epoch)

        # temporal contrasting
        if self.hparams.img_stack > 1:
            ix_rand = np.random.choice(self.hparams.img_stack - 1, replace=False)
            h1_temp = self.forward(x1[:, [ix_rand]])
            h2_temp = self.forward(x1[:, [-1]])  # NOTE: same view by purpose
            loss_temporal, _, _ = self.compute_contrastive_loss(h1_temp, h2_temp)
            self.logger.experiment.add_scalars("1_val/",
                {"contrastive_loss_temporal": loss_temporal}, self.current_epoch)
        else:
            loss_temporal = 0.

        # evaluate action classifiers on frozen embeddings
        self.evaluate_classifiers(h1_spat.detach(), y, train=False, logname="1_val/")

        return {"val_loss": loss_spatial + loss_temporal}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}  # NOTE: required for checkpoints
        if 0 == self.current_epoch % self.hparams.log_interval_generation:
            self.log_inputs()
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def log_inputs(self):
        x1, x2, _ = self.sample
        ix = np.random.choice(len(x1), 10, replace=False)
        x = torch.cat((x1[ix], x2[ix]), dim=0)
        img = make_grid(x[:, [0]], nrow=10)
        self.logger.experiment.add_image('TrainingInputs', img, self.current_epoch)
        return img

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.lr_annealing:
            schedule = CosineAnnealingLR(optim, T_max=50)
            self.schedule = schedule
            return [optim], [schedule]
        else:
            return optim

    @pl.data_loader
    def train_dataloader(self):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(96),
            transforms.ToTensor()])
        # TODO: implement colorjitter
        dl = DataLoader(RacingcarMultiviewDataset(self.hparams.datadir, train=True, transform=trans, img_stack=self.hparams.img_stack),
                        batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
        self.num_train_batches = len(dl)
        # save a sample of training data
        _, batch = next(enumerate(dl))
        x1, x2, y = batch
        self.sample = (x1.cuda(), x2.cuda(), y.cuda())
        return dl

    @pl.data_loader
    def val_dataloader(self):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])
        dl = DataLoader(RacingcarMultiviewDataset(self.hparams.datadir, train=False, transform=trans, img_stack=self.hparams.img_stack),
                        batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
        self.num_val_batches = len(dl)
        return dl

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--max-epochs', default=1000, type=int)
        parser.add_argument('--learning-rate', default=0.001, type=float)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
        parser.add_argument('--ndim', type=int, default=256, help='number of latent dimensions')
        parser.add_argument('--ndim-projection', type=int, default=64, help='number of latent dimensions for contrastive map')
        parser.add_argument('--tau', type=float, default=1.0, help='temperature parameter')
        parser.add_argument('--early-stop-callback', default=False, action="store_true")
        parser.add_argument('--lr-annealing', default=False, action="store_true")
        parser.add_argument('--stateaction-contrasting', default=False, action="store_true")
        parser.add_argument('--nonlinear-projection', default=False, action="store_true")
        return parser
