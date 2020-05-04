import pytorch_lightning as pl
import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import Compose, ToTensor, ToPILImage, Grayscale, Normalize
from argparse import ArgumentParser
from contin_vae.datasets import RacingcarDataset
from contin_vae.utils import Flatten, Unflatten
from contin_vae.utils import LinearRegressor, NonlinearRegressor


class VAE(pl.LightningModule):

    def __init__(self, hparams):
        super(VAE, self).__init__()
        self.hparams = hparams

        # init encoder
        self.encoder = nn.Sequential(  # input shape (img_stack, 96, 96)
            nn.Conv2d(self.hparams.img_stack, 8, kernel_size=4, stride=2),
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
        )  # output shape 256
        self.map_mu = nn.Linear(256, hparams.ndim)
        self.map_logvar = nn.Linear(256, hparams.ndim)

        # init decoder
        self.decoder = nn.Sequential(
            nn.Linear(hparams.ndim, 256),
            nn.ReLU(),
            Unflatten(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, self.hparams.img_stack, kernel_size=4, stride=2),
        )  # output shape (img_stack, 96, 96)

        # init action classifiers
        num_actions = [3, 2, 2]  # steering, acceleration, braking
        self.action_classifiers_linear = []
        self.action_classifiers_nonlinear = []
        for n in num_actions:
            self.action_classifiers_linear.append(LinearRegressor(num_hidden=hparams.ndim).cuda())
            self.action_classifiers_nonlinear.append(NonlinearRegressor(num_hidden=hparams.ndim).cuda())

    def encode(self, x):
        h1 = F.relu(self.encoder(x))
        h1 = h1.view(h1.shape[0], -1)
        return self.map_mu(h1), self.map_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def compute_vae_loss(self, x_hat, x, mu, logvar, annealing_coef=1.0):
        recloss = F.mse_loss(x_hat.view(-1), x.view(-1), reduction='sum') / self.hparams.img_stack
        kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / self.hparams.ndim
        loss = (recloss + kld * self.hparams.beta * annealing_coef) / self.hparams.batch_size
        return loss, recloss, kld

    def annealing_coef(self, batch_idx, train=False):
        epoch = self.current_epoch
        start_annealing = 0
        annealing_epochs = self.hparams.annealing_epochs
        annealing_coef = min((epoch - start_annealing + 1) / (annealing_epochs + 1), 1.)
        # adjust annealing coef to batches
        if train is True:
            min_coef = min((epoch - start_annealing) / (annealing_epochs + 1), 1.)
            annealing_coef = annealing_coef * ((batch_idx + 1) / (self.num_train_batches + 1))
            annealing_coef = max(min_coef, annealing_coef)
            self.logger.experiment.add_scalar('debug/AnnealingCoef', annealing_coef, epoch)
        return annealing_coef

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
        x, y = batch
        y = y[:, np.random.randint(self.hparams.img_stack)]  # TODO: HACK
        x_hat, mu, logvar, z = self.forward(x)
        annealing_coef = self.annealing_coef(batch_idx, train=True)
        loss, recloss, kld = self.compute_vae_loss(x_hat, x, mu, logvar, annealing_coef)
        self.logger.experiment.add_scalar("0_train/kld", kld, self.current_epoch)
        self.logger.experiment.add_scalar("0_train/recloss", recloss, self.current_epoch)

        # train action classifiers on frozen embeddings
        self.evaluate_classifiers(z.detach(), y, train=True, logname="0_train/")

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, np.random.randint(self.hparams.img_stack)]  # TODO: HACK
        x_hat, mu, logvar, z = self.forward(x)
        annealing_coef = self.annealing_coef(batch_idx, train=False)
        loss, recloss, kld = self.compute_vae_loss(x_hat, x, mu, logvar, annealing_coef)
        self.logger.experiment.add_scalar("1_val/kld", kld, self.current_epoch)
        self.logger.experiment.add_scalar("1_val/recloss", recloss, self.current_epoch)

        # evaluate action classifiers on frozen embeddings
        self.evaluate_classifiers(z.detach(), y, train=False, logname="1_val/")

        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}  # NOTE: required for checkpoints
        if 0 == self.current_epoch % self.hparams.log_interval_generation:
            self.log_reconstruction()
            self.log_random_generation()
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def log_reconstruction(self):
        x = self.sample[:10]
        x_hat, _, _, _ = self.forward(x)
        if self.hparams.img_stack == 1:
            img = make_grid(torch.cat((x, x_hat), dim=0), nrow=10, normalize=True)
        elif self.hparams.img_stack > 1:
            img = make_grid(torch.cat((x[:, [0]], x_hat[:, [0]]), dim=0), nrow=10, normalize=True)
        self.logger.experiment.add_image('Reconstruction', img, self.current_epoch)
        return img

    def log_random_generation(self, num_samples=64):
        samples = torch.randn(num_samples, self.hparams.ndim).cuda()
        x_hat = self.decode(samples)
        if self.hparams.img_stack == 1:
            img = make_grid(x_hat, normalize=True)
        elif self.hparams.img_stack > 1:
            img = make_grid(x_hat[:, [0]], normalize=True)
        self.logger.experiment.add_image('RandomGeneration', img, self.current_epoch)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     x_hat, mu, logvar, z = self.forward(x)
    #     loss = self.compute_loss(x_hat, x, mu, logvar)
    #     return {"test_loss": loss}

    # def test_end(self, outputs):
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     tensorboard_logs = {"test_loss": avg_loss}
    #     return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        transforms = Compose([ToTensor()])
        dl = DataLoader(RacingcarDataset(self.hparams.datadir, train=True, transform=transforms, img_stack=self.hparams.img_stack),
                           batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
        self.num_train_batches = len(dl)
        return dl

    @pl.data_loader
    def val_dataloader(self):
        transforms = Compose([ToTensor()])
        dl = DataLoader(RacingcarDataset(self.hparams.datadir, train=False, transform=transforms, img_stack=self.hparams.img_stack),
                           batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
        self.num_val_batches = len(dl)
        _, batch = next(enumerate(dl))
        self.sample = batch[0].cuda()
        return dl

    # @pl.data_loader
    # def test_dataloader(self):
    #     transforms = Compose([ToPILImage(), Grayscale(), ToTensor()])
    #     return DataLoader(RacingcarDataset(self.hparams.datadir, train=False, transform=transforms),
    #                        batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--max-epochs', default=1000, type=int)
        parser.add_argument('--learning-rate', default=0.0003, type=float)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--beta', default=1.0, type=float)
        parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
        parser.add_argument('--ndim', type=int, default=256, help='number of latent dimensions')
        parser.add_argument('--early-stop-callback', default=False, action="store_true")
        parser.add_argument('--annealing-epochs', default=10, type=int)
        return parser
