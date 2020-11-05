import numpy as np
from torch.nn import functional as F
import torch
from argparse import ArgumentParser
from torch import nn
import pytorch_lightning as pl
# print("here")


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, hidden_dim=256):
        super().__init__()
        feats = int(np.prod(img_shape))
        self.img_shape = img_shape
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, feats)

    # forward method
    def forward(self, z):
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        img = torch.tanh(self.fc4(z))
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, hidden_dim=1024):
        super().__init__()
        in_dim = int(np.prod(img_shape))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


class GAN(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        latent_dim: int = 32,
        learning_rate: float = 0.0005,
        **kwargs
    ):
        """
        Args:
            datamodule: the datamodule (train, val, test splits)
            latent_dim: emb dim for encoder
            batch_size: the batch size
            learning_rate: the learning rate
            data_dir: where to store data
            num_workers: data workers
        """
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()
        self.img_dim = (input_channels, input_height, input_width)

        # networks
        self.generator = self.init_generator(self.img_dim)
        self.discriminator = self.init_discriminator(self.img_dim)

        self.p = 1.0 / 2
        beta = 0.1
        self.swa_start = 10
        self.step_count = 0
        def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
            1 - beta) * averaged_model_parameter + beta * model_parameter
        self.swa_discriminator = torch.optim.swa_utils.AveragedModel(
            self.discriminator, avg_fn=avg_fn)

    def init_generator(self, img_dim):
        generator = Generator(
            latent_dim=self.hparams.latent_dim, img_shape=img_dim)
        return generator

    def init_discriminator(self, img_dim):
        discriminator = Discriminator(img_shape=img_dim)
        return discriminator

    def forward(self, z):
        """
        Generates an image given input noise z

        Example::

            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)
        """
        return self.generator(z)

    def generator_loss(self, x):
        # sample noise
        z = torch.randn(
            x.shape[0], self.hparams.latent_dim, device=self.device)
        y = torch.ones(x.size(0), 1, device=self.device)

        # generate images
        generated_imgs = self(z)

        D_output = self.discriminator(generated_imgs)
        SWA_D_output = self.swa_discriminator(generated_imgs)

        # ground truth result (ie: all real)
        g_loss = F.binary_cross_entropy(
            D_output, y) + self.p * F.binary_cross_entropy(SWA_D_output, y)

        return g_loss

    def param_dist(self):
        dist = 0.
        for p1, p2 in zip(self.discriminator.parameters(), self.swa_discriminator.parameters()):
            dist += torch.norm(p1 - p2, p='fro')
        return self.p * dist

    def discriminator_loss(self, x):
        # train discriminator on real
        b = x.size(0)
        x_real = x.view(b, -1)
        y_real = torch.ones(b, 1, device=self.device)

        # calculate real score
        D_output = self.discriminator(x_real)
        D_real_loss = F.binary_cross_entropy(D_output, y_real)

        # train discriminator on fake
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)
        x_fake = self(z)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate fake score
        D_output = self.discriminator(x_fake)
        D_fake_loss = F.binary_cross_entropy(D_output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss + self.param_dist()

        return D_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)
        self.step_count += 1
        return result

    def generator_step(self, x):
        g_loss = self.generator_loss(x)

        # log to prog bar on each step AND for the full epoch
        # use the generator loss for checkpointing
        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x)

        # log to prog bar on each step AND for the full epoch
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        if self.step_count > self.swa_start:
            self.swa_discriminator.update_parameters(self.discriminator)
            # self.swa_scheduler.step()
        return d_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # self.swa_scheduler = torch.optim.swa_utils.SWALR(opt_d, anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)
        return [opt_g, opt_d], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float,
                            default=0.0002, help="adam: learning rate")
        parser.add_argument('--adam_b1', type=float, default=0.5,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument('--adam_b2', type=float, default=0.999,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument('--latent_dim', type=int, default=100,
                            help="generator embedding dim")
        return parser


def cli_main(args=None):
    from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, MNISTDataModule, STL10DataModule

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str,
                        help="mnist, cifar10, stl10, imagenet")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
    elif script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule

    parser = dm_cls.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GAN.add_model_specific_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    model = GAN(*dm.size(), **vars(args))
    callbacks = [TensorboardGenerativeModelImageSampler(
    ), LatentDimInterpolator(interpolate_epoch_interval=5)]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, progress_bar_refresh_rate=20)
    trainer.fit(model, dm)
    return dm, model, trainer


if __name__ == '__main__':
    dm, model, trainer = cli_main()
