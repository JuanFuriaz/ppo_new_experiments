import torch
from argparse import Namespace
from models import *
from agents.ActorCritic import *
from agents.PPO import PPO
from agents.DDPG import DDPG
import numpy as np

from contin_vae.models import VAE
from contin_infomax.models import InfoMax
import torch.optim as optim

os.chdir('/home/jm/Documents/research/self-driving-car/CarRacing/ppo_car_racing')
args = {'gamma': 0.99,
        'ndim': 32,
        'action_repeat': 8,
        'action_vec': 0,
        'eps': 4000,
        'terminate': False,
        'img_stack': 4,
        'seed': 0,
        'render': False,
        'vis': False,
        'tb': False,
        'log_interval': 10,
        'buffer': 10,
        'batch': 128,
        'learning': 0.001,
        'vae': False,
        'infomax': False,
        'raw': True,
        'rnn': False,
        'reward_mod': True,
        'freeze': False,
        'rl_path': 'pretrain_vae/pretrained_vae_32_stack4.ckpt',
        'title': 'debug',
        'debug': True}
args = Namespace(**args)
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# TODO: WORK WITH CUDA
# TODO: More asserts less printing
device = "cpu"


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    From Train.py v1.4
    """

    def __init__(self):
        super(Net, self).__init__()

        """
        if args.rnn:
            self.h = torch.zeros(1, 100)
        """
        if args.vae:  # representational learning
            # load vae model
            self.vae = self.load_rl(args)

        elif args.infomax:
            # load infomax
            self.infomax = self.load_rl(args)
        else:
            self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
                nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
                nn.ReLU(),  # activation
                nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
                nn.ReLU(),  # activation
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
                nn.ReLU(),  # activation
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
                nn.ReLU(),  # activation
                nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
                nn.ReLU(),  # activation
                nn.Conv2d(128, args.ndim, kernel_size=3, stride=1),  # (128, 3, 3)
                nn.ReLU(),  # activation
            )  # output shape (256, 1, 1)
            print("Raw pixel loaded")
            self.apply(self._weights_init)
            if args.freeze:
                print("Freezing ConvLayer Weights")
                for i, chld in enumerate(self.cnn_base.children()):  # Freeze weights
                    for params in chld.parameters():
                        params.requires_grad = False

        if (args.vae or args.infomax) and args.action_vec > 0:
            self.v = nn.Sequential(nn.Linear(args.ndim + args.action_vec * 3, 100), nn.ReLU(), nn.Linear(100, 1))
            self.fc = nn.Sequential(nn.Linear(args.ndim + args.action_vec * 3, 100), nn.ReLU())

        else:
            if args.rnn:
                self.gru = nn.GRUCell(args.ndim, 100)
                self.v = nn.Linear(100, 1)
            else:
                self.v = nn.Sequential(nn.Linear(args.ndim, 100), nn.ReLU(), nn.Linear(100, 1))
                self.fc = nn.Sequential(nn.Linear(args.ndim, 100), nn.ReLU())
                self.fc.forward()

        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())

    @staticmethod
    def _weights_init(m):
        """
         Weights initialization
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Gives two function results
        1) Beta and Gamma for computing the distribution of the policy (using beta distribution)
        2) Value function for advantage term
        """
        if args.vae:  # representational learning
            # load vae model
            if args.action_vec > 0:
                x = torch.cat((self.get_z(x[0]), x[1]), dim=1)
            else:
                x = self.get_z(x)
        elif args.infomax:
            if args.action_vec > 0:
                x = torch.cat((self.infomax.encoder(x[0]), x[1]), dim=1)
            else:
                x = self.infomax.encoder(x)
        else:
            # TODO: Conv with action vector?
            x = self.cnn_base(x)
            x = x.view(-1, args.ndim)
        if args.rnn:
            # h = self.gru(x, self.h)
            h = self.gru(x)
            # self.h = h.detach()
            x = h
            # print(h.shape)
            v = self.v(x)
        else:
            v = self.v(x)
            x = self.fc(x)

        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

    @staticmethod
    def load_rl(args_parser):
        hparams = Namespace(**torch.load(args_parser.rl_path)["hparams"])
        state_dict = torch.load(args.rl_path)["state_dict"]
        assert hparams.ndim == args_parser.ndim
        # assert hparams.img_stack == args_parser.img_stack
        if args.vae:
            rl = VAE(hparams).to(device)  # Load VAE with parameters
            print("VAE Loaded")
        else:
            rl = InfoMax(hparams).to(device)  # Load VAE with parameters
            print("InfoMax loaded")

        rl.load_state_dict(state_dict)  # Load weights
        if args.freeze:
            print("Freezing Representational Learning Model")
            for i, chld in enumerate(rl.children()):  # Freeze weights
                for params in chld.parameters():
                    params.requires_grad = False
        return rl

    def get_z(self, x):
        mu, logvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, logvar).to(device)


def test_select_action_ppo():
    for i in range(3):
        s = np.random.rand(4, 96, 96)
        if i == 0:
            agent = PPO()
            print("Assert for Raw Pixel")
        elif i == 1:
            agent = PPO(vae=True)
            print("Assert for VAE")
        elif i == 2:
            s = np.random.rand(1, 96, 96)
            agent = PPO(infomax=True, img_stack=1, rl_path="contin_infomax/pretrained_infomax_64_stack4_action_conti.ckpt")
            print("Assert for Infomax")
        print(agent.select_action(s))


def test_init_ddpg():
    for i in range(3):
        s = np.random.rand(4, 96, 96)
        if i == 0:
            agent = DDPG()
            print("Init asserted Raw Pixel DDPG")
        elif i == 1:
            agent = DDPG(vae=True)
            print("Init Asserted for VAE DDPG")
        elif i == 2:
            agent = DDPG(infomax=True, rl_path="contin_infomax/pretrained_infomax_64_stack4_action_conti.ckpt")
            print("Init Asserted for Infomax DDPG")


def test_select_action_ddpg():
    for i in range(3):
        s = np.random.rand(4, 96, 96)
        if i == 0:
            agent = DDPG()
            print("Init asserted Raw Pixel DDPG")
        elif i == 1:
            agent = DDPG(vae=True)
            print("Init Asserted for VAE DDPG")
        elif i == 2:
            s = np.random.rand(1, 96, 96)
            agent = DDPG(infomax=True, rl_path="contin_infomax/pretrained_infomax_64_stack1_action_conti.ckpt")
            print("Init Asserted for Infomax DDPG")
        #TODO: WHY RESULTS STRANGE WITH INFOMAX
        print(agent.select_action(s))


def main():
    pass


if __name__ == '__main__':
    test_select_action_ppo()
   # test_init_ddpg()
    test_select_action_ddpg()