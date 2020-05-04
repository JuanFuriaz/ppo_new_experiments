import torch
from argparse import Namespace
from models import *
from agents.ActorCritic import *
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


def test_forward():
    # Testing forward with new structure
    x1 = torch.tensor([0, 2])
    x2 = torch.tensor([3, 2])
    x = (x1, x2)
    if isinstance(x, tuple):
        x, x_vec = x[0], x[1]
    if 'x_vec' in locals():
        x = torch.cat((x, x_vec), dim=0)
    return x


def print_weights(model, device = "cpu"):
    print("Weights Statistics ")
    for k, v in model.state_dict().items():
        if "weight" in k:
            print("Layer {}".format(k),
                  ' Sum {:.4f}\tMean: {:.4f}\tMedian: {:.4f}\tStd: {:.4f}\tMax: {:.2f}\tMin: {:.2f}'.format(
                      v.sum().to(device).numpy().tolist(), v.mean().to(device).numpy().tolist(),
                      v.median().to(device).numpy().tolist(), v.std().to(device).numpy().tolist(),
                      v.max().to(device).numpy().tolist(), v.min().to(device).numpy().tolist()))


def test_weights_two_models(m1, m2, same=True):
    print("Weights Statistics ")
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        print("Asserting layer 1 {}\t Layer 2 {}".format(k1, k2))
        if same:
            assert (v1.sum().to(device).to(device).numpy().tolist(), v1.mean().to(device).numpy().tolist(),
                    v1.median().to(device).numpy().tolist()) == (
                       v2.sum().to(device).numpy().tolist(), v2.mean().to(device).numpy().tolist(),
                       v2.median().to(device).numpy().tolist())
            assert (v1.std().to(device).numpy().tolist(), v1.max().to(device).numpy().tolist(),
                    v1.min().to(device).numpy().tolist()) == (
                       v1.std().to(device).numpy().tolist(), v1.max().to(device).numpy().tolist(),
                       v1.min().to(device).numpy().tolist())
            print("ok ")
        else:  # Here checking just sum
            assert ((v1.sum().to(device).numpy().tolist()) != (v2.sum().to(device).numpy().tolist()))
            print("ok ")


def test_print_weights(model=cnn_base(64, 4)):
    print_weights(model)


def test_forward_actorPPO():
    cnn = cnn_base(64, 1)
    fc = fc_base(64)
    alp_bet = alpha_beta_head()
    actor = ActorPPO(cnn, fc, alp_bet).float().to(device)
    s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
    assert (actor.forward(s).shape[1] == 3)

def test_forward_actorDDPG():
    cnn = cnn_base(64, 1)
    fc = ddpg_base(64)
    actor = Actor(cnn, fc).float().to(device)
    s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
    assert (actor.forward(s).shape[1] == 3)
    #
    # print(actor.forward(s))


def test_forward_critic():
    cnn = cnn_base(64, 1)
    v = v_base(64)
    critic = Critic(cnn, v).float().to(device)
    s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
    assert critic.forward(s)


def test_critic_actorPP0():
    cnn = cnn_base(64, 1)
    v = v_base(64)
    fc = fc_base(64)
    alp_bet = alpha_beta_head()
    actor = ActorPPO(cnn, fc, alp_bet).float().to(device)
    critic = Critic(cnn, v).float().to(device)
    test_weights_two_models(actor.encoder, critic.encoder)
    # print_weights(critic.encoder)


def test_optim_weights(same_weights=True):
    v = v_base(64).float()
    fc = fc_base(64)
    alp_bet = alpha_beta_head()
    cnn = cnn_base(64, 1)
    if same_weights:
        actor = ActorPPO(cnn, fc, alp_bet).float().to(device)
        critic = Critic(cnn, v).float().to(device)
    else:
        cnn2 = cnn_base(64, 1)
        actor = ActorPPO(cnn, fc, alp_bet).float().to(device)
        critic = Critic(cnn2, v).float().to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=0.001)
    opt_critic = optim.Adam(critic.parameters(), lr=0.001)
    for i in range(1001):
        s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
        out_c = critic.forward(s)
        out_a = actor.forward(s)
        loss_fn = nn.MSELoss()
        loss_critic = loss_fn(out_c, torch.tensor(-1).float().to(device))
        loss_actor = loss_fn(torch.cat((out_a[0], out_a[1]), 1),
                             torch.tensor([-1, -1, -1, -1, -1, -1]).float().to(device))

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        opt_critic.step()
        opt_actor.step()

        if i % 200 == 0 and i != 0:
            test_weights_two_models(actor.encoder, critic.encoder, same_weights)
            # print("ACTOR")
        # print_weights(actor.encoder)
        # print("CRITIC")
        # print_weights(critic.encoder)


def test_freeze_cnn_weights(freeze_w=True, range_n=201):
    v = v_base(64)
    fc = fc_base(64)
    alp_bet = alpha_beta_head()
    cnn = cnn_base(64, 1, freeze_w=freeze_w)
    actor = ActorPPO(cnn, fc, alp_bet).float().to(device)
    critic = Critic(cnn, v).float().to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=0.001)
    opt_critic = optim.Adam(critic.parameters(), lr=0.001)
    for i in range(range_n):
        s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
        out_c = critic.forward(s)
        out_a = actor.forward(s)
        loss_fn = nn.MSELoss()
        loss_critic = loss_fn(out_c, torch.tensor(-1).float().to(device))
        loss_actor = loss_fn(torch.cat((out_a[0], out_a[1]), 1),
                             torch.tensor([-1, -1, -1, -1, -1, -1]).float().to(device))

        if i == 0:
            print("INIT ACTOR")
            print_weights(actor.encoder)
            print("INIT CRITIC")
            print_weights(critic.encoder)

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        opt_critic.step()
        opt_actor.step()

        if i % 200 == 0 and i != 0:
            print("ACTOR")
            print_weights(actor.encoder)
            print("CRITIC")
            print_weights(critic.encoder)


def test_load_rl():
    vae_base(vae_path="contin_vae/pretrained_vae_64_stack4_conti.ckpt", freeze_w=True, device=device)
    vae_base(vae_path="contin_vae/pretrained_vae_64_stack4_conti.ckpt", freeze_w=False, device=device)
    infomax_base(freeze_w=True)
    infomax_base(freeze_w=False)


def test_load_foward_actor_rl():
    vae = vae_base(vae_path="contin_vae/pretrained_vae_64_stack1_conti.ckpt", ndim=64, img_stack=1, device=device)
    info = infomax_base(device=device)
    fc = fc_base(64)
    alp_bet = alpha_beta_head()
    actor_info = ActorPPO(info, fc, alp_bet).float().to(device)
    actor_vae = ActorPPOVAE(vae, fc, alp_bet).float().to(device)
    s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
    assert actor_info.forward(s)
    assert actor_vae.forward(s)


def test_optim_actor_rl(freeze_w=False, print_vae=True, range_n=201):
    vae = vae_base(vae_path="contin_vae/pretrained_vae_64_stack1_conti.ckpt", ndim=64, img_stack=1, device=device,
                   freeze_w=freeze_w)
    info = infomax_base(device=device, freeze_w=freeze_w)
    fc = fc_base(64)
    alp_bet = alpha_beta_head()
    actor_info = ActorPPO(info, fc, alp_bet).float().to(device)
    actor_vae = ActorPPOVAE(vae, fc, alp_bet).float().to(device)
    opt_actor_info = optim.Adam(actor_info.parameters(), lr=0.0001)
    opt_actor_vae = optim.Adam(actor_vae.parameters(), lr=0.0001)
    for i in range(range_n):
        s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
        out_vae = actor_vae.forward(s)
        out_info = actor_info.forward(s)
        loss_fn = nn.MSELoss()
        loss_actor_vae = loss_fn(torch.cat((out_vae[0], out_vae[1]), 1),
                                 torch.tensor([-1, -1, -1, -1, -1, -1]).float().to(device))
        loss_actor_info = loss_fn(torch.cat((out_info[0], out_info[1]), 1),
                                  torch.tensor([-1, -1, -1, -1, -1, -1]).float().to(device))

        if i == 0:
            if not print_vae:
                print(" ")
                print("Initial ACTOR INFOMAX")
                print_weights(actor_info)
            else:
                print(" ")
                print("Initial ACTOR VAE")
                print_weights(actor_vae)

        opt_actor_info.zero_grad()
        opt_actor_vae.zero_grad()
        loss_actor_info.backward()
        loss_actor_vae.backward()
        opt_actor_vae.step()
        opt_actor_info.step()

        if i % 200 == 0 and i != 0:
            if not print_vae:
                print(" ")
                print("ACTOR INFOMAX")
                print_weights(actor_info)
            else:
                print(" ")
                print("ACTOR VAE")
                print_weights(actor_vae)


def test_optim_critic_rl(freeze_w=False, print_vae=True, range_n=201):
    vae = vae_base(vae_path="contin_vae/pretrained_vae_64_stack1_conti.ckpt", ndim=64, img_stack=1, device=device,
                   freeze_w=freeze_w)
    info = infomax_base(device=device, freeze_w=freeze_w)
    v = v_base(64)
    critic_info = Critic(info, v).float().to(device)
    critic_vae = CriticVAE(vae, v).float().to(device)
    opt_critic_info = optim.Adam(critic_info.parameters(), lr=0.0001)
    opt_critic_vae = optim.Adam(critic_vae.parameters(), lr=0.0001)
    for i in range(range_n):
        s = torch.tensor(np.random.rand(1, 1, 96, 96)).float().to(device)
        out_vae = critic_vae.forward(s)
        out_info = critic_info.forward(s)
        loss_fn = nn.MSELoss()
        loss_critic_vae = loss_fn(out_vae,
                                  torch.tensor(-1).float().to(device))
        loss_critic_info = loss_fn(out_info,
                                   torch.tensor(-1).float().to(device))

        if i == 0:
            if not print_vae:
                print(" ")
                print("Initial critic INFOMAX")
                print_weights(critic_info)
            else:
                print(" ")
                print("Initial critic VAE")
                print_weights(critic_vae)

        if i % 200 == 0 and i != 0:
            if not print_vae:
                print(" ")
                print("Critic INFOMAX")
                print_weights(critic_info)
            else:
                print(" ")
                print("Critic VAE")
                print_weights(critic_vae)

        opt_critic_info.zero_grad()
        opt_critic_vae.zero_grad()
        loss_critic_info.backward()
        loss_critic_vae.backward()
        opt_critic_vae.step()
        opt_critic_info.step()


def main():
    #test_forward_critic()
    #test_forward_actorPPO()
    test_forward_actorDDPG()
    #test_critic_actorPP0()
    # test_optim_weights()
    # test_optim_weights(False)
    # test_freeze_cnn_weights(True)
    # test_freeze_cnn_weights(False)
    # RL for Representation Learning
    # test_load_rl()
    #test_load_foward_actor_rl()
    # test_optim_actor_rl(False, True) # weights not freeze print vae
    # test_optim_actor_rl(False, False)  # weights not freeze print vae
    # test_optim_actor_rl(True, True) # weights freeze print vae
    # test_optim_actor_rl(True, False) # weights freeze print info
    # test_optim_critic_rl(False, True) # weights not freeze print vae
    # test_optim_critic_rl(False, False) # weights not freeze print info
    # test_optim_critic_rl(True, True) # weights freeze print vae
    # test_optim_critic_rl(True, False)  # weights freeze print info


if __name__ == '__main__':
    main()
