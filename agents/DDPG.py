from agents.utils import *
from agents.ActorCritic import *
from models import *
from torch.distributions import Beta
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
# Deep Deterministic Policy Gradients (DDPG)
# Based on: https://github.com/sfujim/TD3
# Paper: https://arxiv.org/abs/1509.02971

class DDPG(object):
    def __init__(
            self,
            gamma=0.99,
            tau = 0.005,
            ndim=64,
            img_stack=4,
            action_vec=0,
            lr=0.001,
            max_grad_norm=None,
            vae=False,
            infomax=False,
            rnn=False,
            freeze_w=False,
            rl_path="contin_vae/pretrained_vae_64_stack4_conti.ckpt",
            device="cpu",

            writer=DummyWriter(),  # TODO: WRITER HERE?
    ):
        # Parameters necessary for training method
        self.training_step = 0
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.action_vec = action_vec
        self.device = device
        self.test = False
        self.actor_loss = []
        self.critic_loss = []
        # TODO initialize target networks and optimizer!
        # Load models for actor and critic
        v = v_base(ndim=ndim, action_vec=self.action_vec)
        fc = ddpg_base(ndim=ndim, action_vec=self.action_vec)

        # Select ConvNet layer depending on representational learning models
        if vae:  # Variational Autoenconder
            vae = vae_base(vae_path=rl_path,
                           ndim=ndim,
                           img_stack=img_stack,
                           device=self.device,
                           freeze_w=freeze_w)
            self.actor = ActorVAE(vae, fc).float().to(device)
            self.actor_target = ActorVAE(vae, fc).float().to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic = CriticVAE(vae, v).float().to(device)
            self.critic_target = CriticVAE(vae, v).float().to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("DDPG VAE STARTED")
        elif infomax:  # Contrastive Learning
            infomax = infomax_base(infomax_path=rl_path,
                                   ndim=ndim,
                                   device=self.device,
                                   freeze_w=freeze_w)
            self.actor = Actor(infomax, fc).float().to(device)
            self.actor_target = Actor(infomax, fc).float().to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic = Critic(infomax, v).float().to(device)
            self.critic_target = Critic(infomax, v).float().to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("DDPG INFOMAX STARTED")
        elif rnn:
            # TODO: add or remove option for Recurrent Neural Networks
            pass
        else:  # Raw Pixels Learning
            raw_cnn = cnn_base(ndim=ndim, img_stack=img_stack, freeze_w=freeze_w)
            self.actor = Actor(raw_cnn, fc).float().to(device)
            self.actor_target = Actor(raw_cnn, fc).float().to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic = Critic(raw_cnn, v).float().to(device)
            self.critic_target = Critic(raw_cnn, v).float().to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("PPO RAW-PIXEL STARTED")
        # Optimization of actor and critic

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
       # state = state.float().to(self.device)
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def set_test(self, test):
        self.test = test

    def train(self, buffer, batch_size=100):
        # Todo: Use writter for loss metrics
        # Sample replay buffer
        replay_buffer = buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_loss.append(critic_loss)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_loss.append(actor_loss)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



    def save(self, directory = "param/", filename="new_model"):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, directory = "param/", filename="new_model"):
        critic_path = '%s/%s_critic.pth' % (directory, filename)
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, filename), map_location=lambda storage, loc: storage))

        # Critic not necessary for testing
        if os.path.isfile(critic_path):
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=lambda storage, loc: storage))
