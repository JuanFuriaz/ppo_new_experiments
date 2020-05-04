from agents.utils import *
from agents.ActorCritic import *
from models import *
from torch.distributions import Beta
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F


class PPO(object):
    def __init__(
            self,
            gamma=0.99,
            entropy_loss_scaling=0.0001,
            value_loss_scaling=2.0,
            clip_param=0.1,
            ppo_epoch=8,
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
        self.entropy_l_sc = entropy_loss_scaling
        self.value_l_sc = value_loss_scaling
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.max_grad_norm = max_grad_norm
        self.action_vec = action_vec
        self.device = device
        self.test = False

        # Load models for actor and critic
        v = v_base(ndim=ndim, action_vec=self.action_vec)
        fc = fc_base(ndim=ndim, action_vec=self.action_vec)
        alp_bet_h = alpha_beta_head()

        # Select ConvNet layer depending on representational learning models
        if infomax:  # Contrastive Learning
            vae = vae_base(vae_path="contin_vae/pretrained_vae_64_stack1_conti.ckpt",
                           ndim=ndim,
                           img_stack=img_stack,
                           device=self.device,
                           freeze_w=freeze_w)
            infomax = infomax_base(infomax_path="contin_infomax/pretrained_infomax_64_stack4_earlystop_action_conti.ckpt",
                                   ndim=ndim,
                                   device=self.device,
                                   freeze_w=freeze_w)
            self.actor = ActorPPO(infomax, fc, alp_bet_h).float().to(device)
            self.critic = Critic(infomax, v).float().to(device)
            self.critic_target = CriticVAE(vae, v).float().to(device)
            print("\nPPO INFOMAX WITH VAE STARTED\n")

        # Join optimization of actor and critic
        params_opt = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.critic_target.parameters())
        self.optimizer = optim.Adam(params_opt, lr=lr)

    def select_action(self, state):
        # Adaptation in case of using action vector
        if self.action_vec > 0:
            state = (torch.from_numpy(state[0]).float().to(self.device).unsqueeze(0),
                     torch.from_numpy(state[1]).float().to(self.device).unsqueeze(0))
        else:
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            alpha, beta = self.actor(state)

        if not self.test:  # Sampling action using beta distribution
            dist = Beta(alpha, beta)
            action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1)
            a_logp = a_logp.item()
        else:  # Take action without exploration for testing
            action = alpha / (alpha + beta)
            a_logp = None

        action = action.squeeze().cpu().numpy()
        return action, a_logp

    def set_test(self, test):
        self.test = test

    def train(self, buffer, batch_size=100):
        self.training_step += 1
        s = torch.tensor(buffer.buffer['s'], dtype=torch.float).to(self.device)
        a = torch.tensor(buffer.buffer['a'], dtype=torch.float).to(self.device)
        r = torch.tensor(buffer.buffer['r'], dtype=torch.float).to(self.device).view(-1, 1)
        s_ = torch.tensor(buffer.buffer['s_'], dtype=torch.float).to(self.device)
        old_a_logp = torch.tensor(buffer.buffer['a_logp'], dtype=torch.float).to(self.device).view(-1, 1)

        if self.action_vec > 0: a_v = torch.tensor(buffer.buffer['a_v'], dtype=torch.float).to(self.device)

        with torch.no_grad(): # Compute a vector with advantage terms
            # TODO ACTION VEC FOR 1 DONT NEED a_v, you can take just action
            if self.action_vec > 0:
                target_v = r + self.gamma * self.critic_target((s_, a_v[:, 3:]))
                adv = target_v - self.critic((s, a_v[:, :-3]))
            else:
                target_v = r + self.gamma * self.critic_target(s_)
                adv = target_v - self.critic(s)

        for _ in range(self.ppo_epoch):
            # Compute update for mini batch
            for index in BatchSampler(SubsetRandomSampler(range(buffer.buffer_capacity)), batch_size, False):
                if self.action_vec > 0:
                    alpha, beta = self.actor((s[index], a_v[index, :-3]))
                else:
                    alpha, beta = self.actor(s[index])
                dist = Beta(alpha, beta)
                entropy = dist.entropy().mean()
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])    # old/new_policy for Trust Region Method

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]  # Clip Ratio
                action_loss = -torch.min(surr1, surr2).mean()
                # Difference between prediction and real values
                if self.action_vec > 0:
                    value_loss = F.smooth_l1_loss(self.critic((s[index], a_v[index, :-3])), target_v[index])
                else:
                    value_loss = F.smooth_l1_loss(self.critic(s[index]), target_v[index])

                # Loss with Entropy
                loss = action_loss + 2. * value_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_( list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)

                self.optimizer.step()

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
