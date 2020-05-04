import torch
from agents.utils import *
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, encoder, fc_base):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.fc_base = fc_base

    def _forward_encoder(self, x):
        x, x_vec = check_tuple(x)
        x = self.encoder.forward(x)
        x = get_torch_cat(x, x_vec)
        return x

    def _forward_nn_base(self, x):
        return self.fc_base.forward(x)

    def forward(self, x):
        x = self._forward_encoder(x)
        return self._forward_nn_base(x)


class Critic(nn.Module):
    def __init__(self, encoder, v_base):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.v_base = v_base

    def _forward_encoder(self, x):
        x, x_vec = check_tuple(x)
        x = self.encoder.forward(x)
        x = get_torch_cat(x, x_vec)
        return x

    def _forward_nn_base(self, x):
        return self.v_base.forward(x)

    def forward(self, x):
        x = self._forward_encoder(x)
        return self._forward_nn_base(x)


class ActorPPO(Actor):
    def __init__(self, encoder, fc_base, alp_bet_head):
        Actor.__init__(self, encoder, fc_base)
        self.alpha_head, self.beta_head = alp_bet_head

    def _forward_nn_base(self, x):
        x = self.fc_base.forward(x)
        a, b = self.alpha_head.forward(x) + 1, self.beta_head.forward(x) + 1
        return a, b


class ActorPPOVAE(ActorPPO):
    def __init__(self, encoder, fc_base, alp_bet_head):
        ActorPPO.__init__(self, encoder, fc_base, alp_bet_head)

    def _forward_encoder(self, x):
        x, x_vec = check_tuple(x)
        mu, logvar = self.encoder.encode(x)  # Reparametrization trick
        x = self.encoder.reparameterize(mu, logvar)
        x = get_torch_cat(x, x_vec)
        return x


class ActorVAE(Actor):
    def __init__(self, encoder, fc_base):
        Actor.__init__(self, encoder, fc_base)

    def _forward_encoder(self, x):
        x, x_vec = check_tuple(x)
        mu, logvar = self.encoder.encode(x)  # Reparametrization trick
        x = self.encoder.reparameterize(mu, logvar)
        x = get_torch_cat(x, x_vec)
        return x


class CriticVAE(Critic):
    def __init__(self, encoder, v_base):
        Critic.__init__(self, encoder, v_base)

    def _forward_encoder(self, x):
        x, x_vec = check_tuple(x)
        mu, logvar = self.encoder.encode(x)  # Reparametrization trick
        x = self.encoder.reparameterize(mu, logvar)
        x = get_torch_cat(x, x_vec)
        return x
