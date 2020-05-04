"""
PPO Reinforcing learning with Pytorch on RacingCar v0
Author:
xtma (Github-user)
code:
https://github.com/xtma/pytorch_car_caring
"""
import visdom
import numpy as np
import argparse
import pickle
import gzip
import os
import shutil
import datetime
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


class Writer(ABC):
    log_dir = "runs"

    @abstractmethod
    def add_loss(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_evaluation(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_scalar(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_schedule(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_summary(self, name, mean, std, step="frame"):
        pass


class DummyWriter(Writer):
    def add_loss(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        pass

    def add_scalar(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_summary(self, name, mean, std, step="frame"):
        pass


class ExperimentWriterTb(SummaryWriter, Writer):
    def __init__(self, title, seed, loss=True):
        self.title = "".join([title, "_", str(seed)])
        current_time = datetime.datetime.now().strftime("%d_%m_%Y")
        path =  os.path.join(
                "runs", current_time, self.title
            )

        if os.path.exists(path):
            print("\nREMOVING TENSORBOARD FOLDER!:", str(path), "\n")
            shutil.rmtree(path)

        os.makedirs(
            path
        )

        self.log_dir = path
        self._frames = 0
        self._episodes = 1
        self._loss = loss
        super().__init__(log_dir=self.log_dir)

    def add_loss(self, name, value, step="frame"):
        if self._loss:
            self.add_scalar("loss/" + name, value, step)

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar("evaluation/" + name, value, self._get_step(step))

    def add_schedule(self, name, value, step="frame"):
        if self._loss:
            self.add_scalar("schedule" + "/" + name, value, self._get_step(step))

    def add_scalar(self, name, value, step="frame"):
        super().add_scalar(self.title + "/" + name, value, self._get_step(step))

    def add_summary(self, name, mean, std, step="frame"):
        self.add_evaluation(name + "/mean", mean, step)
        self.add_evaluation(name + "/std", std, step)

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        return _type

    @property
    def frames(self):
        return self._frames

    def set_frames(self, frames):
        self._frames = frames

    @property
    def episodes(self):
        return self._episodes

    def set_episodes(self, episodes):
        self._episodes = episodes


class ExperimentWriterVis():
    now_d = datetime.datetime.now().strftime("%d_%m_%Y")

    def __init__(self, title, seed):
        self._frames = 0
        self._episodes = 1
        self.draw_moving = DrawLine(env="car_mov", title="".join(
            ["MovingAv", "_", self.now_d, "_", title, "_", str(seed)]), xlabel="Frame",
                                    ylabel="Moving averaged frame reward")
        self.draw_mean = DrawLine(env="car_stat",
                                  title="".join(["Mean", "_", self.now_d, "_", title, "_", str(seed)]),
                                  xlabel="Episode",
                                  ylabel="Mean frame reward")
        self.draw_median = DrawLine(env="car_stat", title="".join(
            ["Median", "_", self.now_d, "_", title, "_", str(seed)]), xlabel="Frame",
                                    ylabel="Median frame reward")
        self.draw_loss_agent = DrawLine(env="car_loss", title="".join(
            ["Agent Loss", "_", self.now_d, "_", title, "_", str(seed)]), xlabel="Frame",
                                        ylabel="Loss per Update")
        self.draw_loss_ac = DrawLine(env="car_loss", title="".join(
            ["Actor loss", "_", self.now_d, "_", title, "_", str(seed)]), xlabel="Frame",
                                     ylabel="Actor loss per Update")
        self.draw_loss_cr = DrawLine(env="car_loss", title="".join(
            ["Critic loss", "_", self.now_d, "_", title, "_", str(seed)]), xlabel="Frame",
                                     ylabel="Critic loss per Update")

    def add_moving(self, value, step="frame"):
        self.draw_moving(xdata=self._get_step(step), ydata=value)

    def add_mean(self, value, step="frame"):
        self.draw_mean(xdata=self._get_step(step), ydata=value)

    def add_median(self, value, step="frame"):
        self.draw_median(xdata=self._get_step(step), ydata=value)

    def add_loss_critic(self, value, step="frame"):
        self.draw_loss_cr(xdata=self._get_step(step), ydata=value)

    def add_loss_actor(self, value, step="frame"):
        self.draw_loss_ac(xdata=self._get_step(step), ydata=value)

    def add_loss_agent(self, value, step="frame"):
        self.draw_loss_agent(xdata=self._get_step(step), ydata=value)

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        return _type

    @property
    def frames(self):
        return self._frames

    def set_frames(self, frames):
        self._frames = frames

    @property
    def episodes(self):
        return self._episodes

    def set_episodes(self, episodes):
        self._episodes = episodes


def store_data(data, name="data", datasets_dir="./data", ):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, "".join([name, '.pkl.gzip']))
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class DrawLine():
    def __init__(self, env, title, xlabel=None, ylabel=None):
        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )
