"""
PPO Reinforcing learning with Pytorch on RacingCar v0
Juan Montoya

2.1
-Use frames instead of episodes


Orignal Author:
xtma (Github-user)
code:
https://github.com/xtma/pytorch_car_caring
"""

import argparse
from agents.PPO import PPO
from agents.utils import BufferPPO
from envs import Env
import torch
from argparse import Namespace
from utils import *
import datetime
import csv
import os

# TODO add extra values for PPO
# TODO charts in case of multiple reward analyse
parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
subparsers = parser.add_subparsers(description='Representational Learning Models ')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--value-sc', type=float, default=2.0, metavar='G', help='Value scale loss for PPO loss function')
parser.add_argument('--ndim', type=int, default=256, metavar='G',
                    help='Dimension size shared for both VAE and CNN model')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--action-vec', type=int, default=0, metavar='N',
                    help='Action vector for the fully conectected network')
parser.add_argument('--frames', type=int, default=4000000, metavar='N', help='Episodes for testing')
parser.add_argument('--eps', type=int, default=10, metavar='N', help='Episodes for testing')
parser.add_argument('--ppo-epoch', type=int, default=10, metavar='N', help='Epochs for PPO update')
parser.add_argument('--terminate', action='store_true', help='Termination after the predefined threeshold')
parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--tb', action='store_true', help='Tensorboard')
parser.add_argument(
    '--log-interval', type=int, default=10000, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument(
    '--buffer', type=int, default=5000, metavar='N', help='Size of Buffer with Expierence Tupels ')
parser.add_argument(
    '--batch', type=int, default=128, metavar='N', help='Batch size for sampling')
parser.add_argument(
    '--lr', type=float, default=0.001, metavar='N', help='Learning Rate')
parser.add_argument(
    '--lr-min', type=float, default=0, metavar='N',
    help='Min value for Learning Rate using consine Annealing sheduler or Linear sheduler')
parser.add_argument(
    '--grad-norm', type=float, default=None, metavar='N', help='Maximum gradient norm for clipping gradients')
parser.add_argument("--vae", type=str2bool, nargs='?', const=True, default=False, help='select vae')
parser.add_argument("--infomax", type=str2bool, nargs='?', const=True, default=False, help='select infomax')
parser.add_argument("--raw", type=str2bool, nargs='?', const=True, default=True, help='select raw pixel framework')
parser.add_argument('--rnn', action='store_true', help='Use gated recurrend unit')
parser.add_argument("--reward-typ", type=int, default=1,
                    help='Type of reward \n 0 for no reward engineering \n 1 for original reward engineering \n 2 for DonkeyCar reward engineering')
parser.add_argument(
    '--steering', type=float, default=0.001, metavar='N', help='Max steering difference for DonkeyCar reward')
parser.add_argument("--freeze", action='store_true', help='Freeze layers in representational models')
parser.add_argument('--rl-path', type=str, default='contin_vae/pretrained_vae_16.ckpt', metavar='N',
                    help='Give model path for Representational learning models')
parser.add_argument('--title', type=str, default='', metavar='N', help='Name for the image title')
parser.add_argument('--debug', action='store_true', help='Debug on')

args = parser.parse_args()
# print(args)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.debug:
    args = {'gamma': 0.99,
            'value_sc': 2.0,
            'ndim': 32,
            'action_repeat': 8,
            'action_vec': 0,
            'frames': 4000000,
            'eps': 1,
            'ppo_epoch': 10,
            'terminate': False,
            'img_stack': 1,
            'seed': 0,
            'render': False,
            'vis': True,
            'tb': False,
            'log_interval': 100,
            'buffer': 100,
            'batch': 128,
            'lr': 0.001,
            'lr_min': 0.0,
            'grad_norm': None,
            'vae': False,
            'infomax': False,
            'raw': True,
            'rnn': False,
            'reward_typ': 1,
            'freeze': False,
            'rl_path': 'contin_infomax/pretrained_infomax_32_stack4_earlystop_action_conti.ckpt',
            'title': 'debug',
            'debug': True}
    args = Namespace(**args)
    print("DEBUGGING MODUS")

print("Parameters: ")
print(args)
print("")

run_dict = "".join(["runs/", datetime.datetime.now().strftime("%d_%m_%Y")])
PARAMS_DICT = "".join(["params/", datetime.datetime.now().strftime("%d_%m_%Y")])
TITLE = "".join([args.title, "_", str(args.seed)])
BEST_SCORE = 0
csv_log_rew0 = "".join([run_dict, "/", TITLE, "_", "rew0", ".csv"])
csv_log_rew1 = "".join([run_dict, "/", TITLE, "_", "rew1", ".csv"])
csv_log_rew2 = "".join([run_dict, "/", TITLE, "_", "rew2", ".csv"])

# Load Tensorboard writer and checkout for csv files
if args.tb:
    writer_tb = ExperimentWriterTb(args.title, args.seed)
else:
    writer_tb = None
    if not os.path.exists(run_dict):
        os.makedirs(run_dict)

# Load vis writer
if args.vis:
    writer_vs = ExperimentWriterVis(args.title, args.seed)
else:
    writer_vs = None

# Check if csv file already exists if yes erase it
if os.path.isfile(csv_log_rew0):
    os.remove(csv_log_rew0)
if os.path.isfile(csv_log_rew1):
    os.remove(csv_log_rew1)
if os.path.isfile(csv_log_rew2):
    os.remove(csv_log_rew2)
if args.reward_typ == 1:
    CSV_LOG = csv_log_rew1
elif args.reward_typ == 2:
    CSV_LOG = csv_log_rew2
else:
    CSV_LOG = csv_log_rew0

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


def test(agent, env, eps_n, frames, running_score, train_step, reward_typ, vis_writer, tf_writer, csv_log):
    global BEST_SCORE
    # Save count values of training before testing
    eps_train = env.eps
    reward_train = env.reward_typ
    # Configuration for testing
    agent.set_test(True)
    env.set_reward(reward_typ)
    score_l = []
    loss_agent = torch.tensor(agent.loss_agent).mean().cpu().numpy()
    loss_ac = torch.tensor(agent.loss_actor).mean().cpu().numpy()
    loss_cr = torch.tensor(agent.loss_critic).mean().cpu().numpy()

    print(" \nTESTING: ")

    if agent.lr_min > 0:
        lr = agent.scheduler.get_lr()[0]
    else:
        lr = agent.lr

    for i_ep in range(1, eps_n + 1):
        score = 0
        state = env.reset()
        for t in range(1000):
            # TODO: UPDATE FOR ACTION VECTOR
            action, _ = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array(
                [-1., 0., 0.]))  # Transform the actions so that in can read left turn
            score += reward
            state = state_

            if die or done:
                break
        score_l.append(score)
        # print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        with open(csv_log, "a") as csvfile:
            csv.writer(csvfile).writerow(
                [i_ep, frames, agent.training_step, score, running_score, lr, loss_ac, loss_cr, loss_agent])

    av = np.mean(score_l)
    md = np.median(score_l)
    std = np.std(score_l)
    min = np.min(score_l)
    max = np.max(score_l)

    # Saving
    if train_step < agent.training_step:
        agent.save(PARAMS_DICT, TITLE)
        if BEST_SCORE < av:
            agent.save(PARAMS_DICT, TITLE + "_best")
            BEST_SCORE = av

    print(
        'Frame {}\tMoving av.: {:.1f}\tMean: {:.2f}\tMedian: {:.2f}\tStandard Dev: {:.2f}\tMin: {:.1f}\tMax: {:.1f}'.format(
            frames, running_score, av, md, std, min, max))

    print('Steps Update {}\tLoss PPO: {:.2f}\tLoss Actor: {:.2f}\tLoss Critic: {:.2f}\tLr: {:8.5f}'.format(
        agent.training_step, loss_agent, loss_cr, loss_ac, lr))
    print("")
    if vis_writer:
        vis_writer.set_frames(frames)
        vis_writer.add_mean(av)
        vis_writer.add_median(md)
        vis_writer.add_loss_critic(loss_cr)
        vis_writer.add_loss_actor(loss_ac)
        vis_writer.add_loss_agent(loss_agent)
        vis_writer.add_moving(running_score)

    if tf_writer:
        tf_writer.set_frames(frames)
        tf_writer.add_loss("loss_critic", loss_cr)
        tf_writer.add_loss("loss_actor", loss_ac)
        tf_writer.add_loss("loss_agent", loss_agent)
        tf_writer.add_evaluation("moving", running_score)
        tf_writer.add_evaluation("median", md)
        tf_writer.add_summary("mean_std", av, std)

    # Set train values back
    agent.set_test(False)
    env.set_reward(reward_train)
    env.set_eps(eps_train)


def train():
    # Calculate steps for sheduling
    global CSV_LOG
    steps_opt = round(args.buffer / args.batch) * args.ppo_epoch * round(
        args.frames / (args.buffer * args.action_repeat))

    agent = PPO(
        gamma=args.gamma,
        ndim=args.ndim,
        action_vec=args.action_vec,
        img_stack=args.img_stack,
        lr=args.lr,
        lr_min=args.lr_min,
        value_loss_scaling=args.value_sc,
        steps_opt=steps_opt,
        max_grad_norm=args.grad_norm,
        vae=args.vae,
        infomax=args.infomax,
        rnn=args.rnn,
        freeze_w=args.freeze,
        rl_path=args.rl_path,
        device=device
    )
    buffer = BufferPPO(
        img_stack=args.img_stack,
        action_vec=args.action_vec,
        buffer_capacity=args.buffer
    )

    env = Env(seed=args.seed, reward_typ=args.reward_typ, action_repeat=args.action_repeat, img_stack=args.img_stack,
              max_steering_diff=args.steering)
    total_frames = 0
    total_eps = 0
    score_l = []
    running_score = 0
    frames_log = args.log_interval
    train_step = 0

    while total_frames < args.frames:
        score = 0
        state = env.reset()
        # TODO: RNN RESET
        # if args.rnn: agent.reset()
        if args.action_vec > 0: act_vec = [np.zeros(3)] * (args.action_vec + 1)  # initialize stack for past actions
        for t in range(1000):
            if args.action_vec > 0:
                action, a_logp = agent.select_action((state, np.array(act_vec).flatten()[:-3]))
                act_vec.pop(0)  # remove oldest action and add new one
                act_vec.append(action)
            else:
                action, a_logp = agent.select_action(state)

            # TODO: ACTION TRANSFORMED JUST FOR PPO?
            action_step = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])  # reparametrize steering values
            state_, reward, done, die = env.step(action_step)
            env.set_last_action(action_step)

            if args.render:
                env.render()
            if args.action_vec > 0:
                if buffer.store((state, action, a_logp, reward, state_, np.array(act_vec).flatten())):
                    print('updating with action vector')
                    agent.train(buffer, args.batch)
                    train_step = agent.training_step
            else:
                if buffer.store((state, action, a_logp, reward, state_)):
                    print('updating')
                    agent.train(buffer, args.batch)

            score += reward

            state = state_
            if done or die:
                total_frames += env.steps
                total_eps += 1
                break

        score_l.append(score)
        running_score = running_score * 0.99 + score * 0.01
        if frames_log <= total_frames:
            frames_log += args.log_interval
            test(agent, env, args.eps, total_frames, running_score, train_step, args.reward_typ, writer_vs, writer_tb, CSV_LOG)
            # test(agent, env, args.eps, total_frames, running_score, 0, None, None, csv_log_rew0)
            score_l = []


if __name__ == "__main__":
    # TODO: INIT FOR OTHER AGENTS HERE
    train()
