"""
PPO Reinforcing learning with Pytorch on RacingCar v0

Juan Montoya

Version 1.3
TODO
-configure to load vae and infomax

Origina Author:
xtma (Github-user)
code:
https://github.com/xtma/pytorch_car_caring
"""
#python test.py --render --eps 3 --model-path param/ppo_net_params_vae32.pkl ndim 32 --vae True --rl-path pretrain_vae/pretrained_vae_32.ckpt --stop false
#python test.py --render --eps 3 --model-path param/ppo_net_params_imgstack_1.pkl --raw True --stop false
#python test_old.py --render --eps 3 --img-stack 4 --stop true

#python test_old.py --render --eps 3 --img-stack 4 --stop false --model-path param/ppo_net_params_1900.pkl --img-stack 4

import argparse
import numpy as np
from agents.PPO import PPO
import csv
import os
import gym
import torch
from utils import str2bool
from envs import Env
import datetime
from argparse import Namespace



parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
#parser.add_argument('--save-data',action='store_true', help='Save data as zip file')
parser.add_argument('--action-vec', type=int, default=0, metavar='N',
                    help='Action vector for the fully conectected network')
parser.add_argument("--freeze", action='store_true', help='Freeze layers in representational models')
parser.add_argument('--save-csv',action='store_true', help='Save data as csv file')
parser.add_argument('--eps', type=int, default=50, metavar='N', help='Episodes for testing')
parser.add_argument('--model-dict',  type=str, default='params/012', metavar='N', help='Model folder for Representational learning models')
parser.add_argument('--model-name',  type=str, default='po_net_params_imgstack', metavar='N', help='Model name for Representational learning models')
parser.add_argument("--batch-norm", action='store_true', help='Freeze layers in representational models')
parser.add_argument('--rl-path', type=str, default='contin_vae/pretrained_vae_64_stack4_conti_nonexpert_actionregrnonlin_frameskip8.ckpt', metavar='N',
                    help='Give model path for Representational learning models.\n In case of hybrid models used for VAE')
parser.add_argument('--rl-path2', type=str, default='contin_infomax/pretrained_infomax_64_stack4_earlystop_action_conti_nonexpert_frameskip1.ckpt', metavar='N',
                    help='Used just in case of hybrid models for InfoMax path')
parser.add_argument("--vae", type=str2bool, nargs='?', const=True, default=False, help='select vae')
parser.add_argument("--infomax", type=str2bool, nargs='?', const=True, default=False, help='select infomax')
parser.add_argument("--hybrid", type=str2bool, nargs='?', const=True, default=False, help='select hybrid encoder (infomax and vae)')
parser.add_argument("--raw", type=str2bool, nargs='?', const=True, default=True, help='select raw pixel framework')
parser.add_argument('--ndim', type=int, default=64, metavar='G', help='Dimension size shared for both VAE and CNN model')
parser.add_argument("--reward-typ", type=int, default=0,
                    help='Type of reward \n 0 for no reward engineering \n 1 for original reward engineering \n 2 for DonkeyCar reward engineering')
parser.add_argument(
    '--steering', type=float, default=0.2, metavar='N', help='Max steering difference for DonkeyCar reward')

# TODO:
# 1 Load PPO AGENT
# 2 Save logs with csv
# 3 put resume from results
# 4 select type of reward
# 5 Save data option?

#data_dict = {}
args = parser.parse_args()
#print(args)
# CUDA and seed
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Datetime
now_d = datetime.datetime.now().strftime("%d_%m_%Y")
dict_log = "".join(["test/", now_d])
file_log = "".join([args.model_name, "_s", str(args.seed), "_r", str(args.reward_typ),"_ep", str(args.eps)])
CSV_LOG = "".join([dict_log, "/", file_log, ".csv"])

if not os.path.exists(dict_log):
    os.makedirs(dict_log)
if os.path.isfile(CSV_LOG):
    os.remove(CSV_LOG)



if __name__ == "__main__":
    # python test.py --model-dict params/ --model-name vaeSkipNL1000r0_2_best_actor --img-stack 4 --action-repeat 8 --eps 10 --rl-path contin_vae/pretrained_vae_64_stack4_conti_nonexpert_actionregrnonlin_frameskip8.ckpt
    # Parameters:
    # python test_new.py --vae t  --model-dict params/ --model-name vaeSkipNL1000r0_4_best --img-stack 4 --eps 100 --ndim 64 --rl-path contin_vae/pretrained_vae_64_stack4_conti_nonexpert_actionregrnonlin_frameskip8.ckpt
    # python test_new.py --info t  --model-dict params/ --model-name infomaxb1000r0_1_best --img-stack 1 --eps 100 --ndim 64 --rl-path contin_infomax/pretrained_infomax_64_stack4_action_conti.ckpt  --render
    # python test_new.py --model-dict params --model-name raw_b5000_r0_0_best --img-stack 4 --eps 1 --ndim 64  --render
    # python test_new.py --hybrid t  --model-dict params/10_05_2020 --model-name hybridb1000r0_0_best --img-stack 4 --eps 100 --ndim 64
    # python test_new.py --vae t  --model-dict params --model-name vaeFrame6b2000r0_3_best --img-stack 4 --eps 100 --action-repeat 6 --ndim 64 --rl-path contin_vae/pretrained_vae_64_stack4_conti_nonexpert_frameskip8.ckpt
    # python test_new.py --vae t  --model-dict params/imantSkip --model-name vae_b1000_r0_3_best --img-stack 4 --eps 100 --ndim 64 --rl-path contin_vae/pretrained_vae_64_stack4_conti_nonexpert_frameskip8.ckpt
    # python test_new.py --vae t  --model-dict params --model-name vaeNormFS6b1000r0_1_best --img-stack 4 --eps 100 --ndim 64 --rl-path contin_vae/pretrained_vae_64_stack4_conti_nonexpert_frameskip8.ckpt
    # python test_new.py --info t  --model-dict params/18_05_2020 --model-name infoActionFS6b2000r0_1_best --img-stack 1 --eps 100 --ndim 64 --action-repeat 6 --rl-path contin_infomax/pretrained_infomax_64_stack4_action_conti.ckpt  --render
    # python test_new.py --vae t  --model-dict params --model-name vaeSampleAugFS3r0_2_best --img-stack 4 --eps 100 --ndim 64 --rl-path contin_vae/pretrained_vae_64_stack4_conti_nonexpert_frameskip8.ckpt

    agent = PPO(
        ndim=args.ndim,
        action_vec=args.action_vec,
        img_stack=args.img_stack,
        vae=args.vae,
        hybrid=args.hybrid,
        infomax=args.infomax,
        batch_norm= args.batch_norm,
        freeze_w=args.freeze,
        rl_path=args.rl_path,
        rl_path2=args.rl_path2,
        device=device
    )
    agent.load(args.model_dict, args.model_name)
    agent.set_test(True)
    env = Env(seed=args.seed, reward_typ=args.reward_typ, action_repeat=args.action_repeat, img_stack=args.img_stack,
              max_steering_diff=args.steering)
    score_l = []
    for i_ep in range(1, args.eps+1):
        score = 0
        state = env.reset()
        total_reward = 0
        die = 0
        for t in range(1000):
            _, action, _ = agent.select_action(state)
            # TODO: HERE SAVE DATA
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))  # Transform the actions so that in can read left turn
            if args.render:
                env.render()
            score += reward
            state = state_
            if die:
                die = 1
                break
            if done:
                break
        score_l.append(score)
        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        with open(CSV_LOG, "a") as csvfile:
            csv.writer(csvfile).writerow(
                [i_ep, score, die])

    av = np.mean(score_l)
    md = np.median(score_l)
    std = np.std(score_l)
    min = np.min(score_l)
    max = np.max(score_l)
    print(
        'Episodes {}\tMean: {:.2f}\tMedian: {:.2f}\tStandard Dev: {:.2f}\tMin: {:.1f}\tMax: {:.1f}'.format(
            args.eps, av, md, std, min, max))

