# v1.1
# Add donkeycar reward
import gym
import numpy as np
from torchvision.transforms import Compose, ToTensor, ToPILImage, Grayscale


class Env:
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, seed=0, reward_typ=1, action_repeat=8, img_stack=4, max_steering_diff=0.2):
        """
        Create Env Racing  Car
        """
        self.env = gym.make('CarRacing-v0')
        self.env.seed(seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.transform = Compose([ToPILImage(), Grayscale(), ToTensor()])
        self.reward_typ = reward_typ
        self.eps = 0
        self.action_repeat = action_repeat
        self.img_stack = img_stack
        self.max_steering_diff = max_steering_diff

    def reset(self):
        """
        Restart values
        """
        self.steps = 0
        self.eps += 1
        self.av_r = self.reward_memory()
        self.last_action = [0, 0, 0]

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision

        return np.array(self.stack)

    def step(self, action):
        """
        Steps giving rewards and (in case) terminate
        """
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if self.reward_typ == 1:
                if die:
                    reward += 100
                # green penalty
                if np.mean(img_rgb[:, :, 1]) > 185.0:
                    reward -= 0.05
                total_reward += reward
                # if no reward recently, end the episode
                done = True if self.av_r(reward) <= -0.1 else False
            elif self.reward_typ == 2:
                throttle_reward = (action[1] / 1) / self.action_repeat
                break_penalty = -1 * (action[2] / 1) / (self.action_repeat * 2)
                green_penalty = 0
                jerk_penalty = 0

                if np.mean(img_rgb[:, :, 1]) > 180.0:  # Restart and punish to go to green part
                    done = True
                    green_penalty = -100
                    print("DROVE OFF")
                else:
                    done = False

                # Jerk_penalty
                steering_diff = (self.last_action[0] - action[0]) / 2

                if abs(steering_diff) > self.max_steering_diff:
                    error = abs(steering_diff) - self.max_steering_diff
                    jerk_penalty -= error / self.action_repeat
                #print(action)
                #print("break penalty ", break_penalty, "throttle ", throttle_reward, "jerk penalty ", jerk_penalty, "reward ", reward)
                total_reward = reward + green_penalty + break_penalty + throttle_reward + jerk_penalty
            else:
                done = die
                total_reward += reward
            self.steps += 1
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        """
        Show training in video
        """
        self.env.render(*arg)

    def rgb2gray(self, rgb):
        """
        Transform images to gray and (in case selected) normalize
        Use Tensor flow
        """
        gray = self.transform(rgb).to("cpu")
        gray = gray.squeeze(0)
        gray = gray.numpy()
        return gray

    def set_reward(self, reward_typ):
        self.reward_typ = reward_typ

    def set_eps(self, eps):
        self.eps = eps

    def set_steps(self, steps):
        self.steps = steps

    def set_last_action(self, action):
        self.last_action = action

    @staticmethod
    def reward_memory():
        """
        Record reward for last 100 steps
        """
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
