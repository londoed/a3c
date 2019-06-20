import numpy as np
import tensorflow as tf
import sonnet as snt
import gym

import os
import threading
import multiprocessing
import argparse
from queue import Queue

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()


class A3C(snt.AbstractModule):
    def __init__(self, state_size, action_size, name='a3c'):
        super(A3C, self).__init__(name=name)
        self.state_size = state_size
        self.action_size = action_size

    def _build(self, inputs):
        logits = snt.Sequential([
            snt.Linear(100), tf.nn.relu,
            snt.Linear(self.action_size)
        ])

        values = snt.Sequential([
            snt.Linear(100), tf.nn.relu,
            snt.Linear(1)
        ])
        return logits(inputs), values(inputs)

class RandomAgent():
    def __init__(self, env_name, max_eps):
        self.env = env
        self.max_episodes = max_episodes
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            is_dome = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not is_done:
                _, reward, is_done, _ = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        return final_avg

class MasterAgent():
    def __init__(self):
        self.env_name = env_name
        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
        self.global_model = A3C(self.state_size, self.action_size)
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        if args.algorithm == 'random':
            random_agent = RandomAgent(self.game_name, args.max_episodes)
            random_agent.run()
            return
        res_queue = Queue()

        workers = [Worker(self.state_size,
                          self.action_size,
                          self.opt, res_queue,
                          i, game_name=self.game_name)
                          for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(worker):
            worker.start()

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        model = self.global_model
        is_done = False
        step_counter = 0
        reward_sum = 0

        while not is_done:
            env.render(mode='rgb_array')
            policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            policy = tf.nn.softmax(policy)
            action = np.argmax(policy)
            state, reward, is_done, _ = env.step(action)
            reward_sum += reward
            step_counter += 1
        env.close()

class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Worker(threading.Thread):
    global_episode = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, state_size, action_size,
                 global_model, opt, res_queue,
                 idx, game_name):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = A3C(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            time_count = 0
            is_done = False
            while not is_done:
                logits, _ = self.local_model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, is_done, _ = self.env.step(action)
                if done:
                    reward = 1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == args.update_freq or is_done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(is_done, new_state, mem, args.gamma)
                    self.ep_loss += total_loss
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if is_done:
                        if ep_reward > Worker.best_score:
                            Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1
                time_count += 1
                current_state = new_state
                total_step += 1
        self.res_queu.put(None)

    def compute_loss(self, is_done, new_state, memory, gamma=0.99):
        if is_done:
            reward_sum = 0.
        else:
            reward_sum = self.local_model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32)).numpy()[0]

        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values

        value_loss = advantage**2

        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)

        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

if __name__ == '__main__':
    print(args)
    master = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()
