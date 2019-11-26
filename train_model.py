# coding=utf-8
import argparse, os
import tensorflow as tf
from PIL import Image
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from ICMDP_Env import ClassifyEnv
from  get_model import get_text_model, get_image_model
from data_pre import load_data, get_imb_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--data',choices=['mnist', 'cifar10','famnist','imdb'], default='famnist')
parser.add_argument('--model', choices=['image', 'text'], default='image')
parser.add_argument('--imb-rate',type=float, default=0.05)
parser.add_argument('--min-class', type=str, default='456')
parser.add_argument('--maj-class', type=str, default='789')
parser.add_argument('--training-steps', type=int, default=120000)
args = parser.parse_args()
data_name = args.data



x_train, y_train, x_test, y_test = load_data(data_name)
imb_rate = args.imb_rate
maj_class = list(map(int, list(args.maj_class)))
min_class = list(map(int, list(args.min_class)))

x_train, y_train, x_test, y_test = get_imb_data(x_train, y_train, x_test, y_test, imb_rate, min_class, maj_class)
print(x_train.shape, y_train.shape)
in_shape = x_train.shape[1:]
num_classes = len(set(y_test))
mode = 'train'
env = ClassifyEnv(mode, imb_rate, x_train, y_train)
nb_actions = num_classes
training_steps = args.training_steps
if args.model == 'image':
    model = get_image_model(in_shape, num_classes)
else:
    in_shape = [5000, 500]
    model = get_text_model(in_shape, num_classes)

INPUT_SHAPE = in_shape
print(model.summary())


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        if args.model == 'text':
            return observation
        img = observation.reshape(INPUT_SHAPE)
        processed_observation = np.array(img)
        return processed_observation

    def process_state_batch(self, batch):
        if args.model == 'text':
            return batch.reshape((-1, INPUT_SHAPE[1]))
        batch = batch.reshape((-1,) + INPUT_SHAPE)
        processed_batch = batch.astype('float32') / 1.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


memory = SequentialMemory(limit=100000, window_length=1)
processor = ClassifyProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=0.5, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env, nb_steps=training_steps, log_interval=60000)


env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)
env = ClassifyEnv(mode, imb_rate, x_test, y_test)
env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)

