import gym
import numpy as np
from gym import wrappers
import os
class Hp():
	#Hyperparameters
	def __init__(self,
				env_name,
				num_delta=16,
				num_best_delta=16,
				episode_length=2000,
				total_episode=800,
				learning_rate=0.02,
				noise_filter=0.03,
				record_at_epi=50):
		self.num_delta = num_delta
		self.num_best_delta = num_best_delta
		self.episode_length = episode_length
		self.total_episode = total_episode
		self.env_name = env_name
		self.noise_filter = noise_filter
		self.learning_rate = learning_rate
		self.record_at_epi = record_at_epi
class Normalize():
	def __init__(self, input_size):
		self.n = 0
		self.mean = np.zeros(input_size)
		self.mean_diff = np.zeros(input_size)
		self.var = np.zeros(input_size)
	def normalize_input(self, x):
		self.n += 1
		x_old = x
		old_mean = self.mean.copy()
		self.mean += (x-self.mean)/self.n 
		self.mean_diff += (x-self.mean)*(x-old_mean)
		self.var = (self.mean_diff/self.n).clip(min = 1e-2)
		obs_mean = self.mean
		std = np.sqrt(self.var)
		return (x_old -obs_mean)/std

class Policy():
	def __init__(self, input_size, output_size,hp):
		self.theta = np.zeros((output_size, input_size))
		self.learning_rate = hp.learning_rate
		self.hp = hp
		self.num_best_delta = hp.num_best_delta
	def get_noise(self):
		return [  np.random.randn(*self.theta.shape) for _ in range(hp.num_delta) ]
	def format_actions(self, input, noise = None, direction=None):
		if direction == '+':
			#print (self.theta.shape)
			#print (noise)
			return (self.theta + self.hp.noise_filter*noise).dot(input)
		elif direction == '-':
			return (self.theta - self.hp.noise_filter*noise).dot(input)
		elif direction is None:
			return (self.theta).dot(input)
	def update(self, rollouts, sigma_reward):
		step = np.zeros(self.theta.shape)
		for r_pos, r_neg, noise in rollouts:
			step += (r_pos - r_neg)*noise
		self.theta +=  self.learning_rate/(sigma_reward*self.num_best_delta)*step

class ArsTrainer():
	def __init__(self, hp, monitor_dir):
		self.hp = hp
		self.env = gym.make(hp.env_name)
		self.monitor_dir = monitor_dir
		np.random.seed(1946)
		self.input_size = self.env.observation_space.shape[0]
		self.output_size = self.env.action_space.shape[0]
		self.hp.episode_length = self.env.spec.timestep_limit
		self.policy = Policy(self.input_size,self.output_size, hp)
		self.normalize_obj = Normalize(self.input_size)
		self.num_best_delta = hp.num_best_delta
		self.record_epi = False
		if monitor_dir is not None:
			should_record = lambda i: self.record_epi
			self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)


	def epi_run(self, direction=None, noise=None):

		done = False
		k = 0
		total_reward = 0
		state = self.env.reset()
		while not done and k < self.hp.episode_length:
			state = self.normalize_obj.normalize_input(state)
			action = self.policy.format_actions(state, noise, direction)
			state, reward, done, info = self.env.step(action)
			if direction is None:
				self.env.render()
			reward = max(min(reward,1),-1)
			total_reward +=reward
			k +=1
		return total_reward
	def train(self):
		positive_reward = [0]*hp.num_delta
		negative_reward = [0]*hp.num_delta
		self.number=1
		#policy = Policy(input_size, output_size, hp)
		#normalize = Normalize(input_size)
		#state = env.reset()
		for i in range (hp.total_episode):
			#state = normalize.normalize_input(state)
			noise = self.policy.get_noise()
			#print (noise[0])
			for j in range(hp.num_delta):
				#actions_positivte = Policy.format_actions(noise[_],input,direction='+')
				#actions_negative = Policy.format_actions(noise[_],input,direction='-')
				positive_reward[j] = self.epi_run('+', noise[j])
				negative_reward[j] = self.epi_run('-', noise[j])
			#print (positive_reward)
			#print (negative_reward)
			sigma_reward = np.array(positive_reward + negative_reward).std()
			scores = {index:max(pos,neg) for index,(pos,neg) in enumerate(zip(positive_reward,negative_reward))}
			sort_list = sorted(scores.keys() , key = lambda x: scores[x] , reverse= True)[:hp.num_best_delta]
			rollouts = [(positive_reward[sorce_iter], negative_reward[sorce_iter], noise[sorce_iter])for sorce_iter in sort_list]
			self.policy.update(rollouts, sigma_reward)
			if i % self.hp.record_at_epi  == 0:
				self.record_epi = True
			test_run = self.epi_run()
			self.record_epi = False
			
			print (self.number, test_run)
			self.number = self.number +1

if __name__ == '__main__':
	hp = Hp(env_name='BipedalWalker-v2')
	vidoe_dir="shubham_ars"
	if not os.path.exists(vidoe_dir):
		os.makedirs(vidoe_dir)
	ars = ArsTrainer(hp, vidoe_dir)	
	ars.train()