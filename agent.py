import numpy as np 
from policy import Policy
from learning_algorithms import Tabular
from discretize import Discretize
from shared import bounds
import shared
import sys


class Agent:
      def __init__(self, pi:Policy, learning_strategy:Tabular):

            self.strategy = learning_strategy
            self.pi = pi
            self.d = Discretize(bounds)

            self.reward_history = []            

            self.t = 0

      def act(self, env, n_episodes, max_iter):

            max_iter = max_iter

            for i in range(n_episodes):
                  
                  # this variable stores the total reward obtained in a particular episode

                  self.cache = 0

                  s_t, _ = env.reset()

            
                  terminated = False

                  s_t = tuple(self.d.get_index(s_t))
                  a_t = self.pi.get_action(s_t)
      

                  

                  oldq = shared.Q.copy()

                  for j in range(max_iter) :
                        
                        
                        observation, reward, terminated, truncated, info = env.step(a_t)
                        s_t1 = tuple(self.d.get_index(observation))
                        a_t1 = self.pi.get_action(s_t1)
                        
                        
                        self.strategy.update(s_t, a_t, reward, s_t1, a_t1)
                       
                        self.cache += reward
                      
                        s_t = s_t1
                        a_t = a_t1

                        if terminated:
                              break
                  
                  self.reward_history.append(self.cache)

                  if i % 100 == 0:

                        print(f'episode : {i}, Average score : {np.mean(np.array(self.reward_history[-100:]))}', end = '\r')

                        if np.mean(self.reward_history[-100:]) >= 450:
                              break
            

      
                  
            



