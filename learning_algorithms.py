import numpy as np
import shared

class Tabular:
      def __init__(self, *args, **kwargs):
            pass 
      def update(self,*args, **kwargs):
            pass


class Q_learning(Tabular):
      def __init__(self, learning_rate:float, gamma:float):


            self.lr = learning_rate
            self.gamma = gamma
            

      def update(self, s_t, a_t, r_t1, s_t1, *args):

            shared.Q[s_t][a_t] = shared.Q[s_t][a_t] + self.lr * (r_t1 + self.gamma * max(shared.Q[s_t1]) - shared.Q[s_t][a_t])


class SARSA(Tabular):
      def __init__(self, learning_rate:float, gamma:float):


            self.lr = learning_rate
            self.gamma = gamma

      def update(self, s_t, a_t, r_t1, s_t1, a_t1):

            shared.Q[s_t][a_t] +=  self.lr * (r_t1 + self.gamma * shared.Q[s_t1][a_t1] - shared.Q[s_t][a_t])

            
      