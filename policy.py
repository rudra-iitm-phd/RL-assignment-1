import numpy as np
from shared import Q
import shared

class Policy:
      def __init__(self):
            Q = None
      def get_action(self, s_t):
            pass
      


class EpsGreedy(Policy):
      def __init__(self, eps:float):
            
            self.eps = eps
      
      def get_action(self, s_t):
            if np.random.rand() < self.eps :
                  return np.random.choice(range(shared.n_actions))
            else:
                  return np.argmax(Q[s_t])