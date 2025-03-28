import numpy as np 

''' Class for dicretizing the environment '''
class Discretize:
      def __init__(self, bounds:dict):


            assert set(['cp', 'cv', 'pa', 'pv']) == set(bounds.keys())
            ''' 
            bounds is a dictionary in the following format 

            bounds = {'cp':[l, u, n_bins],--> cart position
                      'cv':[l, u, n_bins],--> cart velocity
                      'pa':[l, u, n_bins],--> pole angle
                      'pv':[l, u, n_bins] --> pole velocity
                      }
            u,b --> float values : indicate the upper and the lower bounds
            n_bins --> integer : indicate the resolution

            '''
            self.bounds = bounds
            self.bins = self.generate_bins(self.bounds)

      def generate_bins(self, bounds:dict):
            '''
            Function for initializing the bins 

            '''
            bins = dict()
            for key in bounds.keys():
                  bins[key] = np.linspace(bounds[key][0], bounds[key][1], bounds[key][2])
            return bins
      
      def get_index(self, observation:np.ndarray):
            '''
            observation is taken from gym environment
            and observation for cartpole is a 4 dimensional vector
            where the elements reporesent the cart position, cart velocity, pole angle and pole angular velocity
            respectively

            '''
            idx = []
            observation = observation.copy()
            for (i,j) in zip(observation, self.bins.keys()):
                  idx.append(np.digitize(i, self.bins[j]) - 1)
            return idx

      

