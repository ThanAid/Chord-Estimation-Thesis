import pandas as pd
import numpy as np


class AdaptLabels:
    def __init__(self, label_path, n_steps):
        self.time_labels = pd.read_csv(label_path, delimiter=' ', names=['start', 'end', 'chord'])
        self.n_steps = n_steps
        self.duration = self.get_duration()
        self.timestep = self.get_timestep()
        self.timestamps = np.linspace(0, self.duration, num=n_steps)
        self.labels = self.adapt_labels()

    def get_duration(self):
        """returns duration of track"""
        
        return self.time_labels['end'].iloc[-1]
    
    def get_timestep(self):
        """returns the duration of each time step"""
        
        return self.duration/self.n_steps
        
    def adapt_labels(self):
        """transforms time based labels to frame based labels"""
    
        i = 0
        labels = []
        for timestamp in self.timestamps:
            if timestamp <= self.time_labels["end"][i]:
                labels.append((timestamp, self.time_labels["chord"][i]))
            else:
                i+=1
                labels.append((timestamp, self.time_labels["chord"][i]))
            
        return labels
