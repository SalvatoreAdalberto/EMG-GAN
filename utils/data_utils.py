# Laboratory of Robotics and Cognitive Science
# Version by:  Rafael Anicet Zanini
# Github:      https://github.com/larocs/EMG-GAN

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataLoader():
    def __init__(self, args):
        self.file_path = args['training_file']
        self.features = args['features']
        self.channels = len(self.features)
        self.rescale = args['rescale']
        self.num_steps = args['num_steps']
        self.train_split = args['train_split']
        self.batch_size = args['batch_size']
    
    def load_training_data(self):
    
        data = self.load_timeseries(self.file_path, self.features)
    
        #Normalize data before hand
        values = data.values
        
        if self.rescale:
            values, scalers = self.min_max(values,-1.0,1.0)
        else:
            values, scalers = self.normalize(values)
        
        #Get moving windows samples
        X_windows = self.get_windows(values,self.num_steps)
        data = np.array(X_windows)
        filter_size = round(data.shape[0] * self.train_split)
        data = data[0:filter_size]
        
        return data
    
    def load_timeseries(self, filename, series):
        #Load time series dataset
        loaded_series = pd.read_pickle(filename)
    
        #Applying filter on the selected 
    
        return loaded_series
    
    
    
    def get_training_batch(self):
        x_train = self.load_training_data()
        idx = np.random.randint(0, x_train.shape[0], self.batch_size)
        signals = x_train[idx]
        signals = np.reshape(signals, (signals.shape[0],signals.shape[1],self.channels))
        return signals