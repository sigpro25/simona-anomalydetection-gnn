
import numpy as np
import json

from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, json_file, max_samples=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        y = np.array(sample['data'], dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)
        trend = np.array(sample['trend'], dtype=np.float32)
        if "peaks" in sample.keys():
            peaks = np.array(sample['peaks'], dtype=np.float32)
        else:
            peaks = np.zeros_like(trend, dtype=np.float32)
        
        min_value_y = np.min(y)
        max_value_y = np.max(y)

        y = (y-min_value_y)/(max_value_y-min_value_y)
        trend = (trend-min_value_y)/(max_value_y-min_value_y)
        x /= np.max(x)
        
        return {'x': x, 'y': y, 'trend': trend, 'peaks': peaks}
