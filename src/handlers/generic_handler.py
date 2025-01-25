
import numpy as np

import torch
import torch.nn as nn


class GenericHandler:
    def __init__(self, model, clf):
        self.model = model
        self.clf = clf
        self.evaluation_criterion = nn.MSELoss()

    def _extract_classification_training_data(self, residuas, labels):
        # data shape (batch, N, 1)
        # labels shape (batch, N, 1)
        # outputs shapes (batch*(N-2*11), 11) (batch*(N-2*11))

        offsets = torch.tensor([-11, -9, -7, -5, -3, 0, 3, 5, 7, 9, 11])
        start_idx = offsets.abs().max()
        end_idx = residuas.shape[1] - offsets.abs().max()
        qdim = end_idx - start_idx

        def extend(x):
            N = x.shape[1]
            x = x.squeeze(0)            
            x = torch.cat([
                x,
                (x[0]-x[N//2]).unsqueeze(0),
                (x[1]-x[N//2]).unsqueeze(0),
                (x[2]-x[N//2]).unsqueeze(0),
                (x[3]-x[N//2]).unsqueeze(0),
                (x[4]-x[N//2]).unsqueeze(0),
                (x[N-1] - x[N//2]).unsqueeze(0),
                (x[N-2] - x[N//2]).unsqueeze(0),
                (x[N-3] - x[N//2]).unsqueeze(0),
                (x[N-4] - x[N//2]).unsqueeze(0),
                (x[N-5] - x[N//2]).unsqueeze(0)
            ], dim=0).unsqueeze(0)
            return x

        def normalise(X):
            median = torch.median(X, dim=0).values
            q1 = torch.quantile(X, 0.25, dim=0)
            q3 = torch.quantile(X, 0.75, dim=0)
            iqr = q3 - q1
            return (X - median) / iqr

        newdata = torch.stack([extend(residuas[:, i + offsets]) for i in range(start_idx, end_idx)], dim=1)
        newdata = normalise(newdata.view(qdim * residuas.shape[0], newdata.shape[-1]))
        newlabels = labels[:, start_idx:end_idx]
        newlabels = newlabels.reshape(qdim * labels.shape[0])

        return newdata, newlabels
        
    def evaluate(self, sample):
        inputsY = torch.from_numpy(sample["y"]).unsqueeze(0)
        peaks = torch.from_numpy(sample["peaks"]).unsqueeze(0)
        
        trend = torch.from_numpy(self.model(sample["y"])).unsqueeze(0)        
        
        sample["regression"] = trend.squeeze(0).detach().cpu().numpy()
        sample["classification"] = self.classify(trend, inputsY, peaks)

        return sample

    def classify(self, trend, inputsY, peaks):
        ps = 11
        padded_trend = torch.nn.functional.pad(trend, pad=(ps, ps), value=0)
        padded_inputsY = torch.nn.functional.pad(inputsY, pad=(ps, ps), value=0)
        padded_peaks = torch.nn.functional.pad(peaks, pad=(ps, ps), value=0)

        residuas = padded_inputsY-padded_trend

        transformed_data, _ = self._extract_classification_training_data(residuas, padded_peaks)
        transformed_data = transformed_data.detach().cpu().numpy()

        prediction = self.clf.fit_predict(transformed_data)
        prediction[prediction == 1] = 0
        prediction[prediction == -1] = 1
        prediction = np.where(
            prediction == 1, 
            inputsY.squeeze(0).detach().cpu().numpy(), 
            np.nan
        )

        return prediction

