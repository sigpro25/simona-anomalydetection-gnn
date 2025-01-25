
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from data.data_loader import SignalDataset
from src.model.GNNModel import GNNModel


class GNNhandler:
    def __init__(self, model_weights_pth, clf, load_weights=False, batch_size=10, 
                 num_epochs=10, n_points=1000, scale=1):
        
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.load_weights = load_weights
        self.model_weights_pth = model_weights_pth

        self.model = GNNModel(input_size=2, hidden_size=20, output_size=1, expected_n_points=n_points, expected_scale=scale)
        self.training_criterion = nn.HuberLoss()
        self.evaluation_criterion = nn.MSELoss()
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self._load_model_weights()

        self.clf = clf

    def _load_model_weights(self):
        if self.load_weights:
            self.model.load_state_dict(torch.load(self.model_weights_pth))

    def train_model(self, training_dataset_path, testing_dataset_path):
        trainingdataset = SignalDataset(training_dataset_path)
        testingdataset = SignalDataset(testing_dataset_path)
        
        trainingloader = DataLoader(trainingdataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        testingloader = DataLoader(testingdataset, batch_size=1, shuffle=False)
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for sample in tqdm(trainingloader, leave=False):
                inputsX, inputsY, trend = sample["x"], sample["y"], sample["trend"]
                inputs = torch.concatenate((inputsX.unsqueeze(-1), inputsY.unsqueeze(-1)), 2)

                batch = Batch.from_data_list([Data(x=curve.view(-1, 2)) for curve in inputs.float()])
                
                self.model_optimizer.zero_grad()
                output_1, output_2, output_3 = self.model(batch)

                output_1 = output_1.view(self.batch_size, self.n_points, 1).squeeze(-1)
                output_2 = output_2.view(self.batch_size, self.n_points, 1).squeeze(-1)
                output_3 = output_3.view(self.batch_size, self.n_points, 1).squeeze(-1)

                loss1 = self.training_criterion(output_1, trend)
                loss2 = self.training_criterion(output_2, trend)
                loss3 = self.training_criterion(output_3, trend)
                
                loss1.backward()
                loss2.backward()
                loss3.backward()
                
                self.model_optimizer.step()
                running_loss += (loss1 + loss2 + loss3).item()

            testing_loss = self._evaluate_model(testingloader)
            print(f"Epoch {epoch + 1}, Training Loss: {(running_loss / len(trainingloader)):.5f}, Testing Loss: {(testing_loss / len(testingloader)):.5f}")

        print("Training ended.")
        torch.save(self.model.state_dict(), self.model_weights_pth)

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

    def _evaluate_model(self, testingloader):
        self.model.eval()
        testing_loss = 0.0
        for sample in testingloader:
            inputsX, inputsY, trend = sample["x"], sample["y"], sample["trend"]
            inputs = torch.concatenate((inputsX.unsqueeze(-1), inputsY.unsqueeze(-1)), 2)

            batch = Batch.from_data_list([Data(x=curve.view(-1, 2)) for curve in inputs.float()])
            with torch.no_grad():
                outputs = self.model(batch).t()
            
            loss = self.evaluation_criterion(outputs, trend)
            testing_loss += loss.item()
        return testing_loss
        
    def evaluate(self, sample, pseudoannotate=False):
        inputsX, inputsY = torch.from_numpy(sample["x"]).unsqueeze(0), torch.from_numpy(sample["y"]).unsqueeze(0)
        if not pseudoannotate:
            peaks = torch.from_numpy(sample["peaks"]).unsqueeze(0)
        
        inputs = torch.concatenate((inputsX.unsqueeze(-1), inputsY.unsqueeze(-1)), 2)
        batch = Batch.from_data_list([Data(x=curve.view(-1, 2)) for curve in inputs.float()])

        self.model.eval()
        with torch.no_grad():
            trend = self.model(batch).t()
        
        sample["regression"] = trend.squeeze(0).detach().cpu().numpy()
        if not pseudoannotate:
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

