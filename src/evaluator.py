
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

from src.results_visualiser import MultiLineGraph


class Evaluator:
    def __init__(self, epsilon, model, dataset):
        self.epsilon = epsilon
        self.model = model
        self.dataset = dataset
        self.arr_out = []

    def _custom_confusion_matrix(self, prediction, target):
        n = len(prediction)
        prediction_positive = ~np.isnan(prediction)
        target_positive = (target == 1)
        target_negative = (target == 0)

        TP = FP = 0
        for i in np.where(prediction_positive)[0]:
            start, end = max(0, i - self.epsilon), min(n, i + self.epsilon + 1)
            if np.any(target[start:end] == 1):
                TP += 1
            else:
                FP += 1

        TN = np.sum((~prediction_positive) & target_negative)
        FN = np.sum((~prediction_positive) & target_positive)
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    def _calculate_metrics(self, confusion_matrix):
        TP, FP, TN, FN = confusion_matrix["TP"], confusion_matrix["FP"], confusion_matrix["TN"], confusion_matrix["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = np.sqrt(recall*(TN/(FP+TN))*precision*(TN/(FN+TN)))
        return precision, recall, f1_score, mcc

    def evaluate_all_samples(self):
        predictions = np.array([])
        targets = np.array([])
        trend = np.array([])
        regression = np.array([])

        for sample in tqdm(self.dataset):
            out = self.model.evaluate(sample)
            self.arr_out.append(out)
            predictions = np.append(predictions, out['classification'])
            targets = np.append(targets, out['peaks'])
            trend = np.append(trend, out['trend'])
            regression = np.append(regression, out['regression'])
        
        confmat = self._custom_confusion_matrix(predictions, targets)
        precision, recall, f1_score, mcc = self._calculate_metrics(confmat)

        mse = mean_squared_error(trend, regression)
        r2 = r2_score(trend, regression)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 score: {f1_score:.2f}")
        print(f"Matthews correlation coefficient: {mcc:.2f}")
        print(f"Trend estimation MSE: {mse:.10f}")
        print(f"Trend R2 score: {r2:.10f}")
    
    def visualize_per_sample(self, max_idx=10):
        for idx in range(len(self.arr_out)):
            out = self.arr_out[idx]
            self._plot_with_multilinegraph(out)
            if idx > max_idx:
                break

        for idx in range(len(self.arr_out)):
            out = self.arr_out[idx]
            plt.figure(figsize=(8, 2))
            plt.plot(out['y'], label="data")
            plt.plot(out['regression'], label="estimated trend")
            plt.scatter(out['x']*1000, out['classification'], label="found anomalies", c="r", s=40, marker="x")
            plt.legend()
            plt.show()
            if idx > max_idx:
                break
    
    def _plot_with_multilinegraph(self, out):
        title = f"GNN with attention"
        MultiLineGraph(title,
            out['x']*1000, out['y'], out['trend'], 
            [
                {'y': out['regression'], 'name': 'gnn', 'type': 'plot'},
                {'y': out['classification'], 'name': 'classification', 'type': 'scatter'}
            ]
        ).plot_curves()
        