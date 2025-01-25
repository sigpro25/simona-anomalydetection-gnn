
import numpy as np

from pybaselines import Baseline
from sklearn.neighbors import LocalOutlierFactor

from data.data_loader import SignalDataset
from src.handlers.gnn_handler import GNNhandler
from src.handlers.generic_handler import GenericHandler
from src.evaluator import Evaluator


def evaluate_asls_lof(json_file, max_samples=None, visualise=False):
    baseline_fitter = Baseline(x_data=np.arange(1000)/1000)
    handler = GenericHandler(
        model=lambda x: baseline_fitter.asls(x, lam=200, p=0.3, max_iter=100)[0], 
        clf=LocalOutlierFactor(n_neighbors=900)
    )

    if max_samples is not None and max_samples > 0:
        dataset = SignalDataset(json_file, max_samples=max_samples)
    else:
        dataset = SignalDataset(json_file)
    
    evaluator = Evaluator(epsilon=11, model=handler, dataset=dataset)
    evaluator.evaluate_all_samples()
    
    if visualise:
        evaluator.visualize_per_sample()

def evaluate_snip_lof(json_file, max_samples=None, visualise=False):
    baseline_fitter = Baseline(x_data=np.arange(1000)/1000)
    handler = GenericHandler(
        model=lambda x: baseline_fitter.asls(x, lam=200, p=0.3, max_iter=100)[0], 
        clf=LocalOutlierFactor(n_neighbors=900)
    )

    if max_samples is not None and max_samples > 0:
        dataset = SignalDataset(json_file, max_samples=max_samples)
    else:
        dataset = SignalDataset(json_file)
    
    evaluator = Evaluator(epsilon=11, model=handler, dataset=dataset)
    evaluator.evaluate_all_samples()
    
    if visualise:
        evaluator.visualize_per_sample()

def evaluate_gnn_lof(json_file, model_file, train=False, max_samples=None, visualise=False, test_json_file=None):
    handler = GNNhandler(
        load_weights=True, batch_size=20, num_epochs=10, n_points=1000, scale=1, 
        model_weights_pth=model_file, clf=LocalOutlierFactor(n_neighbors=900)
    )
    
    if train and test_json_file is not None:
        handler.train_model(json_file, test_json_file)

    if max_samples is not None and max_samples > 0:
        dataset = SignalDataset(json_file, max_samples=max_samples)
    else:
        dataset = SignalDataset(json_file)
    
    evaluator = Evaluator(epsilon=11, model=handler, dataset=dataset)
    evaluator.evaluate_all_samples()
    
    if visualise:
        evaluator.visualize_per_sample()



if __name__ == "__main__":

    synthetic_json_file = "data/2000_generated_signals.json"

    print("ASLS and LOF:")
    evaluate_asls_lof(synthetic_json_file)
    print()
    print("SNIP and LOF:")
    evaluate_snip_lof(synthetic_json_file)
    print()
    print("GNN and LOF:")
    evaluate_gnn_lof(synthetic_json_file, "src/model/gnn_frontal.pth", train=False)
