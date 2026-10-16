import os
import sys
import yaml
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import dgl

from utils import set_global_seed
from evaluation import calculate_fpr95
from losses import LogRatioLossAD, AnomalyDetectionLoss  
from model import LGKDE
from dataprocessing import load_graph_dataset
from ng import NegativeSampleGenerator

from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           f1_score, precision_score, recall_score)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, epoch, current_performance):
        if epoch < self.min_epochs:
            return False
            
        if self.best_loss is None:
            self.best_loss = current_performance
            return False
            
        if current_performance > self.best_loss + self.min_delta:
            self.best_loss = current_performance
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
    
class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_experiment()
        self.init_components()
        
    def setup_environment(self):

        set_global_seed(self.config['base']['seed'])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['base']['gpu_id']

        # self.device = torch.device(f"cuda:{self.config['base']['gpu_id']}" if torch.cuda.is_available() else "cpu")        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        

        self.save_dir = os.path.join(self.config['base']['save_dir'])
        for dir_name in ['SavedModels', 'ExperimentRecords']:
            os.makedirs(os.path.join(self.save_dir, dir_name), exist_ok=True)
            
    def setup_experiment(self):

        self.train_loader, self.test_loader, self.train_loader_noshuffle, self.full_loader = load_graph_dataset(
            dataset_name=self.config['dataset']['name'],
            dataset_type=self.config['dataset']['type'],
            normal_class=self.config['dataset']['normal_class'],
            train_ratio=self.config['dataset']['train_ratio'],
            mixed_anomaly_ratio=self.config['dataset']['mixed_anomaly_ratio'],
            self_loop=self.config['dataset']['self_loop'],
            batch_size=self.config['dataset']['batch_size']
        )
        
    def init_components(self):
        in_dim = self.config['model'].get('in_dim')
        if in_dim is None:
            in_dim = self.infer_input_dim()

        self.model = LGKDE(
            in_dim=in_dim,
            hidden_dim=self.config['model']['hidden_dim'],
            out_dim=self.config['model']['out_dim'],
            num_layers=self.config['model']['num_layers'],
            bandwidths=self.config['model']['bandwidths'],
            dropout=self.config['model']['dropout'],
            batch_norm=self.config['model']['batch_norm'],
            learn_kde_weights=self.config['model']['learn_kde_weights']
        ).to(self.device)
        

        self.neg_gen = NegativeSampleGenerator(**self.config['negative_sampling'])
        

        if self.config['training']['loss_type'] == "LogRatioLossAD":
            self.criterion = LogRatioLossAD(epsilon=self.config['training']['epsilon'])
        else:
            self.criterion = AnomalyDetectionLoss(epsilon=self.config['training']['epsilon'])
            

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        

        self.writer = SummaryWriter(f"runs/{self.config['dataset']['name']}_{self.timestamp}")
        self.best_performance = 0
        self.best_model_path = None
        self.train_losses = []
        self.eval_metrics_history = []

        early_stop_config = self.config.get(
            'early_stopping',
            self.config['training'].get('early_stopping', {})
        )
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            min_epochs=early_stop_config.get('min_epochs', 5)
        )

    def infer_input_dim(self):
        first_graph, _ = self.train_loader.dataset[0]
        return first_graph.ndata['attr'].shape[1]
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        for batched_graph, _ in self.train_loader:
            batched_graph = batched_graph.to(self.device)
            self.optimizer.zero_grad()
            

            train_distance_matrix = self.model.dgmmd.compute_distance_matrix(batched_graph, None)
            P_normal = self.model.compute_kde_scores(train_distance_matrix)

            graphs_cpu = batched_graph.to('cpu')
            graphs_list = dgl.unbatch(graphs_cpu)
            neg_graphs_sub = [self.neg_gen.generate_negative_graph(g, increase_edges=False) for g in graphs_list]
            neg_graphs_add = [self.neg_gen.generate_negative_graph(g, increase_edges=True) for g in graphs_list]
            
            neg_sub_batched = dgl.batch(neg_graphs_sub).to(self.device)
            neg_add_batched = dgl.batch(neg_graphs_add).to(self.device)
            

            neg_sub_distance_matrix = self.model.dgmmd.compute_distance_matrix(neg_sub_batched, batched_graph)
            neg_add_distance_matrix = self.model.dgmmd.compute_distance_matrix(neg_add_batched, batched_graph)
            
            P_neg_sub = self.model.compute_kde_scores(neg_sub_distance_matrix)
            P_neg_add = self.model.compute_kde_scores(neg_add_distance_matrix)
            

            loss = 0.5 * self.criterion(P_normal, P_neg_sub) + 0.5 * self.criterion(P_normal, P_neg_add)
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():

            train_scores_list, train_labels_list = [], []
            for train_batched_graph, train_labels in self.train_loader_noshuffle:
                train_batched_graph = train_batched_graph.to(self.device)
                train_labels = train_labels.to(self.device)
                ref_scores = self.model.get_reference_scores(train_batched_graph)
                train_scores_list.append(ref_scores)
                train_labels_list.append(train_labels)
                
            train_scores = torch.cat(train_scores_list, dim=0)
            train_labels_all = torch.cat(train_labels_list, dim=0)
            

            test_scores_list, test_labels_list = [], []
            for test_batched_graph, test_labels in self.test_loader:
                test_batched_graph = test_batched_graph.to(self.device)
                test_labels = test_labels.to(self.device)
                query_scores = self.model.get_query_scores(test_batched_graph, train_batched_graph)
                test_scores_list.append(query_scores)
                test_labels_list.append(test_labels)
                
            test_scores = torch.cat(test_scores_list, dim=0)
            test_labels_all = torch.cat(test_labels_list, dim=0)
            

            combined_scores = torch.cat([train_scores, test_scores], dim=0)
            threshold = torch.quantile(combined_scores, self.config['evaluation']['anomaly_threshold_percentile'])
            predicted_anomalies = (test_scores <= threshold).int()
            
            y_true = test_labels_all.detach().cpu().numpy()
            y_score = -test_scores.detach().cpu().numpy()
            y_pred = predicted_anomalies.detach().cpu().numpy()
            
            metrics = {
                'AUROC': roc_auc_score(y_true, y_score),
                'AUPR': average_precision_score(y_true, y_score),
                'F1-Score': f1_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'FPR95': calculate_fpr95(y_true, y_score),
                'Threshold': threshold.item(),
                'combined_scores': combined_scores.detach().cpu().numpy()
            }
            
            return metrics
            
    def save_results(self, metrics, epoch):

        current_performance = (metrics['AUROC'] + metrics['AUPR']) / 2

        if current_performance > self.best_performance:
            self.best_performance = current_performance
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            self.best_model_path = os.path.join(
                self.save_dir, 
                'SavedModels', 
                f"{self.config['dataset']['name']}_best_{self.timestamp}.pth"
            )
            torch.save(self.model.state_dict(), self.best_model_path)
            

        self.eval_metrics_history.append(metrics)
        

        for k, v in metrics.items():
            if k != 'combined_scores':
                self.writer.add_scalar(f'Metrics/{k}', v, epoch)
                
    def plot_results(self):

        plt.figure(figsize=(12, 4))
        

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        

        plt.subplot(1, 2, 2)
        metrics_to_plot = ['AUROC', 'AUPR', 'FPR95']
        for metric in metrics_to_plot:
            values = [m[metric] for m in self.eval_metrics_history]
            plt.plot(values, label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'ExperimentRecords', 
                                f"{self.config['dataset']['name']}_performance_{self.timestamp}.pdf"))
        plt.close()
        

        best_index = np.argmax([
            (m['AUROC'] + m['AUPR'])/2 for m in self.eval_metrics_history
        ])
            
        best_scores = self.eval_metrics_history[np.argmax([
            (m['AUROC'] + m['AUPR'])/2 for m in self.eval_metrics_history
        ])]['combined_scores']
        print(f"\nBest Model Performance:") 
        print(f"Best AUROC:{self.eval_metrics_history[best_index]['AUROC']:.4f}")
        print(f"Best AUPR:{self.eval_metrics_history[best_index]['AUPR']:.4f}")
        print(f"Best FPR95:{self.eval_metrics_history[best_index]['FPR95']:.4f}")
        print(f"Best Threshold:{self.eval_metrics_history[best_index]['Threshold']:.4f}")
        
        plt.figure(figsize=(8, 4))
        plt.plot(best_scores)
        plt.title(f"Combined Scores Best Model: Epoch {best_index + 1} AUROC: {self.eval_metrics_history[best_index]['AUROC']:.4f} AUPR: {self.eval_metrics_history[best_index]['AUPR']:.4f} FPR95: {self.eval_metrics_history[best_index]['FPR95']:.4f}")
        plt.xlabel("Sample Index")
        plt.ylabel("Score")
        plt.axhline(y=self.eval_metrics_history[-1]['Threshold'], 
                   color='r', linestyle='--', label='Anomaly Threshold')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'ExperimentRecords',
                                f"{self.config['dataset']['name']}_bestscore_{self.timestamp}.pdf"))
        plt.close()
        
    def train(self):
        print(f"Starting training on {self.config['dataset']['name']} dataset...")
        best_epoch = 0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):

            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}, Loss: {avg_loss:.10f}")
            

            metrics = self.evaluate()
            self.save_results(metrics, epoch)

            print(f"\nEvaluation at epoch {epoch}:")
            for k, v in metrics.items():
                if k != 'combined_scores':
                    if isinstance(v, float):
                        print(f"{k}: {v:.4f}")
                    else:
                        print(f"{k}: {v}")
            
            # Early stopping check
            current_performance = (metrics['AUROC'] + metrics['AUPR']) / 2
            if self.early_stopping(epoch, current_performance):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best performance was achieved at epoch {best_epoch + 1}")
                print(f"Best AUROC: {self.eval_metrics_history[best_epoch]['AUROC']:.4f}")
                print(f"Best AUPR: {self.eval_metrics_history[best_epoch]['AUPR']:.4f}")
                break
                
            if current_performance == self.best_performance:
                best_epoch = epoch - 1
                

        record_path = os.path.join(
            self.save_dir, 
            'ExperimentRecords',
            f"{self.config['dataset']['name']}_record_{self.timestamp}.pkl"
        )
        with open(record_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'train_losses': self.train_losses,
                'eval_metrics_history': self.eval_metrics_history,
                'best_model_path': self.best_model_path,
                'best_epoch': best_epoch,
                'timestamp': self.timestamp,
                'early_stopped': self.early_stopping.early_stop
            }, f)
        

        self.plot_results()
        self.writer.close()
        
        print(f"\nTraining completed. Best model saved to {self.best_model_path}")
        print(f"Training record saved to {record_path}")
        if self.early_stopping.early_stop:
            print("Training was early stopped")
        print(f"Best performance achieved at epoch {best_epoch + 1}")
        
        
if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()
