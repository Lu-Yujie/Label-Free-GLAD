import torch
import dgl
from dgl.data import TUDataset, GINDataset
from dgl.dataloading import GraphDataLoader
import pickle

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    
def load_graph_dataset(dataset_name: str, 
                     dataset_type: str = "TU",  # "TU" or "GIN"
                     normal_class: int = None,  # None means auto select the majority class
                     train_ratio: float = 0.8,
                     mixed_anomaly_ratio: float = 0.0,
                     self_loop: bool = True,
                     batch_size: int = None,
                     return_loader: bool = True) -> tuple:
   """
   Load and preprocess graph dataset for anomaly detection.
   
   Args:
       dataset_name: Name of dataset to load
       dataset_type: "TU" or "GIN" dataset type  
       normal_class: Class label to treat as normal class. If None, auto select majority class
       train_ratio: Ratio of normal data for training (default 0.8)
       mixed_anomaly_ratio: Ratio of anomalies to mix into training set (default 0.0)
       self_loop: Whether to add self loops to graphs (default True)
       batch_size: Batch size for training loader. Others use full size
       return_loader: Whether to return DataLoader objects (default True)
       
   Returns:
   if return_loader:
       train_loader: DataLoader for training set
       test_loader: DataLoader for test set 
       train_loader_noshuffle: DataLoader for training set without shuffling
       full_loader: DataLoader for full dataset
   else:
         train_dataset: Dataset for training set
         test_dataset: Dataset for test set
         full_dataset: Dataset for full dataset
   """
   
   # Load dataset
   if dataset_type.upper() == "TU":
       dataset = TUDataset(name=dataset_name)
       # Get graphs and labels from TUDataset
       graphs = []
       labels = []
       for i in range(len(dataset)):
           g, label = dataset[i]
           if self_loop:
               g = dgl.add_self_loop(g)
            # TUDataset ndata 'node_attr': node features, 'node_label': node labels
            # TUDataset edata 'edge_attr': edge features, 'edge_label': edge labels
            # rename and mapping "node_attr" to "attr" and "node_label" to "label"
            # rename and mapping "edge_attr" to "attr" and "edge_label" to "label"
            
           # 
           if 'node_attr' not in g.ndata:
               g.ndata['attr'] = torch.ones(g.num_nodes(), 1)
               # 
               g.ndata['attr'] = g.ndata['attr'].to(torch.float32)
           else:
               g.ndata['attr'] = g.ndata.pop('node_attr')
               g.ndata['attr'] = g.ndata['attr'].to(torch.float32)
            # 
           if 'node_labels' not in g.ndata:
                g.ndata['label'] = torch.zeros(g.num_nodes(), 1)
                g.ndata['label'] = g.ndata['label'].to(torch.float32)
           else:
                g.ndata['label'] = g.ndata.pop('node_labels')
                g.ndata['label'] = g.ndata['label'].to(torch.float32)       
        #    if 'edge_attr' not in g.edata:
        #        g.edata['attr'] = {} 
        #    else:
        #        g.edata['attr'] = g.edata.pop('edge_attr')
        #    if 'edge_labels' not in g.edata:
        #        g.edata['label'] = {}
        #    else:
        #        g.edata['label'] = g.edata.pop('edge_labels')
                           
           graphs.append(g)
           # TUDataset returns label as tensor
           labels.append(label.squeeze().item() if isinstance(label, torch.Tensor) else label)
            
           
   elif dataset_type.upper() == "GIN":
       if dataset_name == "COLLAB":
           # Special case for COLLAB dataset to save time
           with open('/home/xxx/.dgl/GINDataset/dataset/COLLAB-GINDataset.pkl', 'rb') as f:
               dataset = pickle.load(f)
       else:
           dataset = GINDataset(name=dataset_name, self_loop=self_loop)
       graphs = dataset.graphs
       labels = [label.item() if isinstance(label, torch.Tensor) else label 
                 for label in dataset.labels]
   else:
       raise ValueError("dataset_type must be 'TU' or 'GIN'")

   labels = torch.LongTensor(labels)
   
   # Auto select normal class if not specified
   if normal_class is None:
       label_counts = torch.bincount(labels)
       max_count = label_counts.max()
       max_count_classes = torch.nonzero(label_counts == max_count).squeeze()
    #    normal_class = max_count_classes[0].item()  # Choose first class if multiple have same count
    #    print(f"Auto selected class {normal_class} as normal class (count: {max_count})")
       if max_count_classes.dim() == 0:  # Only one max class
           normal_class = max_count_classes.item()
       else:  # Multiple classes have same count
           normal_class = max_count_classes[0].item()
       print(f"Auto selected class {normal_class} as normal class (count: {max_count})")
        
   # Handle multi-class datasets
   unique_labels = torch.unique(labels)
   if len(unique_labels) > 2:
       # Convert to binary: normal_class vs rest
       new_labels = torch.where(labels == normal_class, 0, 1)
   else:
       # Binary dataset - ensure normal_class maps to 0
       if normal_class == 1:
           new_labels = 1 - labels
       else:
           new_labels = labels.clone()
   
   # Split indices
   normal_indices = torch.nonzero(new_labels == 0, as_tuple=True)[0]
   anomaly_indices = torch.nonzero(new_labels == 1, as_tuple=True)[0]
   
   # Random split of normal data
   normal_indices = normal_indices[torch.randperm(len(normal_indices))]
   split_point = int(train_ratio * len(normal_indices))
   train_indices = normal_indices[:split_point]
   
   # Handle mixed anomalies in training set
   if mixed_anomaly_ratio > 0:
       num_normal = len(train_indices)
       num_mixed_anomalies = int(num_normal * mixed_anomaly_ratio / (1 - mixed_anomaly_ratio))
       mixed_anomaly_indices = anomaly_indices[torch.randperm(len(anomaly_indices))[:num_mixed_anomalies]]
       train_indices = torch.cat([train_indices, mixed_anomaly_indices])
       remaining_anomalies = anomaly_indices[~torch.isin(anomaly_indices, mixed_anomaly_indices)]
   else:
       remaining_anomalies = anomaly_indices
       
   # Test set: remaining normal + remaining anomalies
   test_indices = torch.cat([normal_indices[split_point:], remaining_anomalies])
   
   # Create datasets
   full_dataset = CustomDataset(graphs, new_labels)
   train_dataset = CustomDataset([graphs[i] for i in train_indices], new_labels[train_indices])
   test_dataset = CustomDataset([graphs[i] for i in test_indices], new_labels[test_indices])
   
   
   # Print dataset statistics
   normal_count = torch.sum(new_labels == 0).item()
   anomaly_count = torch.sum(new_labels == 1).item()
   print(f"\nDataset Statistics:")
   print(f"Dataset: {dataset_name}")
   print(f"Total graphs: {len(full_dataset)} ({normal_count} normal, {anomaly_count} anomaly)")
   print(f"Training graphs: {len(train_dataset)} ({len(normal_indices[:split_point])} normal, {len(train_indices) - len(normal_indices[:split_point])} anomaly)")
   print(f"Testing graphs: {len(test_dataset)} ({len(normal_indices[split_point:])} normal, {len(remaining_anomalies)} anomaly)")
   print(f"Anomaly ratio in full dataset: {anomaly_count/(normal_count + anomaly_count):.2%}")
   if mixed_anomaly_ratio > 0:
       print(f"Mixed anomaly ratio in training: {(len(train_indices) - len(normal_indices[:split_point]))/len(train_indices):.2%}")

   if not return_loader:
       return train_dataset, test_dataset, full_dataset
    
   # Set batch sizes
   train_batch_size = batch_size if batch_size is not None else len(train_dataset)
   test_batch_size = len(test_dataset)
   full_batch_size = len(full_dataset)

   # Create data loaders
   train_loader = GraphDataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
   test_loader = GraphDataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
   train_loader_noshuffle = GraphDataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
   full_loader = GraphDataLoader(full_dataset, batch_size=full_batch_size, shuffle=False)
   
#    # Print dataset statistics
#    normal_count = torch.sum(new_labels == 0).item()
#    anomaly_count = torch.sum(new_labels == 1).item()
#    print(f"\nDataset Statistics:")
#    print(f"Dataset: {dataset_name}")
#    print(f"Total graphs: {len(full_dataset)} ({normal_count} normal, {anomaly_count} anomaly)")
#    print(f"Training graphs: {len(train_dataset)} ({len(normal_indices[:split_point])} normal, {len(train_indices) - len(normal_indices[:split_point])} anomaly)")
#    print(f"Testing graphs: {len(test_dataset)} ({len(normal_indices[split_point:])} normal, {len(remaining_anomalies)} anomaly)")
#    print(f"Anomaly ratio in full dataset: {anomaly_count/(normal_count + anomaly_count):.2%}")
#    if mixed_anomaly_ratio > 0:
#        print(f"Mixed anomaly ratio in training: {(len(train_indices) - len(normal_indices[:split_point]))/len(train_indices):.2%}")
   
   return train_loader, test_loader, train_loader_noshuffle, full_loader

def test_all_datasets():
    # Define available datasets
    gin_datasets = ['MUTAG', 'PROTEINS', 'IMDBBINARY', 'NCI1', 'COLLAB']
    tu_only_datasets = ['DD', 'ENZYMES', 'BZR', 'COX2', 'AIDS', 'DHFR']
    
    results = {}
    
    def process_dataset(dataset_name, dataset_type):
        try:
            print(f"\nProcessing {dataset_name} from {dataset_type}Dataset...")
            start_time = time.time()
            loaders = load_graph_dataset(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                normal_class=None,
                train_ratio=0.8,
                mixed_anomaly_ratio=0.0,
                self_loop=True,
                return_loader=False
            )
            end_time = time.time()
            process_time = end_time - start_time
            return f"Success (Time: {process_time:.2f}s)"
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            return f"Failed: {str(e)}"

    print("Testing GINDataset available datasets...")
    for dataset_name in gin_datasets:
        results[dataset_name] = ("GIN", process_dataset(dataset_name, "GIN"))
    
    print("\nTesting TUDataset only datasets...")
    for dataset_name in tu_only_datasets:
        results[dataset_name] = ("TU", process_dataset(dataset_name, "TU"))
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Dataset':15} | {'Source':6} | {'Status':50}")
    print("-" * 80)
    for dataset, (source, status) in results.items():
        status_str = status if len(status) < 47 else status[:44] + "..."
        print(f"{dataset:15} | {source:6} | {status_str:50}")
    print("-" * 80)

# Run the test
if __name__ == "__main__":
    import time
    import traceback
    test_all_datasets()
    