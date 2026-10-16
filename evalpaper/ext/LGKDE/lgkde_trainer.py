"""Port of LGKDE experiment.py (Trainer.init_components / train_epoch / evaluate) on top of lgkde_pyg.py.

Copied from src/experiment.py @1bba5ef; every deviation is marked "# CHANGE" (list in NOTES.md).
EarlyStopping is extracted verbatim from src/experiment.py with `ast` (the module itself imports DGL).
"""
import ast
import os.path as osp

import numpy as np
import torch

import lgkde_pyg as P

HERE = osp.dirname(osp.abspath(__file__))
SRC = osp.join(HERE, "src")

_losses = P._load_src("losses")
_evaluation = P._load_src("evaluation")
LogRatioLossAD, AnomalyDetectionLoss = _losses.LogRatioLossAD, _losses.AnomalyDetectionLoss
calculate_fpr95 = _evaluation.calculate_fpr95


def _extract(path, names):
    tree = ast.parse(open(path).read())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), [n.name for n in nodes]
    ns = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


EarlyStopping = _extract(osp.join(SRC, "experiment.py"), ["EarlyStopping"])["EarlyStopping"]
set_global_seed = _extract(osp.join(SRC, "utils.py"), ["set_global_seed"])["set_global_seed"]
# set_global_seed needs `random` in its namespace
import random as _random
set_global_seed.__globals__["random"] = _random


def build_graph(g):
    """CHANGE: dataset graph -> LGKDE graph: dgl.add_self_loop (config self_loop: true), attr = x."""
    return dict(x=g["x"], edge_index=P.add_self_loop(g["edge_index"], g["n"]))


class Trainer:
    """CHANGE: constructed from in-memory graph lists (UB-GOLD folds) instead of load_graph_dataset();
    no SummaryWriter / plots / saved best-test model."""

    def __init__(self, config, train_graphs, test_graphs, test_labels, device, seed):
        self.config = config
        self.device = device
        # CHANGE: dataset split comes from UB-GOLD; seed = UB-GOLD trial seed (config base.seed = 42)
        set_global_seed(seed)
        self.setup_experiment(train_graphs, test_graphs, test_labels)
        self.init_components()

    def setup_experiment(self, train_graphs, test_graphs, test_labels):
        # load_graph_dataset(): training set = normal graphs (label 0); loaders as in dataprocessing.py
        self.train_graphs = []
        for i, g in enumerate(train_graphs):
            gg = build_graph(g)
            gg["gidx"] = i                              # CHANGE: id for the structure cache
            self.train_graphs.append(gg)
        self.test_graphs = []
        for i, g in enumerate(test_graphs):
            gg = build_graph(g)
            gg["gidx"] = -1 - i
            self.test_graphs.append(gg)
        new_labels_train = torch.zeros(len(self.train_graphs), dtype=torch.long)
        new_labels_test = torch.as_tensor(np.asarray(test_labels), dtype=torch.long)
        train_dataset = P.CustomDataset(self.train_graphs, new_labels_train)
        test_dataset = P.CustomDataset(self.test_graphs, new_labels_test)
        batch_size = self.config['dataset']['batch_size']
        train_batch_size = batch_size if batch_size is not None else len(train_dataset)
        test_batch_size = len(test_dataset)
        DL = torch.utils.data.DataLoader      # CHANGE: GraphDataLoader (a torch DataLoader + dgl.batch)
        self.train_loader = DL(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=P.collate)
        self.test_loader = DL(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=P.collate)
        self.train_loader_noshuffle = DL(train_dataset, batch_size=len(train_dataset), shuffle=False,
                                         collate_fn=P.collate)
        # full_loader is never iterated by the Trainer -> not built

    def infer_input_dim(self):
        first_graph, _ = self.train_loader.dataset[0]
        return first_graph["x"].shape[1]

    def init_components(self):
        in_dim = self.config['model'].get('in_dim')
        if in_dim is None:
            in_dim = self.infer_input_dim()

        self.model = P.LGKDE(
            in_dim=in_dim,
            hidden_dim=self.config['model']['hidden_dim'],
            out_dim=self.config['model']['out_dim'],
            num_layers=self.config['model']['num_layers'],
            bandwidths=self.config['model']['bandwidths'],
            dropout=self.config['model']['dropout'],
            batch_norm=self.config['model']['batch_norm'],
            learn_kde_weights=self.config['model']['learn_kde_weights']
        ).to(self.device)

        self.neg_gen = P.NegativeSampleGenerator(**self.config['negative_sampling'])

        if self.config['training']['loss_type'] == "LogRatioLossAD":
            self.criterion = LogRatioLossAD(epsilon=self.config['training']['epsilon'])
        else:
            self.criterion = AnomalyDetectionLoss(epsilon=self.config['training']['epsilon'])

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        self.train_losses = []
        early_stop_config = self.config.get(
            'early_stopping',
            self.config['training'].get('early_stopping', {})
        )
        # CHANGE: kept only to document the own-rule parameters; the run never stops early (the own
        # ORACLE replays EarlyStopping on the recorded trajectory, see derive.py)
        self.early_stopping_params = dict(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            min_epochs=early_stop_config.get('min_epochs', 5))

    def _unbatch_cpu(self, batched_graph):
        """CHANGE: graphs_cpu = batched_graph.to('cpu'); dgl.unbatch(graphs_cpu).
        On the authors' CUDA device .to('cpu') COPIES the features; on a CPU device DGL shares the
        storage and the in-place feature swap of ng.py would also overwrite the normal batch used below.
        We always copy (CUDA semantics). Per-graph features are VIEWS of the copy (as dgl.unbatch), so
        the swaps of the 'sub' pass carry over to the 'add' pass exactly as in the original."""
        x_cpu = batched_graph.x.detach().to('cpu').clone()
        ptr = batched_graph.ptr.tolist()
        gidx = batched_graph.gidx.tolist()
        out = []
        for i in range(len(ptr) - 1):
            g = self.train_graphs[gidx[i]] if gidx[i] >= 0 else self.test_graphs[-1 - gidx[i]]
            out.append(dict(x=x_cpu[ptr[i]:ptr[i + 1]], edge_index=g["edge_index"], gidx=gidx[i]))
        return out

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []

        for batched_graph, _ in self.train_loader:
            batched_graph = batched_graph.to(self.device)
            self.optimizer.zero_grad()

            train_distance_matrix = self.model.dgmmd.compute_distance_matrix(batched_graph, None)
            P_normal = self.model.compute_kde_scores(train_distance_matrix)

            graphs_list = self._unbatch_cpu(batched_graph)
            neg_graphs_sub = [self.neg_gen.generate_negative_graph(g, increase_edges=False) for g in graphs_list]
            neg_graphs_add = [self.neg_gen.generate_negative_graph(g, increase_edges=True) for g in graphs_list]

            neg_sub_batched = P.batch_graphs([x for x, _ in neg_graphs_sub], [e for _, e in neg_graphs_sub]).to(self.device)
            neg_add_batched = P.batch_graphs([x for x, _ in neg_graphs_add], [e for _, e in neg_graphs_add]).to(self.device)

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

    def evaluate(self, extra_queries=(), extra_ctx=None):
        """Returns (test density scores, [densities of each extra query batch]).
        CHANGE: the reference self-scores (get_reference_scores, used only for the F1 threshold) are not
        computed; extra_queries (held-out normals) are scored against the same reference set. Iterating
        the two loaders consumes the same global-RNG draws as the original evaluate()."""
        self.model.eval()
        with torch.no_grad():
            for train_batched_graph, train_labels in self.train_loader_noshuffle:
                train_batched_graph = train_batched_graph.to(self.device)

            test_scores_list = []
            for test_batched_graph, test_labels in self.test_loader:
                test_batched_graph = test_batched_graph.to(self.device)
                query_scores = self.model.get_query_scores(test_batched_graph, train_batched_graph)
                test_scores_list.append(query_scores)
            test_scores = torch.cat(test_scores_list, dim=0)
            import contextlib
            with (extra_ctx() if extra_ctx is not None else contextlib.nullcontext()):
                extra = [self.model.get_query_scores(q.to(self.device), train_batched_graph) for q in extra_queries]
        return test_scores, extra


def metrics_lgkde(y_true, density):
    """LGKDE's own metric computation (experiment.py evaluate): y_score = -density."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_score = -np.asarray(density, dtype=np.float64)
    return dict(lgkde_auroc=float(roc_auc_score(y_true, y_score)),
                lgkde_aupr=float(average_precision_score(y_true, y_score)),
                lgkde_fpr95=float(calculate_fpr95(y_true, y_score)))
