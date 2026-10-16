
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import dgl
import random


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def visualize_single_graph(
    g,
    ax=None,
    pos=None,
    node_color_mode='first_attr',
    fontproperties=None,
    node_size=200,
    edge_cmap=plt.cm.Blues,
    with_labels=False,
    layout_seed=0
):
    """

    """
    

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.get_figure()
    

    edge_attrs_list = []
    if 'weight' in g.edata:
        edge_attrs_list = ['weight']
    

    nx_g = g.to_networkx(edge_attrs=edge_attrs_list).to_undirected()

    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    

    if pos is None:
        pos = nx.spring_layout(nx_g, seed=layout_seed)
    

    if 'attr' in g.ndata:
        if node_color_mode == 'first_attr':
            node_colors = g.ndata['attr'][:, 0].cpu().numpy()
        elif node_color_mode == 'mean_attr':
            node_colors = g.ndata['attr'].mean(dim=1).cpu().numpy()
        else:

            node_colors = 'gray'
    else:
        node_colors = 'gray'
    

    nx.draw_networkx_nodes(
        nx_g, pos=pos, ax=ax,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=node_size
    )
    

    if 'weight' in edge_attrs_list:

        edge_weights = nx.get_edge_attributes(nx_g, 'weight')  # dict
        if edge_weights:
            edges, w_vals = zip(*edge_weights.items())
            w_floats = [float(w) for w in w_vals]  #
            
            edge_coll = nx.draw_networkx_edges(
                nx_g, pos=pos, ax=ax,
                edgelist=edges,
                edge_color='black',
                edge_cmap=edge_cmap,
                edge_vmin=0.0, edge_vmax=1.0,
                width=2.0
            )
        else:

            nx.draw_networkx_edges(
                nx_g, pos=pos, ax=ax,
                edge_color='black',
                width=2.0,
                node_size=node_size
            )
    else:

        nx.draw_networkx_edges(
            nx_g, pos=pos, ax=ax,
            edge_color='black',
            width=2.0,
            node_size=node_size
        )

    if with_labels:
        nx.draw_networkx_labels(nx_g, pos=pos, ax=ax,
                                font_color='black', font_size=10)
    
    ax.axis('off')
    return fig, ax