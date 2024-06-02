import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def create_graph(genome):
    G = nx.DiGraph()

    nodes = list(genome.nodes.keys())
    for i in nodes:
        G.add_nodes_from([i], subset=0, activation=genome.nodes[i].activation)

    # Add input nodes
    for i in range(-24, 0, 1):
        G.add_nodes_from([i], subset=0, activation="Input")

    connections = list(genome.connections.keys())
    for i in connections:
        G.add_edges_from([i], weight=genome.connections[i].weight, enabled=genome.connections[i].enabled)

    return G

def set_node_subsets(G):
    # Set 'subset' of each node to max possible depth
    for i in range(-24, 0, 1):
        for path in nx.all_simple_paths(G, source=i, target=[0, 1, 2, 3]):
            layer = 0
            for j in range(0, len(path)):
                curr_layer = G.nodes[path[j]]['subset']
                if layer > curr_layer:
                    G.nodes[path[j]]['subset'] = layer
                layer += 1

    # Set output nodes 'subset' to max of all of output nodes' 'subset's
    max_subset = max(G.nodes[i]['subset'] for i in range(0, 4, 1))
    for i in range(0, 4, 1):
        G.nodes[i]['subset'] = max_subset

    # Set hidden layer nodes with 'subset' 0 to 'subset' 1
    for i in G.nodes:
        if i > 3 and G.nodes[i]['subset'] == 0:
            G.nodes[i]['subset'] = 1

def set_node_positions(G):
    pos = nx.multipartite_layout(G)

    # Adjust positions of output nodes
    pos[0][1] = -0.75
    pos[1][1] = -0.25
    pos[2][1] = 0.25
    pos[3][1] = 0.75

    # Adjust positions of intermediate nodes
    max_subset = max(G.nodes[j]['subset'] for j in G.nodes)
    for i in range(1, max_subset):
        subset_nodes = [j for j in G.nodes if G.nodes[j]['subset'] == i]
        if len(subset_nodes) == 1:
            positions = [0]
        elif len(subset_nodes) == 2:
            positions = [-0.4, 0.4]
        else:
            positions = np.linspace(-0.6, 0.6, len(subset_nodes))

        for j in range(0, len(subset_nodes)):
            pos[subset_nodes[j]][1] = positions[j]

    return pos

def draw_graph(G, pos, genome):
    G = nx.Graph(G)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw nodes
    node_colors = []
    for i in G.nodes.data():
        if i[1]['activation'] == 'Input':
            node_colors.append("#80b1d3")
        elif i[1]['activation'] == 'clamped':
            node_colors.append("#fb8072")
        elif i[1]['activation'] == 'linear':
            node_colors.append("#8dd3c7")
        elif i[1]['activation'] == 'sin':
            node_colors.append("#ffffb3")
        elif i[1]['activation'] == 'gauss':
            node_colors.append("#bebada")
        elif i[1]['activation'] == 'tanh':
            node_colors.append("#fdb462")
        elif i[1]['activation'] == 'sigmoid':
            node_colors.append("#b3de69")
        elif i[1]['activation'] == 'abs':
            node_colors.append("#fccde5")
        elif i[1]['activation'] == 'relu':
            node_colors.append("#bc80bd")
        elif i[1]['activation'] == 'softplus':
            node_colors.append("#d9d9d9")
        elif i[1]['activation'] == 'identity':
            node_colors.append("#ccebc5")

    nx.draw(G, pos=pos, width=0, node_size=180, node_color=node_colors)

    # Draw edges
    edge_colors = []
    edge_widths = []
    for i in G.edges.data():
        if i[2]['enabled'] == False:
            edge_colors.append('white')
            edge_widths.append(0)
        else:
            if i[2]['weight'] > 0.33:
                edge_colors.append('green')
                edge_widths.append(1)
            elif i[2]['weight'] < -0.33:
                edge_colors.append('red')
                edge_widths.append(1)
            else:
                edge_colors.append('grey')
                edge_widths.append(1)

    nx.draw(G, pos=pos, node_size=0, alpha=0.5, edge_color=edge_colors, width=edge_widths)

    # Legend
    input_dot = Line2D([], [], color='#80b1d3', marker='o', linestyle='None', markersize=10, label='Input')
    clamped_dot = Line2D([], [], color='#fb8072', marker='o', linestyle='None', markersize=10, label='Clamped')
    linear_dot = Line2D([], [], color='#8dd3c7', marker='o', linestyle='None', markersize=10, label='Linear')
    sin_dot = Line2D([], [], color='#ffffb3', marker='o', linestyle='None', markersize=10, label='Sin')
    gauss_dot = Line2D([], [], color='#bebada', marker='o', linestyle='None', markersize=10, label='Gauss')
    tanh_dot = Line2D([], [], color='#fdb462', marker='o', linestyle='None', markersize=10, label='Tanh')
    sigmoid_dot = Line2D([], [], color='#b3de69', marker='o', linestyle='None', markersize=10, label='Sigmoid')
    abs_dot = Line2D([], [], color='#fccde5', marker='o', linestyle='None', markersize=10, label='Abs')
    relu_dot = Line2D([], [], color='#bc80bd', marker='o', linestyle='None', markersize=10, label='Relu')
    softplus_dot = Line2D([], [], color='#d9d9d9', marker='o', linestyle='None', markersize=10, label='Softplus')
    identity_dot = Line2D([], [], color='#ccebc5', marker='o', linestyle='None', markersize=10, label='Identity')
    first_legend = ax.legend(handles=[input_dot, clamped_dot, linear_dot, sin_dot, gauss_dot, tanh_dot, sigmoid_dot, abs_dot, relu_dot, softplus_dot, identity_dot], title='Activations', loc=(0.985, 0.1))
    ax.add_artist(first_legend)

    red_line = Line2D([], [], color='red', label='-1')
    grey_line = Line2D([], [], color='grey', label='0')
    green_line = Line2D([], [], color='green', label='1')
    ax.legend(handles=[red_line, grey_line, green_line], title='Connection Weights', loc=(0.95, 0.8))

    plt.show()

def show(genome):
    G = create_graph(genome)
    set_node_subsets(G)
    pos = set_node_positions(G)
    draw_graph(G, pos, genome)
    
    
