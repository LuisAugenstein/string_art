import torch
from string_art.edges import EdgesAngleBased
import networkx as nx


def get_euler_path(strings: EdgesAngleBased) -> EdgesAngleBased:
    G = nx.Graph()
    # round angles to make values usable as node ids
    node_ids = (strings* 1e4).int().tolist()
    id_to_angle: dict[int, float] = {}
    for i in range(strings.shape[0]):
        angle_i, angle_j = strings[i].tolist()
        id_i, id_j = node_ids[i]
        id_to_angle[id_i] = angle_i
        id_to_angle[id_j] = angle_j

    G.add_edges_from(node_ids)
    odd_nodes = [id for id, deg in dict(G.degree()).items() if deg % 2 == 1]
    if len(odd_nodes) > 0:
        # add new edges to make the graph eulerian
        def dist(id_i: int, id_j: int) -> float:
            angle_i, angle_j = id_to_angle[id_i], id_to_angle[id_j]
            return (torch.exp(torch.tensor(angle_i*1j)) - torch.exp(torch.tensor(angle_j*1j))).norm().item()

        odd_graph = nx.Graph()
        for i in range(len(odd_nodes)):
            for j in range(i + 1, len(odd_nodes)):
                id_i, id_j = odd_nodes[i], odd_nodes[j]
                odd_graph.add_edge(id_i, id_j, weight=dist(id_i, id_j))

        matching = list(nx.min_weight_matching(odd_graph))
        max_length_matching_index = torch.tensor([odd_graph[u][v]['weight'] for u, v in matching]).argmax().item()
        matching.pop(max_length_matching_index)
        for id_i, id_j in matching:
            G.add_edge(id_i, id_j)

    eulerian_path_id_based = list(nx.eulerian_path(G))
    eulerian_path_angle_based = [[id_to_angle[id_i], id_to_angle[id_j]] for id_i, id_j in eulerian_path_id_based]
    return  torch.tensor(eulerian_path_angle_based)