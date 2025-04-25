import networkx as nx
import torch
from tqdm import tqdm
import math
import numpy as np
import copy
import gc
from itertools import product

# --- Gower-Based Star Weisfeiler--Lehman Kernel ---
 
def s_numeric(x, y, range_val):
    """
    Gower numeric similarity: 1 - |x-y|/range_val.
    Returns None if either x or y is None.
    """
    if x is None or y is None:
        return None
    if range_val <= 0:
        return 1.0
    return 1 - abs(x - y) / range_val

def s_categorical(x, y):
    """
    Gower categorical similarity: 1 if equal, 0 if different.
    Returns None if either x or y is None.
    """
    if x is None or y is None:
        return None
    return 1.0 if x == y else 0.0


def s_d_prime(base_sim, gamma):
    """
    Exponential kernel transform: exp(-gamma * (1 - base_sim)).
    """
    return math.exp(-gamma * (1 - base_sim))


def attribute_similarity(attr1, attr2, attr_meta, gamma):
    """
    Compute weighted average of transformed Gower similarities over attributes,
    skipping any attribute where either value is missing.

    attr1, attr2: dict mapping attribute_name -> value
    attr_meta: dict mapping attribute_name -> {
        'type': 'numeric' or 'categorical',
        'range': float (for numeric),
        'weight': float (optional)
    }
    gamma: positive scaling parameter
    """
    total, denom = 0.0, 0.0
    for d, meta in attr_meta.items():
        x, y = attr1.get(d), attr2.get(d)
        if meta['type'] == 'numeric':
            base = s_numeric(x, y, meta['range'])
        else:
            base = s_categorical(x, y)
        if base is None:
            continue
        w = meta.get('weight', 1.0)
        total += w * s_d_prime(base, gamma)
        denom += w
    return (total / denom) if denom > 0 else 0.0


def extract_star_subgraph(G: nx.Graph, center, k=1):
    """
    Extract k-hop induced subgraph centered at node `center`.
    Returns (subgraph, center).
    """
    lengths = nx.single_source_shortest_path_length(G, center, cutoff=k)
    return G.subgraph(lengths.keys()).copy(), center


def k_s(star1: nx.Graph, center1, star2: nx.Graph, center2,
        attr_meta_nodes, attr_meta_edges, gamma, eps=1e-8):
    """
    Local kernel between two star subgraphs.
    If core-node similarity equals exp(-gamma), returns 0.
    Otherwise sums attribute similarities over all node-node and edge-edge pairs.
    """
    # Core-node similarity
    P_center = attribute_similarity(
        star1.nodes[center1], star2.nodes[center2],
        attr_meta_nodes, gamma)
    if abs(P_center - math.exp(-gamma)) < eps:
        return 0.0

    # Gather node and edge attribute dicts
    elems1 = [('node', star1.nodes[n]) for n in star1.nodes]
    elems1 += [('edge', data) for _, _, data in star1.edges(data=True)]
    elems2 = [('node', star2.nodes[n]) for n in star2.nodes]
    elems2 += [('edge', data) for _, _, data in star2.edges(data=True)]

    # Sum attribute similarities
    total = 0.0
    for (t1, a1), (t2, a2) in product(elems1, elems2):
        if t1 != t2:
            continue
        meta = attr_meta_nodes if t1 == 'node' else attr_meta_edges
        total += attribute_similarity(a1, a2, meta, gamma)
    return total


def local_graph_kernel(G: nx.Graph, H: nx.Graph,
                       attr_meta_nodes, attr_meta_edges, gamma):
    """
    1-hop star-based local graph kernel K_S.
    K_S(G,H) = sum_{v in G} sum_{u in H} k_s(S(v), S(u)).
    """
    K = 0.0
    for v in G.nodes:
        star1, _ = extract_star_subgraph(G, v, k=1)
        for u in H.nodes:
            star2, _ = extract_star_subgraph(H, u, k=1)
            K += k_s(star1, v, star2, u,
                     attr_meta_nodes, attr_meta_edges, gamma)
    return K


def swl_kernel(G: nx.Graph, H: nx.Graph,
               attr_meta_nodes, attr_meta_edges, gamma, k):
    """
    Star-based Weisfeiler–Lehman kernel of depth k.
    K_SWL^k(G,H) = sum_{v in G} sum_{u in H} k_s(S^k(v), S^k(u)).
    """
    K = 0.0
    for v in G.nodes:
        star1, _ = extract_star_subgraph(G, v, k=k)
        for u in H.nodes:
            star2, _ = extract_star_subgraph(H, u, k=k)
            K += k_s(star1, v, star2, u,
                     attr_meta_nodes, attr_meta_edges, gamma)
    return K


def perform_swl_graph_kernel_computation(G1, G2, c, H,
                                           attr_meta_nodes, attr_meta_edges):
    """
    Wrapper replacing original edge-based kernel:
      - c: gamma parameter for kernel transform
      - H: depth k for SWL
    """
    return swl_kernel(G1.g, G2.g,
                      attr_meta_nodes, attr_meta_edges,
                      gamma=c, k=H)


# --- Data Structures & Loading ---

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.g = g
        self.label = label
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0


def load_data(dataset, degree_as_tag,
              attr_meta_nodes, attr_meta_edges,
              data_dir='/home/ycy/MSSM-GNN/dataset'):
    """
    Load graphs from text files, compute shortest-path subgraphs,
    apply SWL kernel to augment edge weights.

    Returns:
      - g_list: list of S2VGraph
      - num_classes: number of unique labels
    """
    g_list = []
    label_dict = {}
    feat_dict = {}
    path = f"{data_dir}/{dataset}/{dataset}.txt"
    with open(path, 'r') as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            n, l = map(int, f.readline().split())
            if l not in label_dict:
                label_dict[l] = len(label_dict)
            g = nx.Graph()
            node_tags, node_features = [], []
            for j in range(n):
                g.add_node(j)
                parts = f.readline().split()
                deg = int(parts[1])
                tmp = deg + 2
                if tmp == len(parts):
                    parts_idx = list(map(int, parts))
                    attr = None
                else:
                    parts_idx = list(map(int, parts[:tmp]))
                    attr = np.array(list(map(float, parts[tmp:])))
                # tag mapping
                tag = parts_idx[0]
                if tag not in feat_dict:
                    feat_dict[tag] = len(feat_dict)
                node_tags.append(feat_dict[tag])
                if attr is not None:
                    node_features.append(attr)
                # edges
                for k in parts_idx[2:]:
                    g.add_edge(j, k)
            if node_features:
                node_features = np.stack(node_features)
                feature_flag = True
            else:
                node_features = None
                feature_flag = False
            g_list.append(S2VGraph(g, label_dict[l], node_tags))
    # populate neighbors and edge_mat
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i,j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        g.max_neighbor = max(len(nb) for nb in g.neighbors)
        # edges for torch
        edges = [ [i,j] for i,j in g.g.edges() ]
        edges += [[j,i] for i,j in g.g.edges()]
        g.edge_mat = torch.LongTensor(edges).t()
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    # one-hot node_features
    tagset = sorted({tag for g in g_list for tag in g.node_tags})
    tag2idx = {tag:i for i,tag in enumerate(tagset)}
    for g in g_list:
        ft = torch.zeros(len(g.node_tags), len(tagset))
        for i,tag in enumerate(g.node_tags):
            ft[i, tag2idx[tag]] = 1
        g.node_features = ft

    # generate shortest-path subgraphs for first few graphs, augment with kernel
    all_paths = []
    subset = g_list[:5]
    c = 2.0  # gamma
    H = 3    # WL depth
    # compute pairwise kernel and add as weight
    for i in range(len(subset)):
        G1 = subset[i]
        for j in range(i+1, len(subset)):
            G2 = subset[j]
            kscore = perform_swl_graph_kernel_computation(
                G1, G2, c, H,
                attr_meta_nodes, attr_meta_edges)
            # normalize
            if kscore > 0:
                norm = 3 * kscore / kscore  # trivial since only single value
            else:
                norm = 0
            w = int(norm)
            if w > 0:
                G1.g.add_edge(i, j, weight=w)
                G2.g.add_edge(j, i, weight=w)
    return g_list, len(label_dict) 

class GenGraph(object):
    def __init__(self, data, num_graphs):
        self.data = data
        self.nodes_labels = data.node_labels
        self.vocab = {}
        self.whole_node_count = {}
        self.weight_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        self.g_final = self.gen_components(num_graphs)
        self.num_cliques = self.g_final.number_of_nodes() - len(self.data.g_list)
        del self.data, self.vocab, self.whole_node_count, self.weight_vocab, self.node_count, self.edge_count
        gc.collect()
    def gen_components(self, num_graphs):
        g_list = self.data.g_list
        h_g = nx.Graph()
        for g in tqdm(range(len(g_list)), desc='Gen Components', unit='graph'):
            clique_list = []
            mcb = nx.cycle_basis(g_list[g])
            mcb_tuple = [tuple(ele) for ele in mcb]
            edges = list(g_list[g].edges())[:num_graphs]  # Only use the first num_graphs edges
            for e in edges:
                weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                edge = ((self.nodes_labels[e[0] - 1], self.nodes_labels[e[1] - 1]), weight)
                clique_id = self.add_to_vocab(edge)
                clique_list.append(clique_id)
                if clique_id not in self.whole_node_count:
                    self.whole_node_count[clique_id] = 1
                else:
                    self.whole_node_count[clique_id] += 1

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g_list[g]))
                ring = [self.nodes_labels[m[i] - 1] for i in range(len(m))]
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
                clique_list.append(cycle_id)
                if cycle_id not in self.whole_node_count:
                    self.whole_node_count[cycle_id] = 1
                else:
                    self.whole_node_count[cycle_id] += 1

            for e in clique_list:
                self.add_weight(e, g)

            c_list = tuple(set(clique_list))
            for e in c_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    self.node_count[e] += 1

            for e in c_list:
                h_g.add_edge(g, e + len(g_list), weight=(self.weight_vocab[(g, e)] / len(edges) + len(mcb_tuple)))

            for e in range(len(edges)):
                for i in range(e + 1, len(edges)):
                    for j in edges[e]:
                        if j in edges[i]:
                            weight = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge = ((self.nodes_labels[edges[e][0] - 1], self.nodes_labels[edges[e][1] - 1]), weight)
                            weight_i = g_list[g].get_edge_data(edges[i][0], edges[i][1])['weight']
                            edge_i = ((self.nodes_labels[edges[i][0] - 1], self.nodes_labels[edges[i][1] - 1]), weight_i)
                            final_edge = tuple(sorted((self.add_to_vocab(edge), self.add_to_vocab(edge_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
            for m in range(len(mcb_tuple)):
                for i in range(m + 1, len(mcb_tuple)):
                    for j in mcb_tuple[m]:
                        if j in mcb_tuple[i]:
                            weight = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring = [self.nodes_labels[mcb_tuple[m][t] - 1] for t in range(len(mcb_tuple[m]))]
                            cycle = (tuple(ring), weight)

                            weight_i = tuple(self.find_ring_weights(mcb_tuple[i], g_list[g]))
                            ring_i = [self.nodes_labels[mcb_tuple[i][t] - 1] for t in range(len(mcb_tuple[i]))]
                            cycle_i = (tuple(ring_i), weight_i)

                            final_edge = tuple(sorted((self.add_to_vocab(cycle), self.add_to_vocab(cycle_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

            for e in range(len(edges)):
                for m in range(len(mcb_tuple)):
                    for i in edges[e]:
                        if i in mcb_tuple[m]:
                            weight_e = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge_e = ((self.nodes_labels[edges[e][0] - 1], self.nodes_labels[edges[e][1] - 1]), weight_e)
                            weight_m = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring_m = [self.nodes_labels[mcb_tuple[m][t] - 1] for t in range(len(mcb_tuple[m]))]
                            cycle_m = (tuple(ring_m), weight_m)

                            final_edge = tuple(sorted((self.add_to_vocab(edge_e), self.add_to_vocab(cycle_m))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

        return h_g
    def add_to_vocab(self, clique):
            c = copy.deepcopy(clique[0])
            weight = copy.deepcopy(clique[1])
            for i in range(len(c)):
                if (c, weight) in self.vocab:
                    return self.vocab[(c, weight)]
                else:
                    c = self.shift_right(c)
                    weight = self.shift_right(weight)
            self.vocab[(c, weight)] = len(list(self.vocab.keys()))
            return self.vocab[(c, weight)]

    def add_weight(self, node_id, g):
            if (g, node_id) not in self.weight_vocab:
                self.weight_vocab[(g, node_id)] = 1
            else:
                self.weight_vocab[(g, node_id)] += 1

    def update_weight(self, g):
            for (u, v) in g.edges():
                if u < len(self.data.g_list):
                    g[u][v]['weight'] = g[u][v]['weight'] * (math.log((len(self.data.g_list) + 1) / self.node_count[v - len(self.data.g_list)]))
                else:
                    g[u][v]['weight'] = g[u][v]['weight'] * (
                        math.log((len(self.data.g_list) + 1) / self.node_count[u - len(self.data.g_list)]))
            return g

    def add_edge(self, g):
            edges = list(self.edge_count.keys())
            for i in edges:
                g.add_edge(i[0] + len(self.data.g_list), i[1] + len(self.data.g_list), weight=math.exp(math.log(self.edge_count[i] / math.sqrt(self.whole_node_count[i[0]] * self.whole_node_count[i[1]]))))
            return g

    def drop_node(self, g):
            rank_list = []
            node_list = []
            sub_node_list = []
            for v in sorted(g.nodes()):
                if v > len(self.data.g_list):
                    rank_list.append(self.node_count[v - len(self.data.g_list)] / len(self.data.g_list))
                    node_list.append(v)
            sorted_list = sorted(rank_list)
            a = int(len(sorted_list) * 0.9)
            threshold_num = sorted_list[a]
            for i in range(len(rank_list)):
                if rank_list[i] > threshold_num:
                    sub_node_list.append(node_list[i])
            self.removed_nodes = sub_node_list
            count = 0
            label_mapping = {}
            for v in sorted(g.nodes()):
                if v in sub_node_list:
                    count += 1
                else:
                    label_mapping[v] = v - count
            for v in sub_node_list:
                g.remove_node(v)
            
            g = nx.relabel_nodes(g, label_mapping)
            return g

    @staticmethod
    def shift_right(l):
            if type(l) == int:
                return l
            elif type(l) == tuple:
                l = list(l)
                return tuple([l[-1]] + l[:-1])
            elif type(l) == list:
                return tuple([l[-1]] + l[:-1])
            else:
                print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g):
            weight_list = []
            for i in range(len(ring)-1):
                weight = g.get_edge_data(ring[i], ring[i+1])['weight']
                weight_list.append(weight)
            weight = g.get_edge_data(ring[-1], ring[0])['weight']
            weight_list.append(weight)
            return weight_list

