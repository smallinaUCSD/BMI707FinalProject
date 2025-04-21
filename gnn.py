import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests
import networkx as nx
from scipy.spatial.distance import euclidean
from Bio.PDB import PDBParser, PDBList

# ---------------------------
# GNN Encoder
# ---------------------------
class ProteinGNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.readout = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.readout(x, batch)
        return self.mlp(x)

# ---------------------------
# Delta Predictor Model
# ---------------------------
class DeltaDDGPredictor(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, data_wt, data_mut):
        h_wt = self.encoder(data_wt.x, data_wt.edge_index, data_wt.batch)
        h_mut = self.encoder(data_mut.x, data_mut.edge_index, data_mut.batch)
        h_diff = h_mut - h_wt
        return self.regressor(h_diff).squeeze()

# ---------------------------
# Graph Building Helpers
# ---------------------------
def fetch_alphafold_pdb(uniprot_id, save_dir="wt_pdbs"):
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    save_path = os.path.join(save_dir, f"{uniprot_id}.pdb")
    if not os.path.exists(save_path):
        r = requests.get(url)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
        else:
            raise ValueError(f"❌ Could not fetch AlphaFold PDB for {uniprot_id}")
    return save_path

def fetch_pdb_on_demand(pdb_id, save_dir="pdbs"):
    pdbl = PDBList()
    pdb_id = pdb_id.lower()
    os.makedirs(save_dir, exist_ok=True)
    filepath = pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format='pdb')
    actual_pdb_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if not os.path.exists(actual_pdb_path):
        os.rename(filepath, actual_pdb_path)
    return actual_pdb_path

def extract_residue_coords(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    coords = {}
    for residue in model[chain_id]:
        if 'CA' in residue:
            res_id = residue.get_id()[1]
            coords[res_id] = residue['CA'].get_coord()
    return coords

def build_protein_graph(coords, distance_threshold=8.0):
    G = nx.Graph()
    residue_ids = list(coords.keys())
    for res_id in residue_ids:
        G.add_node(res_id, pos=coords[res_id])
    for i, id1 in enumerate(residue_ids):
        for id2 in residue_ids[i+1:]:
            dist = euclidean(coords[id1], coords[id2])
            if dist <= distance_threshold:
                G.add_edge(id1, id2, distance=dist)
    return G

def construct_graph_from_pdb(pdb_file, chain_id, distance_threshold=8.0):
    coords = extract_residue_coords(pdb_file, chain_id)
    return build_protein_graph(coords, distance_threshold)

def one_hot_encode_aa(aa):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    return [int(aa == x) for x in aa_list]

def enrich_graph_with_features(graph, row, sequence_length, mutate=False):
    sequence = row['sequence']
    catalytic = row['is_in_catalytic_pocket']
    essential = row['is_essential']
    mutation_pos = int(row['position'])
    mutant_aa = row['mutation']
    pdb_positions = sorted(graph.nodes)
    offset = min(pdb_positions)

    for pos in graph.nodes:
        seq_index = pos - offset
        if 0 <= seq_index < len(sequence):
            aa = sequence[seq_index]
            if mutate and pos == mutation_pos:
                aa = mutant_aa
            graph.nodes[pos]['aa_type'] = one_hot_encode_aa(aa)
            graph.nodes[pos]['is_catalytic'] = int(catalytic)
            graph.nodes[pos]['is_essential'] = int(essential)
            graph.nodes[pos]['relative_pos'] = pos / sequence_length
    return graph

def convert_nx_to_pyg(graph, ddG_value=0.0):
    features = []
    node_id_map = {}
    filtered_nodes = []
    for i, n in enumerate(graph.nodes):
        node_data = graph.nodes[n]
        if all(k in node_data for k in ['aa_type', 'is_catalytic', 'is_essential', 'relative_pos']):
            node_id_map[n] = len(filtered_nodes)
            filtered_nodes.append(n)
            feat = node_data['aa_type'] + [
                node_data['is_catalytic'],
                node_data['is_essential'],
                node_data['relative_pos']
            ]
            features.append(feat)
    if len(features) == 0:
        raise ValueError("No usable nodes with complete features.")
    edge_index, edge_attr = [], []
    for u, v, attrs in graph.edges(data=True):
        if u in node_id_map and v in node_id_map:
            edge_index.append([node_id_map[u], node_id_map[v]])
            edge_index.append([node_id_map[v], node_id_map[u]])
            edge_attr.append([attrs['distance']])
            edge_attr.append([attrs['distance']])
    if len(edge_index) == 0:
        raise ValueError("No valid edges in graph.")
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([ddG_value], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# ---------------------------
# Dataset Construction
# ---------------------------
class DeltaDDGDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.pairs = []
        for _, row in df.iterrows():
            try:
                pdb_mut = fetch_pdb_on_demand(str(row['pdb_id']).split('|')[0].lower())
                pdb_wt = fetch_alphafold_pdb(row['uniprot_id'])

                G_mut = construct_graph_from_pdb(pdb_mut, row['chain'])
                G_mut = enrich_graph_with_features(G_mut, row, len(row['sequence']), mutate=True)
                data_mut = convert_nx_to_pyg(G_mut, row['ddG'])
                data_mut.batch = torch.zeros(data_mut.num_nodes, dtype=torch.long)

                G_wt = construct_graph_from_pdb(pdb_wt, 'A')
                G_wt = enrich_graph_with_features(G_wt, row, len(row['sequence']), mutate=False)
                data_wt = convert_nx_to_pyg(G_wt, row['ddG'])
                data_wt.batch = torch.zeros(data_wt.num_nodes, dtype=torch.long)

                self.pairs.append((data_wt, data_mut, torch.tensor(row['ddG'], dtype=torch.float)))
            except Exception as e:
                print(f"Skipping row due to error: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# ---------------------------
# Train Loop
# ---------------------------
def train(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0
    for data_wt, data_mut, ddg in tqdm(dataloader):
        data_wt, data_mut, ddg = data_wt.to(device), data_mut.to(device), ddg.to(device)
        optimizer.zero_grad()
        pred = model(data_wt, data_mut)
        loss = criterion(pred, ddg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ddg.size(0)
    return total_loss / len(dataloader.dataset)

# ---------------------------
# Execute Training
# ---------------------------
if __name__ == '__main__':
    df = pd.read_csv("data/fireprotdb_results.csv")
    columns_to_keep = [
        'experiment_id', 'protein_name', 'uniprot_id', 'pdb_id', 'chain',
        'position', 'wild_type', 'mutation', 'ddG', 'sequence',
        'is_in_catalytic_pocket', 'is_essential'
    ]
    df = df[columns_to_keep].dropna().drop_duplicates(subset=['experiment_id'])

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = DeltaDDGDataset(df_train)
    val_dataset = DeltaDDGDataset(df_val)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ProteinGNNEncoder(in_dim=23, hidden_dim=64, out_dim=128)
    model = DeltaDDGPredictor(encoder, embed_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(3):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
