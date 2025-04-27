import pandas as pd
import torch
import pickle
import numpy as np
import os

DATAPATH = "./dataset/"

# Load node ID mapping
with open('n2id.pkl', 'rb') as handle:
    n2id = pickle.load(handle)

# Load subgraph mapping
cc = pd.read_csv(DATAPATH + "connected_components.csv")
cc2id = {int(row[1]): int(row[0]) for row in cc.itertuples(index=True)}

# Load subgraph nodes
node = pd.read_csv(DATAPATH + "nodes.csv")
sub = {}
subgraph_nodes = set()
for row in node.itertuples(index=False):
    nid = n2id[int(row[0])]
    cc_id = cc2id[int(row[1])]
    sub.setdefault(cc_id, []).append(nid)
    subgraph_nodes.add(nid)

# Only store relevant edges
print("Building adjacency list for subgraph nodes only...")
adj = {}
with open("./edge_list.txt", "r") as f:
    for line in f:
        try:
            c1, c2 = map(int, line.strip().split())
            if c1 in subgraph_nodes and c2 in subgraph_nodes:
                low, high = sorted([c1, c2])
                adj.setdefault(low, []).append(high)
        except:
            continue  # skip any malformed lines

# Prepare output
os.makedirs("./sub2vec/sub2vec_input", exist_ok=True)

# Generate subgraph files
label = {}
isolate = 0
count = 0

print("Generating subgraph files...")
for c, nodes in sub.items():
    with open(f"./sub2vec/sub2vec_input/subGraph{c}", "w") as f:
        for i in range(len(nodes)):
            has_edge = False
            for j in range(i, len(nodes)):
                n1, n2 = sorted((nodes[i], nodes[j]))
                if n1 in adj and n2 in adj[n1]:
                    f.write(f"{n1}\t{n2}\n")
                    has_edge = True
            if not has_edge:
                f.write(f"{nodes[i]}\t{nodes[i]}\n")
                isolate += 1
    label[c] = cc.loc[c, "ccLabel"]
    count += 1
    if count % 500 == 0:
        print(f"Processed {count} subgraphs...")

# Save labels
with open("label.pkl", "wb") as fp:
    pickle.dump(label, fp)

print(f"Done! {count} subgraphs processed. {isolate} had isolated nodes.")
