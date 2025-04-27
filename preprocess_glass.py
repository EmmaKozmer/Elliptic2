import pandas as pd
import torch
import random
import time
import pickle

# Parameters
DATAPATH = "./dataset/"
train = 0.8  # percentage of training subgraph
val = 0.1    # percentage of validation subgraph
chunk_size = 50_000

# Read in background_nodes.csv in chunks (to avoid memory issues)
print("Start loading background_nodes.csv in chunks...")
start = time.time()

n2id = {}
maxid = 0
row_idx = 0

# Iterate through the CSV file in chunks (to avoid memory issues)
for chunk in pd.read_csv(DATAPATH + "background_nodes.csv", usecols=["clId"], chunksize=chunk_size):
    for cl_id in chunk["clId"]:
        n2id[cl_id] = row_idx
        maxid = max(maxid, row_idx)
        row_idx += 1

print("Finished storing all nodes.")
print("Max ID:", maxid)
print("Total nodes:", len(n2id))
print("Time to load background_nodes.csv:", time.time() - start)

# Save n2id mapping
with open('n2id.pkl', 'wb') as fp:
    pickle.dump(n2id, fp)

# Process background_edges.csv in chunks (to avoid memory issues)
print("Start loading background_edges.csv in chunks...")
start = time.time()
total_edges = 0

with open("./edge_list.txt", "w") as file:
    for chunk in pd.read_csv(DATAPATH + "background_edges.csv", usecols=["clId1", "clId2"], chunksize=chunk_size):
        for t in chunk.itertuples(index=False):
            c1, c2 = t
            try:
                file.write(f"{n2id[c1]} {n2id[c2]}\n")
                total_edges += 1
            except KeyError as e:
                print(f"WARNING NODE NOT FOUND IN N2ID: {e}")

print(f"Total edges processed: {total_edges}")
print("Time to store edgelist:", time.time() - start)


# Load Subgraph Components
start = time.time()
cc = pd.read_csv(DATAPATH + "connected_components.csv")
edge = pd.read_csv(DATAPATH + "edges.csv")
node = pd.read_csv(DATAPATH + "nodes.csv")
print("Time to load subgraph data:", time.time() - start)

# Subgraph ID mapping
cc2id = {row[1]: int(row[0]) for row in cc.itertuples(index=True)}
print("Number of subgraphs:", len(cc2id))

# Build subgraph membership
sub = {}
for row in node.itertuples(index=False):
    try:
        sub_id = cc2id[row[1]]
        node_id = str(n2id[row[0]])
        if sub_id in sub:
            sub[sub_id] += "-" + node_id
        else:
            sub[sub_id] = node_id
    except KeyError:
        print(f"Missing node ID in n2id or cc2id for row: {row}")


# Generate subgraphs.pth file 
file = open("./subgraphs.pth", "w")
counter = 0
for i in sub.keys():
    counter += 1
    label = cc.loc[i, "ccLabel"]
    if counter % 10 <= 7:
        split = "train"
    elif counter % 10 == 8:
        split = "val"
    else:
        split = "test"
    file.write(f"{sub[i]}\t{label}\t{split}\n")
file.close()
print("Time to generate subgraph.pth:", time.time() - start)
