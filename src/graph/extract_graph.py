import os

import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from torch_geometric.data import Data

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

query = """
MATCH (sender:Wallet)-[t:TRANSFER]->(receiver:Wallet)
RETURN
    sender.wallet_id AS sender_id,
    receiver.wallet_id AS receiver_id,
    t.amount_eur AS amount_eur,
    t.transactions_1h AS transactions_1h,
    t.transactions_24h AS transactions_24h,
    t.sender_velocity AS sender_velocity,
    t.receiver_velocity AS receiver_velocity,
    t.is_fraud AS is_fraud,
    t.timestamp AS timestamp
"""

with driver.session(database=DATABASE) as session:
    records = session.run(query).data()

driver.close()

print(f"Transfers loaded: {len(records):,}")

wallet_ids = set()

for row in records:
    wallet_ids.add(row["sender_id"])
    wallet_ids.add(row["receiver_id"])

wallet_ids = sorted(wallet_ids)

wallet_to_index = {
    wallet_id: index
    for index, wallet_id in enumerate(wallet_ids)
}

wallet_features = {
    wallet_id: {
        "in_degree": 0,
        "out_degree": 0,
        "total_sent": 0.0,
        "total_received": 0.0,
        "sent_amounts": [],
        "received_amounts": []
    }
    for wallet_id in wallet_ids
}

edge_index = []
edge_features = []
labels = []
timestamps = []

for row in records:

    sender_id = row["sender_id"]
    receiver_id = row["receiver_id"]
    amount = float(row["amount_eur"])

    sender = wallet_to_index[sender_id]
    receiver = wallet_to_index[receiver_id]

    edge_index.append([sender, receiver])

    edge_features.append([
        amount,
        row["transactions_1h"],
        row["transactions_24h"],
        row["sender_velocity"],
        row["receiver_velocity"],
    ])

    labels.append(row["is_fraud"])
    timestamps.append(row["timestamp"])

    wallet_features[sender_id]["out_degree"] += 1
    wallet_features[sender_id]["total_sent"] += amount
    wallet_features[sender_id]["sent_amounts"].append(amount)

    wallet_features[receiver_id]["in_degree"] += 1
    wallet_features[receiver_id]["total_received"] += amount
    wallet_features[receiver_id]["received_amounts"].append(amount)

node_features = []

for wallet_id in wallet_ids:

    wallet = wallet_features[wallet_id]

    avg_sent = (
        sum(wallet["sent_amounts"]) / len(wallet["sent_amounts"])
        if wallet["sent_amounts"]
        else 0
    )

    avg_received = (
        sum(wallet["received_amounts"]) / len(wallet["received_amounts"])
        if wallet["received_amounts"]
        else 0
    )

    node_features.append([
        wallet["in_degree"],
        wallet["out_degree"],
        wallet["total_sent"],
        wallet["total_received"],
        avg_sent,
        avg_received,
    ])

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long
).t().contiguous()

edge_features = torch.tensor(
    edge_features,
    dtype=torch.float
)

labels = torch.tensor(
    labels,
    dtype=torch.long
)

node_features = torch.tensor(
    node_features,
    dtype=torch.float
)

# convert timestamps to numbers for temporal splitting
timestamps = torch.tensor(
    [
        torch.tensor(
            __import__("pandas").Timestamp(timestamp).timestamp()
        )
        for timestamp in timestamps
    ],
    dtype=torch.float64
)

data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_features,
    y=labels
)

data.edge_timestamp = timestamps

print(f"Wallets: {len(wallet_ids):,}")
print(f"Transfers: {data.edge_index.shape[1]:,}")
print(f"Node features: {data.x.shape[1]}")
print(f"Edge features: {data.edge_attr.shape[1]}")
print(f"Fraud transactions: {int(data.y.sum()):,}")
print("Timestamps: saved")

print(data)

os.makedirs("data/processed", exist_ok=True)

torch.save(
    data,
    "data/processed/gnn_graph.pt"
)

print("Saved GNN graph")