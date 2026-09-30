from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from dotenv import load_dotenv
from neo4j import GraphDatabase


BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")


# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "cbdc")


# Load trained GNN
checkpoint = torch.load(
    BASE_DIR / "models" / "gnn_model.pt",
    map_location="cpu",
    weights_only=False,
)


class FraudGNN(nn.Module):
    """
    Must match src/graph/train_gnn.py exactly:
    - 2 SAGEConv layers (encode)
    - classifier: Linear -> ReLU -> Dropout(0.2) -> Linear
      (Dropout has no weights but occupies index 2,
       which is why the saved Linear is classifier.3)
    """

    def __init__(self, node_features, edge_features, hidden_size=64):
        super().__init__()

        self.conv1 = SAGEConv(
            node_features,
            hidden_size,
        )

        self.conv2 = SAGEConv(
            hidden_size,
            hidden_size,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + edge_features, 64),   # classifier.0
            nn.ReLU(),                                         # classifier.1
            nn.Dropout(0.2),                                   # classifier.2 (no weights)
            nn.Linear(64, 1),                                  # classifier.3
        )

    def encode(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x

    def decode(self, z, edge_index, edge_attr):

        source = z[edge_index[0]]
        target = z[edge_index[1]]

        edge_input = torch.cat(
            [source, target, edge_attr],
            dim=1,
        )

        return self.classifier(edge_input).squeeze(-1)


NODE_FEATURES = 8
EDGE_FEATURES = 5


model = FraudGNN(
    node_features=NODE_FEATURES,
    edge_features=EDGE_FEATURES,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


decision_threshold = checkpoint["decision_threshold"]
feature_mean = checkpoint["feature_mean"]
feature_std = checkpoint["feature_std"]


print("GNN model loaded")
print("Decision threshold:", decision_threshold)
print("Node features:", NODE_FEATURES)
print("Edge features:", EDGE_FEATURES)


# Connect to Neo4j
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)


def get_wallet_graph(sender_id, receiver_id):

    with driver.session(database=NEO4J_DATABASE) as session:

        result = session.run(
            """
            MATCH
                (sender:Wallet)
                -[t:TRANSFER]->
                (receiver:Wallet)

            WHERE
                sender.wallet_id = $sender_id
                OR receiver.wallet_id = $sender_id
                OR sender.wallet_id = $receiver_id
                OR receiver.wallet_id = $receiver_id

            RETURN
                sender.wallet_id AS sender_id,
                receiver.wallet_id AS receiver_id,

                t.amount_eur AS amount_eur,
                t.transactions_1h AS transactions_1h,
                t.transactions_24h AS transactions_24h,
                t.sender_velocity AS sender_velocity,
                t.receiver_velocity AS receiver_velocity,

                t.transaction_id AS transaction_id
            """,
            sender_id=sender_id,
            receiver_id=receiver_id,
        )

        return [record.data() for record in result]


def build_graph(history, transaction):

    wallets = set()

    for row in history:
        wallets.add(row["sender_id"])
        wallets.add(row["receiver_id"])

    wallets.add(transaction["sender_id"])
    wallets.add(transaction["receiver_id"])

    wallets = list(wallets)

    wallet_to_index = {
        wallet: index
        for index, wallet in enumerate(wallets)
    }

    # Wallet statistics
    incoming_count = {wallet: 0 for wallet in wallets}
    outgoing_count = {wallet: 0 for wallet in wallets}
    total_received = {wallet: 0.0 for wallet in wallets}
    total_sent = {wallet: 0.0 for wallet in wallets}
    sent_amounts = {wallet: [] for wallet in wallets}
    received_amounts = {wallet: [] for wallet in wallets}
    sent_to = {wallet: set() for wallet in wallets}
    received_from = {wallet: set() for wallet in wallets}

    # Build wallet statistics from history
    for row in history:

        sender = row["sender_id"]
        receiver = row["receiver_id"]
        amount = float(row["amount_eur"] or 0)

        outgoing_count[sender] += 1
        incoming_count[receiver] += 1

        total_sent[sender] += amount
        total_received[receiver] += amount

        sent_amounts[sender].append(amount)
        received_amounts[receiver].append(amount)

        sent_to[sender].add(receiver)
        received_from[receiver].add(sender)

    # Create node features (must match the 8 features used in train_gnn.py:
    # incoming_count, outgoing_count, total_received, total_sent,
    # avg_sent, avg_received, unique_sent_to, unique_received_from)
    node_features = []

    for wallet in wallets:

        if sent_amounts[wallet]:
            avg_sent = sum(sent_amounts[wallet]) / len(sent_amounts[wallet])
        else:
            avg_sent = 0.0

        if received_amounts[wallet]:
            avg_received = sum(received_amounts[wallet]) / len(received_amounts[wallet])
        else:
            avg_received = 0.0

        features = [
            incoming_count[wallet],
            outgoing_count[wallet],
            total_received[wallet],
            total_sent[wallet],
            avg_sent,
            avg_received,
            len(sent_to[wallet]),
            len(received_from[wallet]),
        ]

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float32)

    # Normalize using training statistics
    x = (x - feature_mean) / (feature_std + 1e-8)

    # Build historical edges
    edge_list = []
    edge_features = []

    for row in history:

        sender_index = wallet_to_index[row["sender_id"]]
        receiver_index = wallet_to_index[row["receiver_id"]]

        edge_list.append([sender_index, receiver_index])

        edge_features.append(
            [
                float(row["amount_eur"] or 0),
                float(row["transactions_1h"] or 0),
                float(row["transactions_24h"] or 0),
                float(row["sender_velocity"] or 0),
                float(row["receiver_velocity"] or 0),
            ]
        )

    if edge_list:

        history_edge_index = torch.tensor(
            edge_list, dtype=torch.long
        ).t().contiguous()

        history_edge_attr = torch.tensor(
            edge_features, dtype=torch.float32
        )

    else:

        history_edge_index = torch.empty((2, 0), dtype=torch.long)
        history_edge_attr = torch.empty((0, EDGE_FEATURES), dtype=torch.float32)

    # Target transaction
    target_sender = wallet_to_index[transaction["sender_id"]]
    target_receiver = wallet_to_index[transaction["receiver_id"]]

    target_edge = torch.tensor(
        [[target_sender], [target_receiver]],
        dtype=torch.long,
    )

    target_edge_attr = torch.tensor(
        [
            [
                float(transaction["amount_eur"]),
                float(transaction["transactions_1h"]),
                float(transaction["transactions_24h"]),
                float(transaction["sender_velocity"]),
                float(transaction["receiver_velocity"]),
            ]
        ],
        dtype=torch.float32,
    )

    return (
        x,
        history_edge_index,
        history_edge_attr,
        target_edge,
        target_edge_attr,
    )


def predict_gnn(transaction):

    sender_id = transaction["sender_id"]
    receiver_id = transaction["receiver_id"]

    history = get_wallet_graph(sender_id, receiver_id)

    if not history:
        raise ValueError("No graph history found for the supplied wallets")

    (
        x,
        history_edge_index,
        history_edge_attr,
        target_edge,
        target_edge_attr,
    ) = build_graph(history, transaction)

    # GraphSAGE message passing (undirected, matches to_undirected() in training)
    if history_edge_index.size(1) > 0:
        reverse_edges = history_edge_index.flip(0)
        message_edges = torch.cat([history_edge_index, reverse_edges], dim=1)
    else:
        message_edges = history_edge_index

    with torch.no_grad():

        z = model.encode(x, message_edges)

        logit = model.decode(z, target_edge, target_edge_attr)

        probability = torch.sigmoid(logit).item()

    prediction = "fraud" if probability >= decision_threshold else "normal"

    return {
        "prediction": prediction,
        "fraud_probability": round(float(probability), 4),
    }


if __name__ == "__main__":

    with driver.session(database=NEO4J_DATABASE) as session:

        result = session.run(
            """
            MATCH
                (sender:Wallet)
                -[t:TRANSFER]->
                (receiver:Wallet)

            RETURN
                sender.wallet_id AS sender_id,
                receiver.wallet_id AS receiver_id,

                t.amount_eur AS amount_eur,
                t.transactions_1h AS transactions_1h,
                t.transactions_24h AS transactions_24h,
                t.sender_velocity AS sender_velocity,
                t.receiver_velocity AS receiver_velocity

            LIMIT 1
            """
        )

        record = result.single()

    if record is None:
        raise ValueError("No transactions found in Neo4j")

    test_transaction = {
        "sender_id": record["sender_id"],
        "receiver_id": record["receiver_id"],
        "amount_eur": record["amount_eur"],
        "transactions_1h": record["transactions_1h"],
        "transactions_24h": record["transactions_24h"],
        "sender_velocity": record["sender_velocity"],
        "receiver_velocity": record["receiver_velocity"],
    }

    print()
    print("Testing transaction:")
    print(test_transaction)

    result = predict_gnn(test_transaction)

    print()
    print("GNN prediction:")
    print(result)

    driver.close()