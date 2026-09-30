import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected


# loading graph
data = torch.load(
    "data/processed/gnn_graph.pt",
    weights_only=False
)

print(data)


# sorting transactions by time
time_order = torch.argsort(data.edge_timestamp)

edge_index = data.edge_index[:, time_order]
edge_features = data.edge_attr[time_order]
labels = data.y[time_order]


# 64/16/20 split
total_edges = edge_index.shape[1]

train_size = int(total_edges * 0.64)
validation_size = int(total_edges * 0.16)

train_end = train_size
validation_end = train_size + validation_size

train_edge_index = edge_index[:, :train_end]
validation_edge_index = edge_index[:, train_end:validation_end]
test_edge_index = edge_index[:, validation_end:]

train_edge_features = edge_features[:train_end]
validation_edge_features = edge_features[train_end:validation_end]
test_edge_features = edge_features[validation_end:]

y_train = labels[:train_end]
y_validation = labels[train_end:validation_end]
y_test = labels[validation_end:]


print(f"Total transactions: {total_edges:,}")
print(f"Training transactions: {train_size:,}")
print(f"Validation transactions: {validation_size:,}")
print(f"Testing transactions: {total_edges - validation_end:,}")
print(f"Training fraud rate: {y_train.float().mean():.3f}")
print(f"Validation fraud rate: {y_validation.float().mean():.3f}")
print(f"Testing fraud rate: {y_test.float().mean():.3f}")


# only training transactions are used for message passing
message_edge_index = to_undirected(train_edge_index)


def build_train_node_features(edge_index, edge_features, num_nodes):

    node_features = torch.zeros(
        (num_nodes, 8),
        dtype=torch.float
    )

    source = edge_index[0]
    destination = edge_index[1]

    amounts = edge_features[:, 0]

    # transaction counts and total amounts
    for i in range(edge_index.shape[1]):

        sender = source[i]
        receiver = destination[i]
        amount = amounts[i]

        node_features[receiver, 0] += 1
        node_features[sender, 1] += 1

        node_features[receiver, 2] += amount
        node_features[sender, 3] += amount

    incoming_count = node_features[:, 0]
    outgoing_count = node_features[:, 1]

    # average amounts
    node_features[:, 4] = (
        node_features[:, 3] /
        outgoing_count.clamp(min=1)
    )

    node_features[:, 5] = (
        node_features[:, 2] /
        incoming_count.clamp(min=1)
    )

    # unique counterparties
    outgoing_wallets = {}
    incoming_wallets = {}

    for i in range(edge_index.shape[1]):

        sender = int(source[i])
        receiver = int(destination[i])

        if sender not in outgoing_wallets:
            outgoing_wallets[sender] = set()

        if receiver not in incoming_wallets:
            incoming_wallets[receiver] = set()

        outgoing_wallets[sender].add(receiver)
        incoming_wallets[receiver].add(sender)

    for wallet in outgoing_wallets:
        node_features[wallet, 6] = len(
            outgoing_wallets[wallet]
        )

    for wallet in incoming_wallets:
        node_features[wallet, 7] = len(
            incoming_wallets[wallet]
        )

    return node_features


# build wallet features using training transactions only
train_node_features = build_train_node_features(
    train_edge_index,
    train_edge_features,
    data.x.shape[0]
)

print(
    f"Training node features: "
    f"{train_node_features.shape}"
)


# normalize node features
feature_mean = train_node_features.mean(dim=0)
feature_std = train_node_features.std(dim=0)

feature_std[feature_std == 0] = 1

train_node_features = (
    train_node_features - feature_mean
) / feature_std


class FraudGNN(nn.Module):

    def __init__(self, node_features, edge_features, hidden_size=64):

        super().__init__()

        self.conv1 = SAGEConv(
            node_features,
            hidden_size
        )

        self.conv2 = SAGEConv(
            hidden_size,
            hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * 2 + edge_features,
                64
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def encode(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x

    def decode(self, z, edge_index, edge_features):

        source = edge_index[0]
        destination = edge_index[1]

        source_embedding = z[source]
        destination_embedding = z[destination]

        combined = torch.cat(
            [
                source_embedding,
                destination_embedding,
                edge_features
            ],
            dim=1
        )

        return self.classifier(combined).squeeze(1)


model = FraudGNN(
    node_features=train_node_features.shape[1],
    edge_features=train_edge_features.shape[1]
)


# handle class imbalance
negative = (y_train == 0).sum().item()
positive = (y_train == 1).sum().item()

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float
)

loss_function = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)


# MLflow
mlflow.set_experiment("CBDC Fraud Detection")

with mlflow.start_run(run_name="GraphSAGE_v5_8_features"):

    mlflow.log_params({
        "model": "GraphSAGE",
        "hidden_size": 64,
        "learning_rate": 0.001,
        "epochs": 50,
        "weight_decay": 1e-5,
        "train_rows": train_size,
        "validation_rows": validation_size,
        "test_rows": total_edges - validation_end,
        "node_features": 8,
        "pos_weight": float(pos_weight.item()),
    })

    model.train()

    for epoch in range(50):

        optimizer.zero_grad()

        embeddings = model.encode(
            train_node_features,
            message_edge_index
        )

        logits = model.decode(
            embeddings,
            train_edge_index,
            train_edge_features
        )

        loss = loss_function(
            logits,
            y_train.float()
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1}/50 "
                f"- Loss: {loss.item():.4f}"
            )


    # validation
    model.eval()

    with torch.no_grad():

        embeddings = model.encode(
            train_node_features,
            message_edge_index
        )

        validation_logits = model.decode(
            embeddings,
            validation_edge_index,
            validation_edge_features
        )

        validation_probabilities = torch.sigmoid(
            validation_logits
        ).cpu().numpy()

        validation_labels = y_validation.cpu().numpy()


    # find the best threshold using validation data
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in torch.arange(
        0.05,
        0.96,
        0.01
    ).tolist():

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        validation_f1 = f1_score(
            validation_labels,
            validation_predictions,
            zero_division=0
        )

        if validation_f1 > best_f1:

            best_f1 = validation_f1
            best_threshold = threshold


    print()
    print(f"Best validation threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_f1:.4f}")


    # final test evaluation
    with torch.no_grad():

        test_logits = model.decode(
            embeddings,
            test_edge_index,
            test_edge_features
        )

        probabilities = torch.sigmoid(
            test_logits
        ).cpu().numpy()


    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    test_labels = y_test.cpu().numpy()


    precision = precision_score(
        test_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        test_labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        test_labels,
        probabilities
    )

    pr_auc = average_precision_score(
        test_labels,
        probabilities
    )

    matrix = confusion_matrix(
        test_labels,
        predictions
    )


    print()
    print(f"Test Precision : {precision:.4f}")
    print(f"Test Recall    : {recall:.4f}")
    print(f"Test F1 score  : {f1:.4f}")
    print(f"Test ROC-AUC   : {roc_auc:.4f}")
    print(f"Test PR-AUC    : {pr_auc:.4f}")

    print()
    print("Confusion matrix")
    print(matrix)


    mlflow.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "final_loss": loss.item(),
        "validation_f1": best_f1,
        "decision_threshold": best_threshold,
    })


    os.makedirs("models", exist_ok=True)

    model_path = "models/gnn_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_features": train_node_features.shape[1],
            "edge_features": train_edge_features.shape[1],
            "hidden_size": 64,
            "decision_threshold": best_threshold,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        model_path
    )

    mlflow.log_artifact(model_path)

    print()
    print(f"Saved model: {model_path}")