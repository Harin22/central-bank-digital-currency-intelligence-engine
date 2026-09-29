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


#loading graph
data = torch.load(
    "data/processed/gnn_graph.pt",
    weights_only=False
)

print(data)

#sorting transactions by time
time_order = torch.argsort(data.edge_timestamp)

edge_index = data.edge_index[:, time_order]
edge_features = data.edge_attr[time_order]
labels = data.y[time_order]

#80/20 split
total_edges = edge_index.shape[1]
train_size = int(total_edges * 0.8)

train_edge_index = edge_index[:, :train_size]
test_edge_index = edge_index[:, train_size:]

train_edge_features = edge_features[:train_size]
test_edge_features = edge_features[train_size:]

y_train = labels[:train_size]
y_test = labels[train_size:]

print(f"Total transactions: {total_edges:,}")
print(f"Training transactions: {train_size:,}")
print(f"Testing transactions : {total_edges - train_size:,}")
print(f"Training fraud rate: {y_train.float().mean():.3f}")
print(f"Testing fraud rate : {y_test.float().mean():.3f}")

message_edge_index = to_undirected(train_edge_index)


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
    node_features=data.x.shape[1],
    edge_features=data.edge_attr.shape[1]
)

#to handle class imbalance
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

with mlflow.start_run(run_name="GraphSAGE"):

    mlflow.log_params({
        "model": "GraphSAGE",
        "hidden_size": 64,
        "learning_rate": 0.001,
        "epochs": 50,
        "weight_decay": 1e-5,
        "train_rows": train_size,
        "test_rows": total_edges - train_size,
        "pos_weight": float(pos_weight.item()),
    })

    model.train()

    for epoch in range(50):

        optimizer.zero_grad()

        embeddings = model.encode(
            data.x,
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

    # evaluation
    model.eval()

    with torch.no_grad():

        embeddings = model.encode(
            data.x,
            message_edge_index
        )

        test_logits = model.decode(
            embeddings,
            test_edge_index,
            test_edge_features
        )

        probabilities = torch.sigmoid(
            test_logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print()
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

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
    })

    os.makedirs("models", exist_ok=True)

    model_path = "models/gnn_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_features": data.x.shape[1],
            "edge_features": data.edge_attr.shape[1],
            "hidden_size": 64,
        },
        model_path
    )

    mlflow.log_artifact(model_path)

    print()
    print(f"Saved model: {model_path}")