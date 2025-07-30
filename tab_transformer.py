import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
import numpy as np
import math
import time
import os
from torch.utils.data import TensorDataset, DataLoader


def setupmissval(data, miss) -> None:
    for index, row in miss.iterrows():
        cpg = row["CPG"]
        samp = row["SAMPLE"]
        true = row["TRUEVAL"]

        if data.loc[samp, cpg] != true:
            sys.exit(
                "Methylation data value doesn't match missing value data:",
                true,
                " is different from ",
                data.loc[samp, cpg],
            )
        data.loc[samp, cpg] = pd.NA  # Mask the value as missing


def split_matrix_by_group(data, group_data):
    group_indices = {}
    for _, row in group_data.iterrows():
        cpg_id = row["cpg"]
        grp = row["chr"]
        if cpg_id not in data.columns:
            continue
        col_idx = data.columns.get_loc(cpg_id)
        group_indices.setdefault(grp, []).append(col_idx)

    submatrices = [data.iloc[:, inds] for inds in group_indices.values()]
    return submatrices


def RMSE(pred, orig) -> float:
    return math.sqrt(np.mean((pred - orig) ** 2))


def MAE(pred, orig) -> float:
    return np.mean(np.abs(pred - orig))


# ARGPARSE and RUN PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Path to methylation data CSV")
parser.add_argument("miss", type=str, help="Path to missing-values CSV")
parser.add_argument("groups", type=str, help="Path to group-Info CSV")

# Run_name
parser.add_argument(
    "--run_name",
    type=str,
    required=True,
    help="Short identifier for this hyperparam set, e.g. e64_h4_l2",
)

# hyperparameters as args
parser.add_argument(
    "--dim_embedding", type=int, default=64, help="Transformer embedding dim"
)
parser.add_argument(
    "--num_heads", type=int, default=4, help="Number of attention heads"
)
parser.add_argument(
    "--num_layers", type=int, default=4, help="Number of transformer layers"
)
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--learning_rate", type=float, default=1e-3, help="Adam learning rate"
)
parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs per sample")
parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")

args = parser.parse_args()

# Assign Hyper_parameters
data_path = args.data
miss_path = args.miss
groups_path = args.groups
run_name = args.run_name
dim_embedding = args.dim_embedding
num_heads = args.num_heads
num_layers = args.num_layers
dropout = args.dropout
learning_rate = args.learning_rate
batch_size = args.batch_size
max_epochs = args.max_epochs
patience_limit = args.patience

start_time = time.time()

print("Reading methylation data")
data = pd.read_csv(data_path, index_col=0)
print("Transposing matrix so that samples are rows and CpG sites are columns")
data = data.T
print("Data shape:", data.shape)
original_data = data.copy(deep=True)

print("Reading missing values information")
miss = pd.read_csv(miss_path)
out = miss.copy(deep=True)


print("Marking designated missing values as NA in the data")
setupmissval(data, miss)

print("Reading and grouping the CpGs")
group_data = pd.read_csv(groups_path)
grouped_data = split_matrix_by_group(data, group_data)


# TabTransforemer
class TabTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim_embedding=64,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_embed = nn.Linear(1, dim_embedding)
        encoder = nn.TransformerEncoderLayer(
            d_model=dim_embedding, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, num_features, 1]
        x = self.feature_embed(x)  # [B, num_features, dim_emb]
        x = self.transformer(x)  # [B, num_features, dim_emb]
        x = x.mean(dim=1)  # [B, dim_emb]
        return self.activation(self.classifier(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

imputed_results = {}

# Loop over groups and samples
for group_idx, submatrix in enumerate(grouped_data):
    print(f"\nGroup {group_idx + 1}/{len(grouped_data)}")

    target_samples = [s for s in submatrix.index if submatrix.loc[s].isnull().any()]

    for i, target_sample in enumerate(target_samples, start=1):
        print(f"\n Sample {i}/{len(target_samples)}: {target_sample}")

        # Identify missing vs observed
        missing_cols = submatrix.columns[submatrix.loc[target_sample].isnull()]
        observed_cols = submatrix.columns[~submatrix.loc[target_sample].isnull()]
        true_values = original_data.loc[target_sample, missing_cols]

        # Build training data
        train_data = submatrix.drop(index=target_sample)
        X_train = train_data[observed_cols].fillna(
            original_data.loc[train_data.index, observed_cols]
        )
        Y_train = train_data[missing_cols].fillna(
            original_data.loc[train_data.index, missing_cols]
        )

        # Tensors
        X = torch.FloatTensor(X_train.to_numpy())
        Y = torch.FloatTensor(Y_train.to_numpy())

        # Train/val split
        N = X.size(0)
        v = int(0.2 * N)
        idx = torch.randperm(N)
        X_val, Y_val = X[idx[:v]], Y[idx[:v]]
        X_tr, Y_tr = X[idx[v:]], Y[idx[v:]]

        train_loader = DataLoader(
            TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

        # model instantiation with args
        model = TabTransformer(
            input_dim=len(observed_cols),
            output_dim=len(missing_cols),
            dim_embedding=dim_embedding,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # training + early stopping
        best_val = float("inf")
        patience = 0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            # validation pass
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_losses.append(criterion(model(xb), yb).item())
            avg_val = sum(val_losses) / len(val_losses)
            print(f"  Epoch {epoch + 1}/{max_epochs}, Val Loss {avg_val:.4f}")

            if avg_val < best_val:
                best_val = avg_val
                patience = 0
                best_state = model.state_dict()
            else:
                patience += 1
                if patience >= patience_limit:
                    print("  Early stopping")
                    break

        model.load_state_dict(best_state)

        # Predict the target sample
        tobs = (
            torch.FloatTensor(submatrix.loc[target_sample, observed_cols].to_numpy())
            .unsqueeze(0)
            .to(device)
        )

        model.eval()
        with torch.no_grad():
            pred = model(tobs).cpu().numpy().reshape(-1)
        imputed_results[target_sample] = dict(zip(missing_cols, pred))

        # Fill out DataFrame
        for cpg, val in imputed_results[target_sample].items():
            ind = out[(out.CPG == cpg) & (out.SAMPLE == target_sample)].index[0]
            out.loc[ind, "TabImpute"] = val

    #  filename per run
    output_filename = f"ti_{run_name}.csv"
    out.to_csv(output_filename, index=False)

# Final evaluation & summary
all_preds = []
all_trues = []
for sample, pdict in imputed_results.items():
    cols = list(pdict.keys())
    pvals = np.array(list(pdict.values()))
    tvals = original_data.loc[sample, cols].to_numpy()
    all_preds.extend(pvals)
    all_trues.extend(tvals)

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)
mae_value = MAE(all_preds, all_trues)
rmse_value = RMSE(all_preds, all_trues)

print(f"\nOverall MAE: {mae_value:.4f}, RMSE: {rmse_value:.4f}")

# append to summary.csv
summary_file = "summary.csv"
header = not os.path.exists(summary_file)
with open(summary_file, "a") as f:
    if header:
        f.write(
            "run_name,dim_embedding,num_heads,"
            "num_layers,dropout,learning_rate,batch_size,MAE,RMSE\n"
        )
    f.write(
        f"{run_name},{dim_embedding},{num_heads},"
        f"{num_layers},{dropout},{learning_rate},"
        f"{batch_size},{mae_value:.4f},{rmse_value:.4f}\n"
    )

end_time = time.time()
print(f"Total running time: {end_time - start_time:.1f}s")
