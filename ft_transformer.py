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
import gc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings

warnings.filterwarnings("ignore")


def setup_missing_values(data, miss_data):
    """Setup missing values in the data matrix vectorized version using numpy -> faster"""
    print("Setting up missing values ...")

    # Ensure we only work with CPGs and Samples that actually exist in the data
    valid_cpgs = miss_data["CPG"].isin(data.columns)
    valid_samples = miss_data["SAMPLE"].isin(data.index)
    miss_data_filtered = miss_data[valid_cpgs & valid_samples]

    # Get the integer positions for rows (samples) and columns (cpgs)
    row_indices = data.index.get_indexer(miss_data_filtered["SAMPLE"])
    col_indices = data.columns.get_indexer(miss_data_filtered["CPG"])

    data_vals = data.values.copy()
    data_vals[row_indices, col_indices] = np.nan

    # Create a new DataFrame from the modified numpy array
    data_modified = pd.DataFrame(data_vals, index=data.index, columns=data.columns)

    print("Finished setting up missing values.")
    return data_modified


def create_missing_mask(data):
    """Create a mask indicating which values are missing"""
    return data.isnull()


class MethylationTransformer(nn.Module):
    """
    Architecture is same as the original and the missing values handeling is same as another model which handeling missing values.
    I just adopt another model's missing values handeling and add some new features to the model.
    Key properties:
    1. CPG-specific embeddings to capture site-specific patterns
    2. Value embeddings to handle different methylation levels
    3. Missing value embeddings for proper handling of NaN values
    4. Multi-head attention to capture complex relationships
    5. Residual connections and layer normalization
    """

    def __init__(
        self,
        num_cpgs,
        embedding_dim=128,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_value=1.0,
    ):
        super().__init__()

        self.num_cpgs = num_cpgs
        self.embedding_dim = embedding_dim
        self.max_value = max_value

        # CPG site embeddings - each CpG site gets its own embedding
        # This allows the model to learn site-specific patterns
        self.cpg_embedding = nn.Embedding(num_cpgs, embedding_dim)

        # Value embeddings - transform methylation values to embeddings
        # This helps the model understand different methylation levels
        self.value_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        # Missing value embedding - special embedding for NaN values
        # This tells the model when a value is missing vs. when it's 0
        self.missing_embedding = nn.Parameter(torch.randn(1, embedding_dim))

        # Position embeddings - capture positional information
        self.position_embedding = nn.Parameter(torch.randn(1, num_cpgs, embedding_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,  # Larger feedforward network
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # GELU activation for better performance (what is suggested based on their paper)
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection layers
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid(),  # Ensure output is between 0 and 1
        )

    def forward(self, x, mask):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, num_cpgs] - methylation values
            mask: Boolean mask [batch_size, num_cpgs] - True for missing values

        Returns:
            Output tensor [batch_size, num_cpgs] - predicted methylation values
        """
        batch_size, num_cpgs = x.shape
        device = x.device

        # Create CPG indices for embedding lookup
        cpg_indices = (
            torch.arange(num_cpgs, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Get CPG embeddings [batch_size, num_cpgs, embedding_dim]
        cpg_emb = self.cpg_embedding(cpg_indices)

        # Handle value embeddings
        # Replace NaN values with 0 for embedding computation
        x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x_expanded = x_clean.unsqueeze(-1)  # [batch_size, num_cpgs, 1]

        # Get value embeddings
        value_emb = self.value_embedding(x_expanded)

        # Apply missing value embeddings where mask is True
        missing_emb = self.missing_embedding.expand(batch_size, num_cpgs, -1)
        value_emb = torch.where(mask.unsqueeze(-1), missing_emb, value_emb)

        # Add position embeddings
        pos_emb = self.position_embedding.expand(batch_size, -1, -1)

        # Combine all embeddings
        combined = cpg_emb + value_emb + pos_emb

        # Apply transformer encoder
        # The transformer will learn relationships between different CpG sites
        encoded = self.transformer(combined)

        # Project to output values
        output = self.output_head(encoded).squeeze(-1)

        return output


def create_data_loaders(
    X, y, mask, batch_size=32, train_indices=None, val_indices=None
):
    """Create data loaders for training and validation"""

    if train_indices is not None and val_indices is not None:
        # Split data
        X_train, y_train, mask_train = (
            X[train_indices],
            y[train_indices],
            mask[train_indices],
        )
        X_val, y_val, mask_val = X[val_indices], y[val_indices], mask[val_indices]

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.BoolTensor(mask_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.BoolTensor(mask_val),
        )

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        # Single dataset
        dataset = TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y), torch.BoolTensor(mask)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (x, y, mask) in enumerate(train_loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(x, mask)

        # Only compute loss on missing values (where mask is True)
        loss = criterion(output[mask], y[mask])

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent Gradiant to become excessively large
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_predictions(model, data, missing_mask, original_data, device):
    """Evaluate model predictions on actual missing values"""
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        # Process each sample with missing values
        for i in range(len(data)):
            sample = data.iloc[i]
            sample_mask = missing_mask.iloc[i]

            # Only process if there are missing values
            if sample_mask.any():
                # Get observed and missing indices
                missing_indices = np.where(sample_mask)[0]
                observed_indices = np.where(~sample_mask)[0]

                if len(observed_indices) > 0:
                    # Create input tensor
                    x = torch.FloatTensor(sample.values).unsqueeze(0).to(device)
                    mask = torch.BoolTensor(sample_mask.values).unsqueeze(0).to(device)

                    # Get predictions
                    output = model(x, mask).cpu().numpy()[0]

                    # Store predictions for missing values
                    for idx in missing_indices:
                        predictions.append(output[idx])
                        true_values.append(original_data.iloc[i, idx])

    if len(predictions) > 0:
        predictions = np.array(predictions)
        true_values = np.array(true_values)

        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        correlation = np.corrcoef(true_values, predictions)[0, 1]

        return mae, rmse, correlation, predictions, true_values
    else:
        return None, None, None, [], []


def evaluate_predictions_with_reference(
    model, data, missing_mask, original_data, device, reference_csv, output_csv=None
):
    """Evaluate model predictions and write detailed results with reference columns."""
    ref_df = pd.read_csv(reference_csv)
    # Build lookup for (CPG, SAMPLE) -> MethyLImp2, MethyLImp2_10000
    ref_lookup = {
        (str(row["CPG"]), str(row["SAMPLE"])): (
            row["MethyLImp2"],
            row["MethyLImp2_10000"],
        )
        for _, row in ref_df.iterrows()
    }

    model.eval()
    results = []

    with torch.no_grad():
        for i in range(len(data)):
            sample = data.iloc[i]
            sample_mask = missing_mask.iloc[i]
            sample_id = str(data.index[i])

            if sample_mask.any():
                missing_indices = np.where(sample_mask)[0]
                cpg_names = data.columns[missing_indices]

                x = torch.FloatTensor(sample.values).unsqueeze(0).to(device)
                mask = torch.BoolTensor(sample_mask.values).unsqueeze(0).to(device)
                output = model(x, mask).cpu().numpy()[0]

                for idx, cpg in zip(missing_indices, cpg_names):
                    cpg_str = str(cpg)
                    methylimp2, methylimp2_10000 = ref_lookup.get(
                        (cpg_str, sample_id), (None, None)
                    )
                    results.append(
                        {
                            "CPG": cpg_str,
                            "SAMPLE": sample_id,
                            "TRUEVAL": original_data.iloc[i, idx],
                            "MODEL_PREDICTION": output[idx],
                            "MethyLImp2": methylimp2,
                            "MethyLImp2_10000": methylimp2_10000,
                        }
                    )

    if output_csv is not None:
        pd.DataFrame(results).to_csv(output_csv, index=False)
    return results


def get_detailed_predictions(
    model, data, missing_mask, original_data, device, miss_file, group_cpgs
):
    """Get detailed predictions for missing values in the current group only."""
    import csv

    miss_df = pd.read_csv(miss_file)
    # Filter to only CPGs in the current group
    miss_df_filtered = miss_df[miss_df["CPG"].isin(group_cpgs)]

    predictions = []
    sample_to_index = {str(idx): i for i, idx in enumerate(data.index)}
    model.eval()

    with torch.no_grad():
        for idx, row in miss_df_filtered.iterrows():
            cpg = str(row["CPG"])
            sample = str(row["SAMPLE"])
            if sample in sample_to_index and cpg in data.columns:
                i = sample_to_index[sample]
                j = data.columns.get_loc(cpg)
                sample_values = data.iloc[i].values
                sample_mask = missing_mask.iloc[i].values
                x = torch.FloatTensor(sample_values).unsqueeze(0).to(device)
                mask = torch.BoolTensor(sample_mask).unsqueeze(0).to(device)
                output = model(x, mask).cpu().numpy()[0]
                pred = output[j]
            else:
                pred = None

            # Keep all original columns from the missing file and add our prediction
            row_out = row.to_dict()
            row_out["FT_TRANSFORMER_PREDICTION"] = pred
            predictions.append(row_out)

    return predictions


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Advanced Transformer for Methylation Imputation"
    )
    parser.add_argument("data", type=str, help="Path to methylation data CSV")
    parser.add_argument("miss", type=str, help="Path to missing-values CSV")
    parser.add_argument("--run_name", type=str, required=True, help="Run identifier")
    parser.add_argument(
        "--group_file",
        type=str,
        required=True,
        help="Path to CSV file containing feature groups",
    )

    # Model hyperparameters
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    start_time = time.time()

    print("FT-Transformer Methylation Imputation with Leave-One-Out")
    print(f"Run name: {args.run_name}")
    print(
        f"Model: {args.embedding_dim}d, {args.num_heads} heads, {args.num_layers} layers"
    )

    # Load data
    print("\nLoading data...")
    data = pd.read_csv(args.data, index_col=0)
    print(f"Original data shape: {data.shape}")

    # Transpose to get samples as rows
    data = data.T
    print(f"Transposed data shape: {data.shape}")

    # Load group file
    print("\nLoading group file...")
    group_df = pd.read_csv(args.group_file)
    print(f"Group file loaded with {len(group_df)} entries")

    # Get unique groups (using 'chr' column as group identifier)
    unique_groups = group_df["chr"].unique()
    print(f"Found {len(unique_groups)} unique groups")

    # Load missing values information
    miss_data = pd.read_csv(args.miss)
    print(f"Missing values to impute: {len(miss_data)}")

    # Setup missing values
    original_data = data.copy()
    data = setup_missing_values(data, miss_data)

    # Create missing mask
    missing_mask = create_missing_mask(data)
    print(f"Samples with missing values: {missing_mask.any(axis=1).sum()}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Store summary results
    summary_results = []

    # Create output files for incremental saving
    predictions_filename = f"ft_TabTransformer_{args.run_name}.csv"
    hyperparams_filename = "ft_hyperparameters_results.csv"

    # Initialize hyperparameters results file if it doesn't exist
    if not os.path.exists(hyperparams_filename):
        hyperparams_headers = [
            "run_name",
            "embedding_dim",
            "num_heads",
            "num_layers",
            "dropout",
            "learning_rate",
            "batch_size",
            "max_epochs",
            "overall_mae",
            "overall_rmse",
            "total_groups",
            "total_cpgs",
            "runtime_seconds",
        ]
        pd.DataFrame(columns=hyperparams_headers).to_csv(
            hyperparams_filename, index=False
        )

    # Initialize prediction file with headers (read from missing file to get all columns)
    miss_df = pd.read_csv(args.miss)
    prediction_headers = list(miss_df.columns) + ["FT_TRANSFORMER_PREDICTION"]
    pd.DataFrame(columns=prediction_headers).to_csv(predictions_filename, index=False)

    print(f"\nStarting group-based training with leave-one-out...")
    print(f"Predictions will be saved incrementally to: {predictions_filename}")

    imputed_results = {}

    # Main loop: iterate through each group (similar to TabTransformer)
    for group_idx, group_id in enumerate(unique_groups, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing Group {group_id} ({group_idx}/{len(unique_groups)})")
        print(f"{'=' * 60}")

        # Get CPGs for this group
        group_cpgs = group_df[group_df["chr"] == group_id]["cpg"].tolist()

        # Filter to CPGs that exist in our data
        existing_cpgs = [cpg for cpg in group_cpgs if cpg in data.columns]

        if len(existing_cpgs) == 0:
            print(f"No valid CPGs found for group {group_id}, skipping...")
            continue

        print(f"Group {group_id}: {len(existing_cpgs)} CPGs")

        # Create group-specific data slices
        data_group = data[existing_cpgs]
        original_data_group = original_data[existing_cpgs]
        missing_mask_group = missing_mask[existing_cpgs]

        # Find samples with missing values in this group (like TabTransformer)
        target_samples = [
            s for s in data_group.index if missing_mask_group.loc[s].any()
        ]

        if len(target_samples) == 0:
            print(f"No samples with missing values in group {group_id}, skipping...")
            continue

        print(f"Samples with missing values: {len(target_samples)}")

        # Leave-one-out loop for each sample (similar to TabTransformer)
        for i, target_sample in enumerate(target_samples, start=1):
            print(f"\n  Sample {i}/{len(target_samples)}: {target_sample}")

            # Identify missing vs observed columns for this sample
            missing_cols = data_group.columns[missing_mask_group.loc[target_sample]]
            observed_cols = data_group.columns[~missing_mask_group.loc[target_sample]]

            if len(observed_cols) == 0 or len(missing_cols) == 0:
                print(
                    f"    Skipping sample {target_sample} - no observed or missing values"
                )
                continue

            # Build training data (exclude target sample)
            train_data = data_group.drop(index=target_sample)
            train_missing_mask = missing_mask_group.drop(index=target_sample)
            train_original = original_data_group.drop(index=target_sample)

            # Fill missing values in training data with original values
            train_data_filled = train_data.fillna(train_original)

            # Prepare training arrays
            X_train = train_data_filled.values
            y_train = train_original.values
            mask_train = train_missing_mask.values

            # Create data loader for whole training dataset
            train_loader = create_data_loaders(
                X_train, y_train, mask_train, args.batch_size
            )

            # Initialize model for this sample
            model = MethylationTransformer(
                num_cpgs=len(existing_cpgs),
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)

            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-4
            )

            # Training loop - train for all epochs
            for epoch in range(args.max_epochs):
                # Train
                train_loss = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                if epoch % 20 == 0:
                    print(f"    Epoch {epoch:3d}: Train Loss = {train_loss:.6f}")

            # Predict on target sample
            target_sample_data = data_group.loc[target_sample].fillna(0).values
            target_sample_mask = missing_mask_group.loc[target_sample].values

            x_target = torch.FloatTensor(target_sample_data).unsqueeze(0).to(device)
            mask_target = torch.BoolTensor(target_sample_mask).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(x_target, mask_target).cpu().numpy()[0]

            # Store results for missing columns only
            sample_predictions = {}
            for j, col in enumerate(missing_cols):
                col_idx = data_group.columns.get_loc(col)
                sample_predictions[col] = output[col_idx]

            imputed_results[target_sample] = sample_predictions

            # Save predictions incrementally
            group_predictions = []
            for cpg, pred_val in sample_predictions.items():
                # Find the corresponding row in miss_data
                mask = (miss_df["CPG"] == cpg) & (miss_df["SAMPLE"] == target_sample)
                if mask.any():
                    row_data = miss_df[mask].iloc[0].to_dict()
                    row_data["FT_TRANSFORMER_PREDICTION"] = pred_val
                    group_predictions.append(row_data)

            if len(group_predictions) > 0:
                group_predictions_df = pd.DataFrame(group_predictions)
                group_predictions_df.to_csv(
                    predictions_filename, mode="a", header=False, index=False
                )

            # Memory cleanup
            del model, optimizer, train_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final evaluation (similar to TabTransformer)
    all_preds = []
    all_trues = []
    for sample, pdict in imputed_results.items():
        cols = list(pdict.keys())
        pvals = np.array(list(pdict.values()))
        tvals = original_data.loc[sample, cols].to_numpy()
        all_preds.extend(pvals)
        all_trues.extend(tvals)

    if len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        avg_mae = mean_absolute_error(all_trues, all_preds)
        avg_rmse = np.sqrt(mean_squared_error(all_trues, all_preds))

        print(f"\n{'=' * 60}")
        print("=== Final Results ===")
        print(f"{'=' * 60}")
        print(f"Overall MAE: {avg_mae:.6f}")
        print(f"Overall RMSE: {avg_rmse:.6f}")
        print(f"Total samples processed: {len(imputed_results)}")
        print(f"All predictions saved to {predictions_filename}")

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"\nTotal running time: {total_runtime:.1f}s")

        # Save hyperparameters results
        hyperparams_result = {
            "run_name": args.run_name,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "overall_mae": avg_mae,
            "overall_rmse": avg_rmse,
            "total_groups": len(unique_groups),
            "total_cpgs": len(
                [
                    cpg
                    for group_cpgs in [
                        group_df[group_df["chr"] == gid]["cpg"].tolist()
                        for gid in unique_groups
                    ]
                    for cpg in group_cpgs
                    if cpg in data.columns
                ]
            ),
            "runtime_seconds": total_runtime,
        }

        hyperparams_df = pd.DataFrame([hyperparams_result])
        hyperparams_df.to_csv(hyperparams_filename, mode="a", header=False, index=False)
        print(f"Hyperparameters results appended to {hyperparams_filename}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
