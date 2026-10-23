import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from torch_geometric.nn import to_hetero
from util import create_parser, logger_setup
from data_loading import get_data
from training import get_model
from train_util import AddEgoIds, add_arange_ids, get_loaders, extract_param

def run_detection(model_name="gin_hi_small", data_name="HI-Small", top_k=25, alert_threshold=0.35):
    logger_setup()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device for detection: {device}")

    with open('data_config.json', 'r') as f:
        data_config = json.load(f)

    # Setup dummy args for model and loader construction
    parser = create_parser()
    args = parser.parse_args([
        "--data", data_name,
        "--model", "gin",
        "--emlps",
        "--reverse_mp",
        "--ports",
        "--batch_size", "2048",
        "--num_neighs", "20", "10",
        "--testing"
    ])
    args.unique_name = model_name

    # Load data (will instantly load from preprocessed cache)
    logging.info("Loading transaction graph data...")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    # Build data loader for test split
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    _, _, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    # Initialize model architecture
    sample_batch = next(iter(te_loader))
    
    # Dummy config matching model settings
    class Config:
        n_gnn_layers = extract_param("n_gnn_layers", args) or 2
        n_hidden = extract_param("n_hidden", args) or 66
        dropout = extract_param("dropout", args) or 0.01
        final_dropout = extract_param("final_dropout", args) or 0.1
    
    config = Config()
    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    # Load trained checkpoint
    checkpoint_path = f"{data_config['paths']['model_to_load']}/checkpoint_{model_name}.tar"
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found at: {checkpoint_path}")
        return

    logging.info(f"Loading trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info(f"Model successfully loaded (trained up to epoch {checkpoint['epoch']}).")

    # Load raw transactions to map back to original account IDs, amounts, and dates
    raw_csv_path = f"{data_config['paths']['aml_data']}/{data_name}/formatted_transactions.csv"
    logging.info(f"Reading transaction metadata from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)

    # Run inference on test batches
    logging.info(f"Scoring test transactions ({len(te_inds):,} test transactions)...")
    all_scores = []
    all_ground_truths = []
    all_edge_ids = []

    inds_cpu = te_inds.detach().cpu()
    te_data_edge_ids = te_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[:, 0]

    with torch.no_grad():
        for batch in te_loader:
            batch_edge_inds = inds_cpu[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = te_data_edge_ids[batch_edge_inds]
            raw_edge_col0 = batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu()
            mask = torch.isin(raw_edge_col0, batch_edge_ids)
            matched_edge_ids = raw_edge_col0[mask].numpy().astype(int)

            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')][mask]

            # Softmax to get probability of laundering (Class 1)
            probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()
            ground_truth = batch['node', 'to', 'node'].y[mask].cpu().numpy()

            all_scores.extend(probs)
            all_ground_truths.extend(ground_truth)
            all_edge_ids.extend(matched_edge_ids)

    all_scores = np.array(all_scores)
    all_ground_truths = np.array(all_ground_truths)
    all_edge_ids = np.array(all_edge_ids)

    # Build detection results dataframe
    results_df = df_raw.iloc[all_edge_ids].copy()
    results_df['AML_Risk_Score'] = all_scores
    results_df['Predicted_Laundering'] = (all_scores >= alert_threshold).astype(int)
    results_df['Ground_Truth'] = all_ground_truths

    # Sort by risk score descending
    flagged_df = results_df.sort_values(by='AML_Risk_Score', ascending=False)

    # Print detection statistics
    num_flagged = (results_df['Predicted_Laundering'] == 1).sum()
    actual_illicit = (results_df['Ground_Truth'] == 1).sum()
    true_positives = ((results_df['Predicted_Laundering'] == 1) & (results_df['Ground_Truth'] == 1)).sum()

    print("\n" + "="*80)
    print("              MULTI-GNN ANTI-MONEY LAUNDERING DETECTION REPORT              ")
    print("="*80)
    print(f"Total Test Transactions Evaluated: {len(results_df):,}")
    print(f"Transactions Flagged as Suspicious (Risk >= {alert_threshold:.2f}): {num_flagged:,}")
    print(f"Actual Illicit Transactions in Test Set: {actual_illicit:,}")
    print(f"True Positives Detected: {true_positives:,}")
    if num_flagged > 0:
        print(f"Detection Precision: {true_positives / num_flagged * 100:.2f}%")
    print("="*80 + "\n")

    print(f"Top {top_k} Highest-Risk Detected Transactions:")
    print("-" * 105)
    header = f"{'EdgeID':<10} {'From Acct':<12} {'To Acct':<12} {'Amount ($)':<14} {'Laundering Risk':<18} {'Actual Label':<15}"
    print(header)
    print("-" * 105)
    for _, row in flagged_df.head(top_k).iterrows():
        actual_str = "ILLICIT [AML]" if row['Ground_Truth'] == 1 else "Normal"
        print(f"{int(row['EdgeID']):<10} {int(row['from_id']):<12} {int(row['to_id']):<12} {row['Amount Received']:<14.2f} {row['AML_Risk_Score']*100:>6.2f}%            {actual_str:<15}")
    print("-" * 105)

    # Save alerts to CSV
    alerts_path = "checkpoints/detected_aml_alerts.csv"
    flagged_df.to_csv(alerts_path, index=False)
    print(f"\nAll scored test transactions and alerts saved to: {alerts_path}")

    # Identify high risk accounts (money mules)
    print("\nTop 5 Most Suspicious Accounts (Highest Flagged Outgoing Transaction Volume):")
    suspicious_senders = flagged_df[flagged_df['Predicted_Laundering'] == 1].groupby('from_id')['Amount Received'].agg(['count', 'sum']).reset_index()
    suspicious_senders.columns = ['Account_ID', 'Flagged_Transactions', 'Total_Flagged_Volume']
    suspicious_senders = suspicious_senders.sort_values(by='Total_Flagged_Volume', ascending=False).head(5)
    print(suspicious_senders.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_detection()
