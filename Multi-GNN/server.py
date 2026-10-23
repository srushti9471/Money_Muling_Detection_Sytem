import os
import io
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

from models import GINe
from torch_geometric.nn import to_hetero
from data_util import GraphData, z_norm, create_hetero_obj
import networkx as nx

app = FastAPI(title="MONEY MULING DETECTION SYSTEM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FORMAT_MAP = {
    0: 'Reinvestment',
    1: 'Cheque',
    2: 'Credit Card',
    3: 'ACH',
    4: 'Cash',
    5: 'Wire',
    6: 'Bitcoin'
}
REV_FORMAT_MAP = {v.lower(): k for k, v in FORMAT_MAP.items()}

CURRENCY_MAP = {
    0: 'USD',
    1: 'BTC',
    2: 'EUR',
    3: 'AUD',
    4: 'CNY',
    5: 'INR',
    6: 'JPY',
    7: 'MXN',
    8: 'GBP',
    9: 'RUB',
    10: 'CAD',
    11: 'CHF',
    12: 'BRL',
    13: 'SAR',
    14: 'ILS'
}
REV_CURRENCY_MAP = {
    'us dollar': 0, 'usd': 0, 'bitcoin': 1, 'btc': 1, 'euro': 2, 'eur': 2,
    'australian dollar': 3, 'aud': 3, 'yuan': 4, 'cny': 4, 'rupee': 5, 'inr': 5,
    'yen': 6, 'jpy': 6, 'mexican peso': 7, 'uk pound': 8, 'gbp': 8, 'ruble': 9,
    'canadian dollar': 10, 'cad': 10, 'swiss franc': 11, 'chf': 11
}

ALERTS_CSV_PATH = "checkpoints/detected_aml_alerts.csv"
CHECKPOINT_PATH = "checkpoints/checkpoint_gin_hi_small.tar"

df_alerts: Optional[pd.DataFrame] = None
account_stats: Dict[int, Dict[str, Any]] = {}
precomputed_subgraphs: Dict[str, pd.DataFrame] = {}
active_dataset_info: Dict[str, Any] = {
    "name": "IBM HI-Small (Pre-Computed)",
    "is_custom": False,
    "total_tx": 0,
    "inference_time": "Pre-computed"
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_account_stats(df: pd.DataFrame):
    """Precompute fast per-account metrics and role classification."""
    stats = {}
    for _, row in df.iterrows():
        f_id = int(row['from_id'])
        t_id = int(row['to_id'])
        amt = float(row['Amount Received'])
        risk = float(row['AML_Risk_Score'])
        is_susp = risk >= 0.35

        for acc in (f_id, t_id):
            if acc not in stats:
                stats[acc] = {
                    'account_id': acc,
                    'in_count': 0,
                    'out_count': 0,
                    'in_volume': 0.0,
                    'out_volume': 0.0,
                    'flagged_in': 0,
                    'flagged_out': 0,
                    'flagged_volume': 0.0,
                    'max_risk': 0.0,
                    'counterparties': set(),
                    'is_suspicious': False
                }

        stats[f_id]['out_count'] += 1
        stats[f_id]['out_volume'] += amt
        stats[f_id]['counterparties'].add(t_id)
        if risk > stats[f_id]['max_risk']:
            stats[f_id]['max_risk'] = risk
        if is_susp:
            stats[f_id]['flagged_out'] += 1
            stats[f_id]['flagged_volume'] += amt
            stats[f_id]['is_suspicious'] = True

        stats[t_id]['in_count'] += 1
        stats[t_id]['in_volume'] += amt
        stats[t_id]['counterparties'].add(f_id)
        if risk > stats[t_id]['max_risk']:
            stats[t_id]['max_risk'] = risk
        if is_susp:
            stats[t_id]['flagged_in'] += 1
            stats[t_id]['flagged_volume'] += amt
            stats[t_id]['is_suspicious'] = True

    for acc, info in stats.items():
        info['counterparty_count'] = len(info['counterparties'])
        info.pop('counterparties', None)
        
        if info['is_suspicious']:
            if info['flagged_out'] >= 2 and info['flagged_in'] >= 2:
                info['role'] = 'Money Mule (Gather-Scatter Hub)'
            elif info['flagged_out'] >= 3:
                info['role'] = 'Smurfer / Fan-Out Source'
            elif info['flagged_in'] >= 3:
                info['role'] = 'Cash-Out Sink / Aggregator'
            else:
                info['role'] = 'Suspicious Intermediary'
        else:
            info['role'] = 'Normal Counterparty'

    return stats

def precompute_dense_subgraphs(df: pd.DataFrame, stats: Dict[int, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Precomputes connected dense multigraph topologies around detected laundering activity."""
    subgraphs = {}
    
    # Identify suspicious accounts (threshold >= 0.35)
    susp_df = df[df['AML_Risk_Score'] >= 0.35]
    susp_nodes = set(susp_df['from_id']).union(set(susp_df['to_id']))
    
    # All transactions in the test split touching suspicious accounts (licit and illicit)
    connected_df = df[df['from_id'].isin(susp_nodes) | df['to_id'].isin(susp_nodes)].copy()
    
    # Build graph to extract connected components
    G = nx.Graph()
    for _, r in connected_df.iterrows():
        G.add_edge(int(r['from_id']), int(r['to_id']))
        
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    
    # 1. Dense Banking Core (Giant connected component: 382 nodes, 688 transactions)
    if len(comps) > 0:
        c0_nodes = comps[0]
        subgraphs['dense_core'] = connected_df[
            connected_df['from_id'].isin(c0_nodes) & connected_df['to_id'].isin(c0_nodes)
        ].copy()
    else:
        subgraphs['dense_core'] = connected_df.head(600).copy()

    # 2. Top Mule Rings: Centered around high-centrality gather-scatter hubs
    top_mules = {acc for acc, s in stats.items() if s['flagged_out'] >= 1 and s['flagged_in'] >= 1}
    if len(top_mules) > 0:
        subgraphs['mule_rings'] = connected_df[
            connected_df['from_id'].isin(top_mules) | connected_df['to_id'].isin(top_mules)
        ].copy()
    else:
        subgraphs['mule_rings'] = subgraphs['dense_core'].copy()

    # 3. Fan-out / Smurfing Networks: High out-degree dispersion accounts
    fan_out_accs = {acc for acc, s in stats.items() if s['flagged_out'] >= 2}
    if len(fan_out_accs) > 0:
        subgraphs['fan_out'] = connected_df[
            connected_df['from_id'].isin(fan_out_accs) | connected_df['to_id'].isin(fan_out_accs)
        ].copy()
    else:
        subgraphs['fan_out'] = subgraphs['dense_core'].copy()

    # 4. Multi-Ring Clusters: Top 3 largest laundering networks combined (492 nodes, 890 edges)
    top3_nodes = set().union(*comps[:min(3, len(comps))])
    subgraphs['all_clusters'] = connected_df[
        connected_df['from_id'].isin(top3_nodes) & connected_df['to_id'].isin(top3_nodes)
    ].copy()

    print(f"Precomputed dense subgraphs: dense_core={len(subgraphs['dense_core'])}, mule_rings={len(subgraphs['mule_rings'])}, fan_out={len(subgraphs['fan_out'])}, all_clusters={len(subgraphs['all_clusters'])}")
    return subgraphs

def load_default_data():
    global df_alerts, account_stats, active_dataset_info, precomputed_subgraphs
    if not os.path.exists(ALERTS_CSV_PATH):
        raise FileNotFoundError(f"{ALERTS_CSV_PATH} not found. Run detect_aml.py first.")
    
    print(f"Loading {ALERTS_CSV_PATH} into memory...")
    df = pd.read_csv(ALERTS_CSV_PATH)
    df['Format_Name'] = df['Payment Format'].map(lambda x: FORMAT_MAP.get(int(x), f"Format {x}"))
    df['Currency_Name'] = df['Received Currency'].map(lambda x: CURRENCY_MAP.get(int(x), f"Curr {x}"))
    df['Risk_Percent'] = (df['AML_Risk_Score'] * 100).round(2)
    df_alerts = df

    print("Computing account aggregation metrics...")
    account_stats = compute_account_stats(df)
    
    print("Precomputing dense financial multigraph components...")
    precomputed_subgraphs = precompute_dense_subgraphs(df, account_stats)

    active_dataset_info = {
        "name": "IBM HI-Small (Pre-Computed)",
        "is_custom": False,
        "total_tx": len(df),
        "inference_time": "Pre-computed"
    }
    print(f"Loaded {len(df_alerts):,} transactions and {len(account_stats):,} unique accounts.")

def run_gnn_inference_on_df(df_input: pd.DataFrame) -> pd.DataFrame:
    """Runs GNN model inference on an arbitrary uploaded dataframe."""
    t0 = time.perf_counter()
    
    # 1. Check if raw format (From Bank, To Bank, etc.) or formatted format
    cols = [c.strip() for c in df_input.columns]
    df_input.columns = cols
    
    if 'From Bank' in cols or 'Account' in cols:
        print("Detected raw transaction format. Preprocessing features...")
        account_map = {}
        # From Account
        from_col = df_input.iloc[:, 2] if len(df_input.columns) > 2 else df_input['Account']
        to_col = df_input.iloc[:, 4] if len(df_input.columns) > 4 else df_input['Account.1']
        
        from_str = df_input['From Bank'].astype(str) + from_col.astype(str)
        to_str = df_input['To Bank'].astype(str) + to_col.astype(str)

        def map_accounts(series):
            res = np.empty(len(series), dtype=np.int64)
            for idx, val in enumerate(series):
                if val not in account_map:
                    account_map[val] = len(account_map)
                res[idx] = account_map[val]
            return res

        from_ids = map_accounts(from_str)
        to_ids = map_accounts(to_str)

        recv_curr = df_input['Receiving Currency'].astype(str).str.lower().map(lambda c: REV_CURRENCY_MAP.get(c, 0)).values.astype(np.int64)
        sent_curr = df_input['Payment Currency'].astype(str).str.lower().map(lambda c: REV_CURRENCY_MAP.get(c, 0)).values.astype(np.int64)
        pay_fmt = df_input['Payment Format'].astype(str).str.lower().map(lambda f: REV_FORMAT_MAP.get(f, 3)).values.astype(np.int64)

        try:
            timestamps_dt = pd.to_datetime(df_input['Timestamp'])
            start_ts = timestamps_dt.min().timestamp() - 10
            ts_seconds = (timestamps_dt.astype('int64') // 10**9) - int(start_ts)
        except Exception:
            ts_seconds = np.arange(len(df_input), dtype=np.int64) * 60

        amt_recv = df_input['Amount Received'].astype(float).values
        amt_paid = df_input['Amount Paid'].astype(float).values if 'Amount Paid' in df_input.columns else amt_recv
        is_laund = df_input['Is Laundering'].astype(int).values if 'Is Laundering' in df_input.columns else np.zeros(len(df_input), dtype=int)

        formatted_df = pd.DataFrame({
            'EdgeID': np.arange(len(df_input)),
            'from_id': from_ids,
            'to_id': to_ids,
            'Timestamp': ts_seconds,
            'Amount Sent': amt_paid,
            'Sent Currency': sent_curr,
            'Amount Received': amt_recv,
            'Received Currency': recv_curr,
            'Payment Format': pay_fmt,
            'Is Laundering': is_laund
        })
    elif 'from_id' in cols and 'to_id' in cols:
        print("Detected formatted transaction format.")
        formatted_df = df_input.copy()
        if 'EdgeID' not in formatted_df.columns:
            formatted_df['EdgeID'] = np.arange(len(formatted_df))
        if 'Is Laundering' not in formatted_df.columns:
            formatted_df['Is Laundering'] = 0
    else:
        raise ValueError("Unrecognized CSV columns. CSV must contain either ['From Bank', 'Account', 'To Bank', 'Amount Received'] or ['from_id', 'to_id', 'Amount Received'].")

    # 2. Build PyG Graph Data
    max_id = int(max(formatted_df['from_id'].max(), formatted_df['to_id'].max()) + 1)
    x = torch.ones((max_id, 1), dtype=torch.float)
    edge_index = torch.tensor(formatted_df[['from_id', 'to_id']].values.T, dtype=torch.long)
    timestamps = torch.tensor(formatted_df['Timestamp'].values, dtype=torch.float)
    edge_attr_raw = torch.tensor(formatted_df[['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']].values, dtype=torch.float)

    gdata = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr_raw, timestamps=timestamps)
    gdata.add_ports()
    gdata.edge_attr = z_norm(gdata.edge_attr)
    gdata.x = z_norm(gdata.x)

    class DummyArgs:
        ports = True

    hdata = create_hetero_obj(gdata.x, torch.zeros(len(formatted_df)), gdata.edge_index, gdata.edge_attr, gdata.timestamps, DummyArgs())

    # 3. Model Inference
    base_model = GINe(num_features=1, num_gnn_layers=2, n_classes=2, n_hidden=66, edge_updates=True, edge_dim=6, dropout=0.01, final_dropout=0.1)
    model = to_hetero(base_model, hdata.metadata(), aggr='mean')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in hdata.x_dict.items()}
        e_idx = {k: v.to(device) for k, v in hdata.edge_index_dict.items()}
        e_attr = {k: v.to(device) for k, v in hdata.edge_attr_dict.items()}
        out = model(x_dict, e_idx, e_attr)
        probs = F.softmax(out[('node', 'to', 'node')], dim=-1)[:, 1].cpu().numpy()

    formatted_df['AML_Risk_Score'] = probs
    formatted_df['Predicted_Laundering'] = (probs >= 0.35).astype(int)
    formatted_df['Format_Name'] = formatted_df['Payment Format'].map(lambda x: FORMAT_MAP.get(int(x), f"Format {x}"))
    formatted_df['Currency_Name'] = formatted_df['Received Currency'].map(lambda x: CURRENCY_MAP.get(int(x), f"Curr {x}"))
    formatted_df['Risk_Percent'] = (formatted_df['AML_Risk_Score'] * 100).round(2)

    elapsed = time.perf_counter() - t0
    print(f"GNN Inference completed in {elapsed:.2f}s for {len(formatted_df):,} transactions.")
    return formatted_df, elapsed

@app.on_event("startup")
def startup_event():
    load_default_data()

@app.get("/api/stats")
def get_stats():
    global df_alerts, account_stats, active_dataset_info
    if df_alerts is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    flagged_df = df_alerts[df_alerts['AML_Risk_Score'] >= 0.35]
    total_flagged_vol = float(flagged_df['Amount Received'].sum())
    suspicious_mules = [acc for acc, s in account_stats.items() if s['is_suspicious']]

    return {
        "total_transactions": len(df_alerts),
        "flagged_transactions": len(flagged_df),
        "total_flagged_volume": total_flagged_vol,
        "suspicious_accounts_count": len(suspicious_mules),
        "top_precision_pct": 96.0,
        "overall_precision_pct": 54.21,
        "dataset_name": active_dataset_info["name"],
        "is_custom": active_dataset_info["is_custom"],
        "inference_time": active_dataset_info["inference_time"],
        "model_architecture": "Multi-GNN (GINe + Edge MLPs + Ports + Reverse MP)"
    }

@app.post("/api/upload")
async def upload_dataset(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Query(False)
):
    """Uploads a CSV dataset, runs Multi-GNN inference, and updates detection results."""
    global df_alerts, account_stats, active_dataset_info

    try:
        if use_sample:
            sample_path = "sample_dataset/sample_aml_test_batch.csv"
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample dataset not found on server")
            df_raw = pd.read_csv(sample_path)
            filename = "sample_aml_test_batch.csv (1,797 Transactions)"
        else:
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded and use_sample is false")
            filename = file.filename
            contents = await file.read()
            df_raw = pd.read_csv(io.BytesIO(contents))

        # Check if already has pre-scored predictions or needs GNN forward pass
        if 'AML_Risk_Score' in df_raw.columns:
            t0 = time.perf_counter()
            df_scored = df_raw.copy()
            df_scored['Format_Name'] = df_scored['Payment Format'].map(lambda x: FORMAT_MAP.get(int(x), f"Format {x}"))
            df_scored['Currency_Name'] = df_scored['Received Currency'].map(lambda x: CURRENCY_MAP.get(int(x), f"Curr {x}"))
            df_scored['Risk_Percent'] = (df_scored['AML_Risk_Score'] * 100).round(2)
            elapsed = time.perf_counter() - t0
        else:
            # Run full live GNN inference
            df_scored, elapsed = run_gnn_inference_on_df(df_raw)

        # Update active state
        df_alerts = df_scored
        account_stats = compute_account_stats(df_scored)
        active_dataset_info = {
            "name": filename,
            "is_custom": True,
            "total_tx": len(df_scored),
            "inference_time": f"{elapsed:.2f}s"
        }

        flagged_count = int((df_scored['AML_Risk_Score'] >= 0.35).sum())
        susp_mules = sum(1 for s in account_stats.values() if s['is_suspicious'])

        return {
            "status": "success",
            "message": f"Successfully processed {len(df_scored):,} transactions",
            "filename": filename,
            "total_transactions": len(df_scored),
            "flagged_transactions": flagged_count,
            "suspicious_mules_count": susp_mules,
            "inference_time_sec": round(elapsed, 3),
            "max_risk_detected": round(float(df_scored['AML_Risk_Score'].max()) * 100, 2)
        }

    except Exception as e:
        print("Upload Error:", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/reset_default")
def reset_to_default_dataset():
    """Resets the dashboard back to the default baseline dataset."""
    load_default_data()
    return {"status": "success", "message": "Reset to default IBM AML test dataset."}

@app.get("/api/mules")
def get_top_mules(limit: int = 15):
    """Returns ranked list of most suspicious accounts (money mules)."""
    global account_stats
    susp = [info for info in account_stats.values() if info['is_suspicious']]
    susp.sort(key=lambda x: (x['flagged_out'] + x['flagged_in'], x['flagged_volume']), reverse=True)
    return susp[:limit]

@app.get("/api/graph")
def get_graph(
    threshold: float = Query(0.35, ge=0.0, le=1.0),
    account_id: Optional[int] = None,
    limit: int = Query(800, ge=10, le=2000),
    pattern: str = Query("dense_core")
):
    """Returns nodes and edges formatted for interactive dense network visualization."""
    global df_alerts, account_stats, precomputed_subgraphs
    if df_alerts is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    if account_id is not None:
        direct_edges = df_alerts[(df_alerts['from_id'] == account_id) | (df_alerts['to_id'] == account_id)].copy()
        counterparties = set(direct_edges['from_id']).union(set(direct_edges['to_id']))
        ego_edges = df_alerts[df_alerts['from_id'].isin(counterparties) & df_alerts['to_id'].isin(counterparties)].copy()
        edges_df = ego_edges.sort_values(by=['AML_Risk_Score', 'Amount Received'], ascending=[False, False]).head(limit)
    else:
        if pattern in precomputed_subgraphs:
            edges_df = precomputed_subgraphs[pattern].copy()
        elif pattern == "all":
            edges_df = df_alerts[df_alerts['AML_Risk_Score'] >= threshold].copy()
        else:
            edges_df = precomputed_subgraphs.get('dense_core', df_alerts.head(600)).copy()
            
        edges_df = edges_df.sort_values(by=['AML_Risk_Score', 'Amount Received'], ascending=[False, False]).head(limit)

    node_ids = set(edges_df['from_id']).union(set(edges_df['to_id']))
    
    nodes = []
    for nid in node_ids:
        s = account_stats.get(nid, {
            'account_id': nid,
            'is_suspicious': False,
            'max_risk': 0.0,
            'role': 'Normal Counterparty',
            'in_volume': 0.0,
            'out_volume': 0.0,
            'flagged_out': 0,
            'flagged_in': 0
        })

        is_susp = s['is_suspicious'] or (s['max_risk'] >= threshold)
        
        if is_susp:
            if 'Mule' in s.get('role', ''):
                node_color = {
                    'background': '#ef4444',
                    'border': '#ffffff',
                    'highlight': {'background': '#dc2626', 'border': '#ffffff'},
                    'hover': {'background': '#f87171', 'border': '#ffffff'}
                }
                size = 30
            else:
                node_color = {
                    'background': '#e11d48',
                    'border': '#fda4af',
                    'highlight': {'background': '#be123c', 'border': '#ffffff'},
                    'hover': {'background': '#fb7185', 'border': '#ffffff'}
                }
                size = 24
            shadow = {
                'enabled': True,
                'color': 'rgba(239, 68, 68, 0.85)',
                'size': 16,
                'x': 0,
                'y': 0
            }
            label = f"🚨 Acc #{nid}"
        else:
            node_color = {
                'background': '#1e293b',
                'border': '#38bdf8',
                'highlight': {'background': '#0284c7', 'border': '#ffffff'},
                'hover': {'background': '#0369a1', 'border': '#38bdf8'}
            }
            size = 14
            shadow = {
                'enabled': True,
                'color': 'rgba(0, 0, 0, 0.4)',
                'size': 4,
                'x': 1,
                'y': 1
            }
            label = f"#{nid}"

        nodes.append({
            'id': nid,
            'label': label,
            'title': f"Account #{nid}\nRole: {s.get('role', 'Normal')}\nMax Risk: {s['max_risk']*100:.1f}%\nOut: ${s['out_volume']:,.2f}\nIn: ${s['in_volume']:,.2f}",
            'is_suspicious': is_susp,
            'role': s.get('role', 'Normal Counterparty'),
            'max_risk': round(s['max_risk'] * 100, 2),
            'in_volume': round(s['in_volume'], 2),
            'out_volume': round(s['out_volume'], 2),
            'flagged_count': s.get('flagged_out', 0) + s.get('flagged_in', 0),
            'color': node_color,
            'size': size,
            'shadow': shadow,
            'font': {'color': '#f8fafc', 'face': 'Inter', 'size': 10 if not is_susp else 12, 'strokeWidth': 2, 'strokeColor': '#090d16'}
        })

    edges = []
    for _, r in edges_df.iterrows():
        risk = float(r['AML_Risk_Score'])
        amt = float(r['Amount Received'])
        is_susp = risk >= threshold
        
        amt_str = f"${amt/1000:.1f}K" if amt >= 1000 else f"${amt:.0f}"

        if is_susp:
            edge_color = {
                'color': '#ef4444',
                'highlight': '#f87171',
                'hover': '#fca5a5'
            }
            width = 2.4
            edge_label = f"{amt_str} [{risk*100:.1f}%]"
            font_opt = {'color': '#fca5a5', 'size': 9, 'face': 'Inter', 'background': '#180709', 'strokeWidth': 0}
        else:
            edge_color = {
                'color': 'rgba(71, 85, 105, 0.35)',
                'highlight': '#38bdf8',
                'hover': '#94a3b8'
            }
            width = 1.0
            edge_label = ""
            font_opt = {'size': 0}

        edges.append({
            'id': int(r['EdgeID']),
            'from': int(r['from_id']),
            'to': int(r['to_id']),
            'label': edge_label,
            'title': f"Tx #{int(r['EdgeID'])}: ${amt:,.2f} ({r['Format_Name']}) - Risk {risk*100:.1f}%",
            'amount': amt,
            'currency': r['Currency_Name'],
            'format': r['Format_Name'],
            'risk': round(risk * 100, 2),
            'is_suspicious': is_susp,
            'color': edge_color,
            'width': width,
            'arrows': {'to': {'enabled': True, 'scaleFactor': 0.7 if not is_susp else 0.9}},
            'font': font_opt
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "suspicious_nodes_count": sum(1 for n in nodes if n['is_suspicious']),
        "filter_threshold": threshold
    }

@app.get("/api/node/{account_id}")
def get_node_details(account_id: int):
    """Detailed profile when clicking a node in the graph."""
    global df_alerts, account_stats
    if df_alerts is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    info = account_stats.get(account_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    connected = df_alerts[(df_alerts['from_id'] == account_id) | (df_alerts['to_id'] == account_id)].copy()
    connected = connected.sort_values(by=['AML_Risk_Score', 'Timestamp'], ascending=[False, False])
    
    trans_list = []
    for _, r in connected.head(30).iterrows():
        is_outgoing = int(r['from_id']) == account_id
        trans_list.append({
            'edge_id': int(r['EdgeID']),
            'direction': 'OUTGOING' if is_outgoing else 'INCOMING',
            'counterparty': int(r['to_id'] if is_outgoing else r['from_id']),
            'amount': float(r['Amount Received']),
            'currency': r['Currency_Name'],
            'format': r['Format_Name'],
            'risk_percent': float(r['Risk_Percent']),
            'is_laundering': bool(r['Is Laundering']),
            'timestamp': int(r['Timestamp'])
        })

    return {
        "account_id": account_id,
        "role": info['role'],
        "is_suspicious": info['is_suspicious'],
        "max_risk_percent": round(info['max_risk'] * 100, 2),
        "in_volume": round(info['in_volume'], 2),
        "out_volume": round(info['out_volume'], 2),
        "in_count": info['in_count'],
        "out_count": info['out_count'],
        "flagged_in": info['flagged_in'],
        "flagged_out": info['flagged_out'],
        "counterparty_count": info['counterparty_count'],
        "transactions": trans_list
    }

@app.get("/api/alerts")
def get_alerts(
    min_risk: float = Query(0.35, ge=0.0, le=1.0),
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100)
):
    """Paginated list of alerts for the bottom alert feed."""
    global df_alerts
    if df_alerts is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    df = df_alerts[df_alerts['AML_Risk_Score'] >= min_risk].copy()
    if len(df) == 0:
        df = df_alerts.sort_values(by='AML_Risk_Score', ascending=False).head(50)
    
    if search:
        search = search.strip()
        if search.isdigit():
            target_id = int(search)
            df = df[(df['from_id'] == target_id) | (df['to_id'] == target_id) | (df['EdgeID'] == target_id)]

    df = df.sort_values(by='AML_Risk_Score', ascending=False)
    total_count = len(df)
    
    start_idx = (page - 1) * page_size
    page_df = df.iloc[start_idx : start_idx + page_size]

    records = []
    for _, r in page_df.iterrows():
        records.append({
            'edge_id': int(r['EdgeID']),
            'from_id': int(r['from_id']),
            'to_id': int(r['to_id']),
            'amount': float(r['Amount Received']),
            'currency': r['Currency_Name'],
            'format': r['Format_Name'],
            'risk_percent': float(r['Risk_Percent']),
            'is_laundering': bool(r['Is Laundering']),
            'timestamp': int(r['Timestamp'])
        })

    return {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "alerts": records
    }

app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("web/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
