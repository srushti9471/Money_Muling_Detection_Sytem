import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

def format_kaggle_file(in_path, out_path=None):
    if not os.path.exists(in_path):
        print(f"Error: input file {in_path} does not exist.")
        sys.exit(1)

    if out_path is None:
        base_name = os.path.basename(in_path)
        dir_name = os.path.dirname(in_path)
        # E.g. HI-Small_Trans.csv -> folder HI-Small
        dataset_name = base_name.replace("_Trans.csv", "").replace(".csv", "")
        target_dir = os.path.join(dir_name, dataset_name)
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, "formatted_transactions.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"Reading {in_path} ...")
    df = pd.read_csv(in_path, dtype=str)
    print(f"Loaded {len(df):,} transactions. Processing features...")

    # Account ID mapping (shared between From Bank + Account and To Bank + Account)
    account_map = {}
    from_acc_str = df['From Bank'] + df.iloc[:, 2]
    to_acc_str = df['To Bank'] + df.iloc[:, 4]

    def map_accounts(series):
        res = np.empty(len(series), dtype=np.int64)
        for idx, val in enumerate(series):
            if val not in account_map:
                account_map[val] = len(account_map)
            res[idx] = account_map[val]
        return res

    from_ids = map_accounts(from_acc_str)
    to_ids = map_accounts(to_acc_str)

    # Currencies (shared mapping)
    currency_map = {}
    def map_currency(series):
        res = np.empty(len(series), dtype=np.int64)
        for idx, val in enumerate(series):
            if val not in currency_map:
                currency_map[val] = len(currency_map)
            res[idx] = currency_map[val]
        return res

    received_curr = map_currency(df['Receiving Currency'])
    sent_curr = map_currency(df['Payment Currency'])

    # Payment formats
    format_map = {}
    payment_formats = np.empty(len(df), dtype=np.int64)
    for idx, val in enumerate(df['Payment Format']):
        if val not in format_map:
            format_map[val] = len(format_map)
        payment_formats[idx] = format_map[val]

    # Timestamp conversion (seconds relative to start of first day)
    print("Converting timestamps...")
    timestamps_dt = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    first_dt = timestamps_dt.iloc[0]
    start_time = datetime(first_dt.year, first_dt.month, first_dt.day).timestamp() - 10
    ts_seconds = (timestamps_dt.astype('int64') // 10**9) - int(start_time)

    # Amounts & Labels
    amount_received = df['Amount Received'].astype(float).values
    amount_paid = df['Amount Paid'].astype(float).values
    is_laundering = df['Is Laundering'].astype(int).values

    out_df = pd.DataFrame({
        'EdgeID': np.arange(len(df)),
        'from_id': from_ids,
        'to_id': to_ids,
        'Timestamp': ts_seconds,
        'Amount Sent': amount_paid,
        'Sent Currency': sent_curr,
        'Amount Received': amount_received,
        'Received Currency': received_curr,
        'Payment Format': payment_formats,
        'Is Laundering': is_laundering
    })

    print("Sorting transactions chronologically...")
    out_df.sort_values(by='Timestamp', inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    out_df['EdgeID'] = np.arange(len(out_df))

    print(f"Saving formatted dataset to {out_path} ...")
    out_df.to_csv(out_path, index=False)
    print(f"Done! Formatted dataset successfully saved ({len(out_df):,} rows).")
    return out_path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python format_kaggle_files.py <path_to_raw_csv> [optional_output_path]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    format_kaggle_file(in_path, out_path)