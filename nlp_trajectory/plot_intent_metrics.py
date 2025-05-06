#!/usr/bin/env python3
"""
plot_intent_metrics.py

Scan intent_model/ for CSV logs, plot each file's metrics over training steps,
and save the plots into intent_model/plots/ with matching filenames.
Any malformed lines (e.g. JSON dumps at the end) will be skipped.
The 'epoch' metric will not be plotted.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    log_dir   = "intent_model"
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # find all CSV logs in intent_model/
    for fname in os.listdir(log_dir):
        if not fname.lower().endswith(".csv"):
            continue

        csv_path = os.path.join(log_dir, fname)
        try:
            # skip any bad lines that don't match the header
            df = pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
            continue

        # ensure 'step' column exists and is numeric
        if "step" not in df.columns:
            print(f"Skipping {fname}: no 'step' column found.")
            continue
        df = df[pd.to_numeric(df["step"], errors="coerce").notnull()]

        if df.empty or df.shape[1] < 2:
            print(f"Skipping {fname}: not enough numeric columns to plot.")
            continue

        # plot each metric column on the same axes, skipping 'epoch'
        plt.figure()
        for col in df.columns:
            if col in ("step", "epoch"):
                continue
            elif col in ("step", "learning_rate"):
                continue
            plt.plot(df["step"], df[col], label=col)

        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.title(f"Metrics from {fname}")
        plt.legend()
        plt.tight_layout()

        # save the figure
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(plots_dir, f"{base}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    main()

