"""
funnel_analysis.py
------------------
Acquisition funnel analysis: step-by-step conversion rates,
drop-off identification, and segment-level breakdowns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

FUNNEL_STEPS = [
    "page_view", "listing_view", "signup_start",
    "signup_complete", "add_to_cart", "checkout_start", "purchase_complete"
]
STEP_LABELS = [
    "Page View", "Listing View", "Signup Start",
    "Signup Complete", "Add to Cart", "Checkout Start", "Purchase"
]
COLORS = ["#1B4F72", "#2874A6", "#2E86C1", "#3498DB", "#5DADE2", "#85C1E9", "#AED6F1"]


def compute_funnel(events: pd.DataFrame, segment_col: str = None) -> pd.DataFrame:
    """
    Compute funnel conversion rates at each step.

    Parameters
    ----------
    events : pd.DataFrame with columns [user_id, event_type]
    segment_col : str, optional — column to segment by (e.g., 'device_type')

    Returns
    -------
    pd.DataFrame with funnel step counts and conversion rates.
    """
    if segment_col:
        groups = events.groupby(segment_col)
    else:
        groups = [("All Users", events)]

    results = []
    for segment_val, group in groups:
        user_reached = {}
        for step in FUNNEL_STEPS:
            users_at_step = group[group["event_type"] == step]["user_id"].nunique()
            user_reached[step] = users_at_step

        top = user_reached[FUNNEL_STEPS[0]]
        for i, step in enumerate(FUNNEL_STEPS):
            prev = user_reached[FUNNEL_STEPS[i - 1]] if i > 0 else user_reached[step]
            results.append({
                "segment": segment_val,
                "step": STEP_LABELS[i],
                "step_order": i,
                "users": user_reached[step],
                "pct_of_top": round(user_reached[step] / top * 100, 2) if top > 0 else 0,
                "pct_of_prev": round(user_reached[step] / prev * 100, 2) if prev > 0 else 0,
            })

    return pd.DataFrame(results)


def plot_funnel(funnel_df: pd.DataFrame, segment: str = "All Users", save_path: str = None):
    """Plot a waterfall-style funnel chart for a single segment."""
    data = funnel_df[funnel_df["segment"] == segment].sort_values("step_order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars = axes[0].barh(data["step"][::-1], data["users"][::-1], color=COLORS)
    axes[0].set_xlabel("Users")
    axes[0].set_title(f"Funnel Volume — {segment}", fontweight="bold")
    for bar, val in zip(bars, data["users"][::-1]):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=9)

    axes[1].barh(data["step"][::-1], data["pct_of_prev"][::-1], color=COLORS)
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_xlabel("Conversion from Previous Step (%)")
    axes[1].set_title(f"Step Conversion Rate — {segment}", fontweight="bold")
    for i, (_, row) in enumerate(data[::-1].iterrows()):
        axes[1].text(row["pct_of_prev"] + 0.5, i, f"{row['pct_of_prev']}%", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_funnel_by_segment(funnel_df: pd.DataFrame, save_path: str = None):
    """Compare purchase conversion rate across segments."""
    purchase_conv = funnel_df[funnel_df["step"] == "Purchase"].copy()
    purchase_conv = purchase_conv.sort_values("pct_of_top", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(purchase_conv) * 0.7)))
    bars = ax.barh(purchase_conv["segment"], purchase_conv["pct_of_top"],
                   color="#2E86C1", alpha=0.85)
    for bar, val in zip(bars, purchase_conv["pct_of_top"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=10)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Purchase Conversion Rate (% of page views)")
    ax.set_title("Purchase Conversion by Segment", fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    events = pd.read_csv("data/marketplace_events.csv", parse_dates=["event_timestamp"])

    print("Overall Funnel:")
    funnel = compute_funnel(events)
    print(funnel[funnel["segment"] == "All Users"][["step", "users", "pct_of_top", "pct_of_prev"]].to_string(index=False))
    plot_funnel(funnel)

    print("\nFunnel by Device Type:")
    funnel_device = compute_funnel(events, segment_col="device_type")
    print(funnel_device[funnel_device["step"] == "Purchase"][["segment", "users", "pct_of_top"]].to_string(index=False))
    plot_funnel_by_segment(funnel_device)
