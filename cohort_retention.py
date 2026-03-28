"""
cohort_retention.py
-------------------
Weekly cohort retention analysis.
Assigns users to cohorts by signup week and tracks
what % remain active in subsequent weeks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def build_retention_matrix(events: pd.DataFrame, users: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """
    Build a cohort retention matrix.

    Parameters
    ----------
    events : pd.DataFrame with [user_id, event_timestamp]
    users : pd.DataFrame with [user_id, signup_date]
    n_weeks : int — number of weeks to track retention

    Returns
    -------
    pd.DataFrame — pivot table: rows=cohort_week, cols=week_number, values=retention_rate
    """
    events = events.copy()
    users = users.copy()

    users["cohort_week"] = pd.to_datetime(users["signup_date"]).dt.to_period("W")
    events["activity_week"] = pd.to_datetime(events["event_timestamp"]).dt.to_period("W")

    merged = events.merge(users[["user_id", "cohort_week"]], on="user_id", how="left")
    merged["week_number"] = (merged["activity_week"] - merged["cohort_week"]).apply(
        lambda x: x.n if hasattr(x, "n") else np.nan
    )
    merged = merged[merged["week_number"].between(0, n_weeks)]

    activity = (
        merged.groupby(["cohort_week", "week_number"])["user_id"]
        .nunique()
        .reset_index(name="active_users")
    )

    cohort_sizes = activity[activity["week_number"] == 0].set_index("cohort_week")["active_users"]

    activity["cohort_size"] = activity["cohort_week"].map(cohort_sizes)
    activity["retention_rate"] = activity["active_users"] / activity["cohort_size"] * 100

    matrix = activity.pivot(index="cohort_week", columns="week_number", values="retention_rate")
    matrix.index = matrix.index.astype(str)
    return matrix.round(1)


def plot_retention_heatmap(matrix: pd.DataFrame, save_path: str = None):
    """Plot cohort retention as a heatmap."""
    fig, ax = plt.subplots(figsize=(14, max(6, len(matrix) * 0.55)))

    mask = matrix.isnull()
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd_r",
        linewidths=0.4,
        linecolor="white",
        mask=mask,
        vmin=0, vmax=100,
        ax=ax,
        cbar_kws={"label": "Retention Rate (%)"},
    )
    ax.set_title("Weekly Cohort Retention Heatmap (%)", fontweight="bold", pad=15, fontsize=14)
    ax.set_xlabel("Weeks Since Signup")
    ax.set_ylabel("Cohort (Signup Week)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_retention_curves(matrix: pd.DataFrame, n_cohorts: int = 6, save_path: str = None):
    """Plot retention curves for the most recent cohorts."""
    recent = matrix.tail(n_cohorts)
    palette = sns.color_palette("Blues_d", n_cohorts)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (cohort, row) in enumerate(recent.iterrows()):
        ax.plot(row.dropna().index, row.dropna().values,
                marker="o", linewidth=2, markersize=5,
                label=str(cohort), color=palette[i])

    ax.set_xlabel("Weeks Since Signup")
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title(f"Cohort Retention Curves — Last {n_cohorts} Cohorts", fontweight="bold")
    ax.legend(title="Cohort Week", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    events = pd.read_csv("data/marketplace_events.csv", parse_dates=["event_timestamp"])
    users = pd.read_csv("data/users.csv", parse_dates=["signup_date"])

    matrix = build_retention_matrix(events, users, n_weeks=12)
    print("Retention Matrix (first 4 cohorts):")
    print(matrix.head(4).to_string())

    plot_retention_heatmap(matrix)
    plot_retention_curves(matrix, n_cohorts=6)
