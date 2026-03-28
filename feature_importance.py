"""
feature_importance.py
---------------------
Analyzes behavioral drivers of conversion and retention.
Identifies which user behaviors most strongly predict
purchasing, repeat buying, and long-term engagement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def build_conversion_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build features to predict whether a user converts to a first purchase.

    Parameters
    ----------
    events : pd.DataFrame with [user_id, event_type, event_timestamp, session_id, device_type, traffic_source]

    Returns
    -------
    pd.DataFrame with behavioral features and conversion label
    """
    events = events.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

    converters = set(events[events["event_type"] == "purchase_complete"]["user_id"].unique())

    features = (
        events.groupby("user_id")
        .agg(
            total_sessions=("session_id", "nunique"),
            total_events=("event_id", "count"),
            unique_days=("event_timestamp", lambda x: x.dt.date.nunique()),
            viewed_listing=("event_type", lambda x: (x == "listing_view").sum()),
            started_signup=("event_type", lambda x: (x == "signup_start").sum()),
            completed_signup=("event_type", lambda x: (x == "signup_complete").sum()),
            added_to_cart=("event_type", lambda x: (x == "add_to_cart").sum()),
            started_checkout=("event_type", lambda x: (x == "checkout_start").sum()),
            used_mobile=("device_type", lambda x: (x == "mobile").sum()),
            used_desktop=("device_type", lambda x: (x == "desktop").sum()),
            came_from_social=("traffic_source", lambda x: (x == "social").sum()),
            came_from_paid=("traffic_source", lambda x: (x == "paid_search").sum()),
            came_from_organic=("traffic_source", lambda x: (x == "organic").sum()),
        )
        .reset_index()
    )

    features["converted"] = features["user_id"].isin(converters).astype(int)
    features["mobile_pct"] = features["used_mobile"] / (features["total_events"] + 1)
    features["listing_per_session"] = features["viewed_listing"] / (features["total_sessions"] + 1)
    features["cart_per_listing"] = features["added_to_cart"] / (features["viewed_listing"] + 1)

    return features.fillna(0)


def plot_feature_importance(feature_cols: list, importances: np.ndarray, title: str, save_path: str = None):
    """Plot horizontal bar chart of feature importances."""
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_cols) * 0.45)))
    colors = ["#1B4F72" if v >= df["importance"].quantile(0.75) else "#85C1E9" for v in df["importance"]]
    ax.barh(df["feature"], df["importance"], color=colors, alpha=0.9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(df["importance"].mean(), color="red", linestyle="--",
               alpha=0.6, label=f"Mean: {df['importance'].mean():.3f}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(features: pd.DataFrame, feature_cols: list, target: str, save_path: str = None):
    """Plot correlation heatmap between features and target variable."""
    corr_data = features[feature_cols + [target]].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(
        corr_data,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def analyze_conversion_drivers(events: pd.DataFrame):
    """
    Full analysis pipeline: build features, train model,
    plot importance and correlations.
    """
    print("Building conversion features...")
    features = build_conversion_features(events)
    print(f"  Users: {len(features):,} | Conversion rate: {features['converted'].mean():.1%}")

    feature_cols = [
        "total_sessions", "total_events", "unique_days",
        "viewed_listing", "completed_signup", "added_to_cart",
        "started_checkout", "mobile_pct", "listing_per_session",
        "cart_per_listing", "came_from_social", "came_from_paid",
        "came_from_organic"
    ]

    X = features[feature_cols]
    y = features["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"\nRandom Forest ROC-AUC: {rf_auc:.4f}")

    plot_feature_importance(
        feature_cols,
        rf.feature_importances_,
        "Behavioral Drivers of Conversion — Random Forest"
    )

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    print(f"Gradient Boosting ROC-AUC: {gb_auc:.4f}")

    plot_feature_importance(
        feature_cols,
        gb.feature_importances_,
        "Behavioral Drivers of Conversion — Gradient Boosting"
    )

    plot_correlation_heatmap(features, feature_cols, "converted")

    return features, rf, gb


if __name__ == "__main__":
    events = pd.read_csv("data/marketplace_events.csv")
    features, rf, gb = analyze_conversion_drivers(events)
```

4. Scroll down and click **"Commit changes"**

---

That completes `user-behavior-analysis`. Your repo now has:
```
user-behavior-analysis/
├── generate_data.py
├── funnel_analysis.py
├── cohort_retention.py
├── churn_model.py
└── feature_importance.py
