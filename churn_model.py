"""
churn_model.py
--------------
Churn prediction model using logistic regression and random forest.
Trains on simulated marketplace data and outputs churn probabilities
and model performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)


CHURN_DAYS = 30


def build_features(orders: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for churn prediction from orders and events data.

    Parameters
    ----------
    orders : pd.DataFrame
    events : pd.DataFrame

    Returns
    -------
    pd.DataFrame with one row per user and engineered features + churn label
    """
    orders = orders.copy()
    events = events.copy()

    orders["created_at"] = pd.to_datetime(orders["created_at"])
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

    snapshot_date = orders["created_at"].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=CHURN_DAYS)

    # Order-based features (using data before cutoff only)
    order_features = (
        orders[orders["created_at"] <= cutoff_date]
        .groupby("buyer_id")
        .agg(
            total_orders=("order_id", "count"),
            total_gmv=("gmv", "sum"),
            avg_order_value=("gmv", "mean"),
            max_order_value=("gmv", "max"),
            unique_sellers=("seller_id", "nunique"),
            unique_categories=("category", "nunique"),
            first_order_date=("created_at", "min"),
            last_order_date=("created_at", "max"),
        )
        .reset_index()
        .rename(columns={"buyer_id": "user_id"})
    )

    order_features["days_since_first_order"] = (
        cutoff_date - order_features["first_order_date"]
    ).dt.days

    order_features["days_since_last_order"] = (
        cutoff_date - order_features["last_order_date"]
    ).dt.days

    order_features["order_frequency"] = (
        order_features["total_orders"] /
        (order_features["days_since_first_order"] + 1)
    )

    # Event-based features
    event_features = (
        events[events["event_timestamp"] <= cutoff_date]
        .groupby("user_id")
        .agg(
            total_sessions=("session_id", "nunique"),
            total_events=("event_id", "count"),
            unique_days_active=("event_timestamp", lambda x: x.dt.date.nunique()),
        )
        .reset_index()
    )

    # Churn label: did the user purchase after the cutoff?
    post_cutoff_buyers = set(
        orders[orders["created_at"] > cutoff_date]["buyer_id"].unique()
    )
    order_features["churned"] = (~order_features["user_id"].isin(post_cutoff_buyers)).astype(int)

    # Merge features
    features = order_features.merge(event_features, on="user_id", how="left")
    features = features.fillna(0)

    return features


def train_and_evaluate(features: pd.DataFrame):
    """
    Train logistic regression and random forest churn models.
    Print evaluation metrics and plot ROC curves.
    """
    feature_cols = [
        "total_orders", "total_gmv", "avg_order_value", "max_order_value",
        "unique_sellers", "unique_categories", "days_since_first_order",
        "days_since_last_order", "order_frequency", "total_sessions",
        "total_events", "unique_days_active"
    ]

    X = features[feature_cols]
    y = features["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    print("=" * 55)
    print("LOGISTIC REGRESSION")
    print("=" * 55)
    print(classification_report(y_test, lr_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")

    print("\n" + "=" * 55)
    print("RANDOM FOREST")
    print("=" * 55)
    print(classification_report(y_test, rf_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")

    # Plot ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, probs in [("Logistic Regression", lr_probs), ("Random Forest", rf_probs)]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves — Churn Models", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Feature importance from Random Forest
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=True)

    axes[1].barh(importance_df["feature"], importance_df["importance"], color="#2E86C1", alpha=0.85)
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title("Random Forest Feature Importance", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return lr, rf, scaler, feature_cols


if __name__ == "__main__":
    orders = pd.read_csv("data/orders.csv")
    events = pd.read_csv("data/marketplace_events.csv")

    print("Building features...")
    features = build_features(orders, events)
    print(f"Dataset: {len(features):,} users, churn rate: {features['churned'].mean():.1%}")

    print("\nTraining models...")
    lr, rf, scaler, feature_cols = train_and_evaluate(features)
