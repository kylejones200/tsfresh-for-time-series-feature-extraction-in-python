"""Generated from Jupyter notebook: TSFresh for Time Series Feature Extraction in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction.feature_calculators import mean
from tsfresh.utilities.dataframe_functions import impute


def create_a_larger_sample_dataset() -> None:
    np.random.seed(42)

    n_series = 100

    n_timepoints = 100

    time_series_list = []

    for i in range(n_series):
        frequency = np.random.uniform(0.5, 2)
        phase = np.random.uniform(0, 2 * np.pi)
        noise_level = np.random.uniform(0.05, 0.2)
        values = np.sin(
            frequency * np.linspace(0, 10, n_timepoints) + phase
        ) + np.random.normal(0, noise_level, n_timepoints)
        df = pd.DataFrame({"id": i, "time": range(n_timepoints), "value": values})
        time_series_list.append(df)

    time_series = pd.concat(time_series_list, ignore_index=True)

    print("Original time series data:")

    print(time_series.head())

    print(f"Number of time series: {n_series}")

    print(f"Number of timepoints per series: {n_timepoints}")

    plt.figure(figsize=(12, 6))

    for i in range(5):
        plt.plot(
            time_series[time_series["id"] == i]["time"],
            time_series[time_series["id"] == i]["value"],
            label=f"Series {i}",
        )

    plt.title("Sample of Time Series")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("sample_TS.png")

    plt.show()

    features = extract_features(
        time_series, column_id="id", column_sort="time", n_jobs=0
    )

    print("\nExtracted features:")

    print(features.head())

    features_imputed = impute(features)

    target = pd.Series(index=range(n_series), dtype=int)

    target[features_imputed.index % 2 == 0] = 0

    target[features_imputed.index % 2 == 1] = 1

    selected_features = select_features(features_imputed, target)

    if selected_features.empty:
        print("\nNo features were selected. Using all features.")
        selected_features = features_imputed
    else:
        print("\nSelected features:")
        print(selected_features.head())

    print(f"\nNumber of features: {selected_features.shape[1]}")

    print("\nNames of features (first 10):")

    print(selected_features.columns.tolist()[:10])

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        selected_features, target, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train_clf, y_train_clf)

    y_pred_clf = clf.predict(X_test_clf)

    print("\nClassification Model Performance:")

    print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.2f}")

    print("\nClassification Report:")

    print(classification_report(y_test_clf, y_pred_clf))

    cm = confusion_matrix(y_test_clf, y_pred_clf)

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")

    plt.show()

    feature_importance = pd.DataFrame(
        {"feature": X_train_clf.columns, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Most Important Features:")

    print(feature_importance.head(10))

    plt.figure(figsize=(12, 6))

    sns.barplot(x="importance", y="feature", data=feature_importance.head(20))

    plt.title("Top 20 Most Important Features")

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.savefig("feature_importance.png")

    plt.show()


def multivariate_feature_extraction() -> None:
    time_series["value2"] = time_series["value"] * 0.5 + np.random.normal(
        0, 0.05, len(time_series)
    )

    features_multivariate = extract_features(
        time_series,
        column_id="id",
        column_sort="time",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=0,
    )

    print("\nMultivariate features:")

    print(features_multivariate.head())


def compute_the_mean_for_each_time_series() -> None:
    custom_features = time_series.groupby("id")["value"].apply(mean)

    print("\nCustom features (mean of each time series, first 5):")

    print(custom_features.head())

    plt.figure(figsize=(10, 6))

    sns.histplot(custom_features, kde=True)

    plt.title("Distribution of Mean Values for Each Time Series")

    plt.xlabel("Mean Value")

    plt.ylabel("Count")

    plt.savefig("dist_of_means_TS.png")

    plt.show()

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=custom_features, y=target)

    plt.title("Relationship between Mean Values and Target")

    plt.xlabel("Mean Value")

    plt.ylabel("Target")

    plt.savefig("means_v_target_TS.png")

    plt.show()


def main() -> None:
    create_a_larger_sample_dataset()
    multivariate_feature_extraction()
    compute_the_mean_for_each_time_series()


if __name__ == "__main__":
    main()
