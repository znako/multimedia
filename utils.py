from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def regression_cross_validate(model_class, X, y, n_folds=5, **model_params):
    """
    Perform cross-validation for a regression model and return average metrics and their standard deviations.

    Parameters:
    - model_class: Class of the model (e.g., sklearn.linear_model.LinearRegression).
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - n_folds: Number of folds for cross-validation (default: 5).
    - model_params: Parameters to initialize the model.

    Returns:
    - metrics_avg: Dictionary with average values of MAE, MSE, and R2.
    - metrics_std: Dictionary with standard deviations of MAE, MSE, and R2.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    mae_scores = []
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Compute averages and standard deviations
    metrics_avg = {
        'MAE': np.mean(mae_scores),
        'MSE': np.mean(mse_scores),
        'R2': np.mean(r2_scores)
    }

    metrics_std = {
        'MAE': np.std(mae_scores),
        'MSE': np.std(mse_scores),
        'R2': np.std(r2_scores)
    }

    return metrics_avg, metrics_std


def display_metrics_table(metrics_avg, metrics_std):
    """
    Display metrics in a formatted table.

    Parameters:
    - metrics_avg: Dictionary with average values of metrics.
    - metrics_std: Dictionary with standard deviations of metrics.
    """
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "R2"],
        "Mean": [metrics_avg["MAE"], metrics_avg["MSE"], metrics_avg["R2"]],
        "Std Dev": [metrics_std["MAE"], metrics_std["MSE"], metrics_std["R2"]]
    })
    print(metrics_df.to_markdown(index=False))


def classification_cross_validate(model_class, X, y, n_folds=5, **model_params):
    """
    Perform cross-validation for a regression model and return average metrics and their standard deviations.

    Parameters:
    - model_class: Class of the model (e.g., sklearn.linear_model.LinearRegression).
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - n_folds: Number of folds for cross-validation (default: 5).
    - model_params: Parameters to initialize the model.

    Returns:
    - metrics_avg: Dictionary with average values of Accuracy, Precision, Recall and F1-score.
    - metrics_std: Dictionary with standard deviations of Accuracy, Precision, Recall and F1-score.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        f1.append(f1_score(y_test, y_pred, average='weighted'))

    # Compute averages and standard deviations
    metrics_avg = {
        'Accuracy': np.mean(accuracy),
        'Precision': np.mean(precision),
        'Recall': np.mean(recall),
        'F1-score': np.mean(f1),
    }

    metrics_std = {
        'Accuracy': np.std(accuracy),
        'Precision': np.std(precision),
        'Recall': np.std(recall),
        'F1-score': np.std(f1),
    }

    return metrics_avg, metrics_std

def display_metrics_classification_table(metrics_avg, metrics_std):
    """
    Display metrics in a formatted table.

    Parameters:
    - metrics_avg: Dictionary with average values of metrics.
    - metrics_std: Dictionary with standard deviations of metrics.
    """
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Mean": [metrics_avg["Accuracy"], metrics_avg["Precision"], metrics_avg["Recall"], metrics_avg["F1-score"]],
        "Std Dev": [metrics_std["Accuracy"], metrics_std["Precision"], metrics_std["Recall"], metrics_std["F1-score"]]
    })
    print(metrics_df.to_markdown(index=False))
