import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Create an MLflow experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Classification_Experiments")


# 4. Define different hyperparameter combinations
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10}
]


# 5. Run experiments
for params in experiments:

    with mlflow.start_run():

        # Create Random Forest model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted"
        )

        # Log parameters
        mlflow.log_param(
            "n_estimators",
            params["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            params["max_depth"]
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            "random_forest_model"
        )

        # Display results
        print("Experiment completed")
        print("Parameters:", params)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("-" * 50)

# -----------------------------------------
# STEP 13: COMPARE ALL EXPERIMENTS
# -----------------------------------------

experiment = mlflow.get_experiment_by_name(
    "Iris_Classification_Experiments"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

print("\n========== MODEL COMPARISON ==========")

print(
    runs[
        [
            "run_id",
            "params.n_estimators",
            "params.max_depth",
            "metrics.accuracy",
            "metrics.precision",
            "metrics.recall"
        ]
    ].to_string(index=False)
)

# Select best model
best_run = runs.iloc[0]

print("\n========== BEST MODEL ==========")
print("Run ID:", best_run["run_id"])
print("n_estimators:", best_run["params.n_estimators"])
print("max_depth:", best_run["params.max_depth"])
print("Accuracy:", best_run["metrics.accuracy"])
print("Precision:", best_run["metrics.precision"])
print("Recall:", best_run["metrics.recall"])


# -----------------------------------------
# REGISTER BEST MODEL
# -----------------------------------------

best_run_id = best_run["run_id"]

model_uri = f"runs:/{best_run_id}/random_forest_model"

model_name = "Iris_RandomForest_Best_Model"

mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("\n========== MODEL REGISTRY ==========")
print("Best model registered successfully!")
print("Model name:", model_name)
print("Model URI:", model_uri)