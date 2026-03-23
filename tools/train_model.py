import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def train_model(file_path: str, target_column: str):
    try:
        file_path = file_path.strip().split("\n")[0]
        file_path = file_path.replace("'", "").replace('"', "")

        df = pd.read_csv(file_path)

        # Handle categorical data
        df = pd.get_dummies(df, drop_first=True)

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Detect problem type
        if y.nunique() <= 10:
            problem_type = "classification"
            metric_name = "Accuracy"
        else:
            problem_type = "regression"
            metric_name = "Mean Squared Error (MSE)"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}

        # Classification
        if problem_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = accuracy_score(y_test, preds)

            best_model = max(results, key=results.get)

        # Regression
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = mean_squared_error(y_test, preds)

            best_model = min(results, key=results.get)

        return {
            "problem_type": problem_type,
            "target": target_column,
            "metric": metric_name,
            "results": results,
            "best_model": best_model
        }

    except Exception as e:
        return {"error": str(e)}