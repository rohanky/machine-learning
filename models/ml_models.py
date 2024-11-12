from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score, average_precision_score, confusion_matrix

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name="svm"):
    """
    Train a specified sklearn model based on the model_name flag and evaluate it.
    
    Parameters:
    - X_train, y_train: Training data and labels
    - X_test, y_test: Testing data and labels
    - model_name (str): Name of the model to train. Options are "svm", "random_forest", 
                        "logistic_regression", "knn".
    
    Returns:
    - model: The trained model
    - metrics: A dictionary of evaluation metrics
    """ 

    # Initialize the model based on the flag
    if model_name == "svm":
        model = SVC()
    elif model_name == "random_forest":
        model = RandomForestClassifier()
    elif model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "knn":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Invalid model_name. Choose from 'svm', 'random_forest', 'logistic_regression', or 'knn'.")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate evaluation metrics
# Calculate evaluation metrics with a focus on imbalanced data
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1_score": f1_score(y_test, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_pred),


    }
    print()
    print(conf_matrix)
    print()
    return model, metrics
