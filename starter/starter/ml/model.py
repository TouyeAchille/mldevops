from sklearn.metrics import fbeta_score, precision_score, recall_score

# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # create classifier model
    # clf_model=LogisticRegression(penalty="l2", C=1.0, n_jobs=-1,
    # random_state=42, solver="lbfgs",max_iter=100)

    clf_model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=False,
        random_state=42,
        early_stopping=True,
    )

    # Train model
    clf_model.fit(X_train, y_train)

    return clf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)

    return preds
