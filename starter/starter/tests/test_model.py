from ml.model import compute_model_metrics, train_model, inference


def test_train_model():
    pass


def test_compute_model_metrics():
    y_true = [0, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 0.5
    assert recall == 0.5
    assert fbeta == 0.5


def test_inference():
    pass
