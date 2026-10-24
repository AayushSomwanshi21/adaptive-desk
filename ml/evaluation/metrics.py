from sklearn.metrics import accuracy_score, f1_score, classification_report


def calculate_metrics(y_true, y_pred, class_names=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")

    print(
        "\nClassification report:\n",
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))) if class_names else None,
            target_names=class_names,
            zero_division=0,
        ),
    )

    return results
