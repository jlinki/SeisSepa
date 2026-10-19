"""Protected metrics. All fitted references/probes use training samples only."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EVALUATION = {
    "version": 1,
    "objective": "mean validation event/station cosine top-1 accuracy against training gallery",
    "knn_k": 5,
    "silhouette_max_samples": 2048,
    "probe": "StandardScaler + LogisticRegression(C=1, solver=lbfgs, max_iter=2000)",
    "raw_spectra": "8192 normalized samples, log1p(rFFT power), mean pooled to 256 bins/channel",
    "confidence": "95% Wilson, z=1.959963984540054",
}


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be finite matrices")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def wilson(correct, n):
    if n < 1:
        raise ValueError("An accuracy needs at least one query")
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [0.0 if correct == 0 else float(max(0, center - half)),
            1.0 if correct == n else float(min(1, center + half))]


def accuracy(predicted, labels):
    correct = int(np.sum(np.asarray(predicted) == labels))
    return {"accuracy": correct / len(labels), "correct": correct, "n": len(labels),
            "wilson95": wilson(correct, len(labels))}


def nearest(ref, labels, query, leave_one_out=False, k=1):
    """Blocked cosine retrieval; stable gallery order resolves similarity ties."""
    ref, query = normalize(ref), normalize(query)
    if leave_one_out and len(ref) != len(query):
        raise ValueError("Leave-one-out requires aligned training references and queries")
    if len(ref) - int(leave_one_out) < k:
        raise ValueError("Not enough gallery samples for fixed k")
    predictions = []
    for start in range(0, len(query), 256):
        similarity = query[start:start + 256] @ ref.T
        if leave_one_out:
            similarity[np.arange(len(similarity)), start + np.arange(len(similarity))] = -np.inf
        indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
        neighbors = np.asarray(labels)[indices]
        for row in neighbors:
            classes, counts = np.unique(row, return_counts=True)
            predictions.append(classes[counts.argmax()])
    return np.asarray(predictions)


def retrieval_scores(train, validation):
    result = {}
    for scope, key in (("event", "ze"), ("station", "zs")):
        prediction = nearest(train[key], train[scope], validation[key])
        result[f"val/{scope}/retrieval_top1"] = accuracy(prediction, validation[scope])["accuracy"]
    result["val/retrieval_accuracy"] = float(np.mean(list(result.values())))
    return result


def reconstruction_metrics(target, prediction):
    # Averages over traces; R² and Pearson correlation use flattened channel/time vectors.
    x = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    y = np.asarray(prediction, dtype=np.float64).reshape(len(prediction), -1)
    residual = np.square(x - y).sum(1)
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    variance = np.square(xc).sum(1)
    r2 = np.where(variance > 1e-12, 1 - residual / np.maximum(variance, 1e-12),
                  np.where(residual <= 1e-12, 1.0, 0.0))
    denominator = np.sqrt(np.square(xc).sum(1) * np.square(yc).sum(1))
    correlation = (xc * yc).sum(1) / np.maximum(denominator, 1e-12)
    return {"mse": float(np.mean(np.square(x - y))), "r2": float(r2.mean()),
            "mean_correlation": float(correlation.mean())}


def silhouette(z, labels):
    count = min(len(labels), EVALUATION["silhouette_max_samples"])
    indices = np.random.default_rng(42).choice(len(labels), count, replace=False)
    selected = np.asarray(labels)[indices]
    if not 1 < len(np.unique(selected)) < count:
        return {"score": None, "n": count, "reason": "requires 2..n-1 classes"}
    return {"score": float(silhouette_score(normalize(z)[indices], selected, metric="cosine")), "n": count}


def probe(ref, labels, query):
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
    estimator.fit(ref, labels)
    return estimator.predict(query)


def full_metrics(train, validation, counts):
    result = {"objective": retrieval_scores(train, validation), "scopes": {}}
    for scope, key in (("event", "ze"), ("station", "zs")):
        tr_z, va_z = normalize(train[key]), normalize(validation[key])
        labels, queries = train[scope], validation[scope]
        classes = np.unique(labels)
        centroids = normalize(np.stack([tr_z[labels == c].mean(0) for c in classes]))
        item = {"chance": 1.0 / counts[scope], "classes": counts[scope],
                "validation_classes_present": len(np.unique(queries)), "training": {}, "validation": {}}
        for name, z, y, loo in (("training", tr_z, labels, True), ("validation", va_z, queries, False)):
            item[name]["retrieval_top1"] = accuracy(nearest(tr_z, labels, z, leave_one_out=loo), y)
            item[name]["nearest_centroid"] = accuracy(nearest(centroids, classes, z), y)
            item[name]["silhouette_cosine"] = silhouette(z, y)
        item["validation"]["knn5"] = accuracy(nearest(tr_z, labels, va_z, k=5), queries)
        item["validation"]["linear_probe"] = accuracy(probe(tr_z, labels, va_z), queries)
        item["validation"]["raw_spectra_linear_probe"] = accuracy(probe(train["spectra"], labels, validation["spectra"]), queries)
        result["scopes"][scope] = item
    result["reconstruction"] = {"training": train["reconstruction"], "validation": validation["reconstruction"]}
    result["counts"] = {"training": len(train["event"]), "validation": len(validation["event"]), **counts}
    return result


def markdown_report(result):
    rows = ["| Scope | Metric | Training | Validation | Reference |",
            "|---|---|---:|---:|---|"]
    for key, title, reference in (("mse", "MSE", "0 = perfect"), ("r2", "R²", "1 = perfect"),
                                   ("mean_correlation", "Mean correlation", "1 = perfect")):
        rec = result["reconstruction"]
        rows.append(f"| Reconstruction | {title} | {rec['training'][key]:.4f} | {rec['validation'][key]:.4f} | {reference} |")
    def value(record):
        if "accuracy" in record:
            low, high = record["wilson95"]
            return f"{record['accuracy']:.4f} [{low:.3f}–{high:.3f}]"
        return "undefined" if record["score"] is None else f"{record['score']:.4f}"
    for scope, item in result["scopes"].items():
        for key, title in (("retrieval_top1", "Retrieval top-1"), ("nearest_centroid", "Nearest-centroid accuracy"),
                           ("knn5", "k-NN accuracy (k=5)"), ("linear_probe", "Linear probe accuracy"),
                           ("silhouette_cosine", "Silhouette (cosine)")):
            training = value(item["training"][key]) if key in item["training"] else "—"
            validation = value(item["validation"][key])
            reference = f"chance {item['chance']:.4f}"
            if key == "linear_probe":
                reference += f"; raw spectra {item['validation']['raw_spectra_linear_probe']['accuracy']:.4f}"
            if key == "silhouette_cosine":
                reference = "−1…1; >0 separated"
            rows.append(f"| {scope.title()} ({'ze' if scope == 'event' else 'zs'}) | {title} | {training} | {validation} | {reference} |")
    counts = result["counts"]
    rows.extend(["", f"Traces: training {counts['training']}, validation {counts['validation']}; "
                 f"classes: {counts['event']} events, {counts['station']} stations. 95% Wilson intervals.",
                 "", f"**Validation retrieval accuracy: {result['objective']['val/retrieval_accuracy']:.6f}**",
                 "", "Training retrieval excludes each query from the gallery; training centroids include it.",
                 "R² = mean per-trace 1−SSE/SST. Silhouette uses at most 2,048 fixed-seed samples per split."])
    return "\n".join(rows) + "\n"
