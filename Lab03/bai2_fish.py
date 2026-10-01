"""PHẦN 5 - BÀI 2: dự báo Width và Species từ Weight trong Fish.csv.

Lê Quyết Tiến - 2474802010386.
Chạy: py -3.13 bai2_fish.py --weight 300
Linear Regression dự báo chiều rộng; KNN dự báo loài (bài toán phân loại).
Cả hai chỉ nhận Weight, không dùng Width/Species thật của mẫu test làm đầu vào.
Loài không thể xác định chắc chắn từ trọng lượng; xem accuracy và ghi chú.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output" / "phan5" / "bai2"


def train_model(path=None):
    """Chia 60/40 có stratify để tập test chứa đại diện các loài."""
    raw = pd.read_csv(path or BASE_DIR / "Fish.csv")
    data = raw[["Weight", "Width", "Species"]].copy()
    for column in ["Weight", "Width"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["Species"] = data.Species.astype(str).str.strip()
    # Có một mẫu Weight=0: không hợp lệ cho một con cá có kích thước đo được.
    data = data[(data.Weight > 0) & (data.Width > 0) & data.Species.ne("")]
    train, test = train_test_split(data, test_size=0.4, random_state=1, stratify=data.Species)
    regression = LinearRegression().fit(train[["Weight"]], train.Width)
    classifier = KNeighborsClassifier(n_neighbors=5).fit(train[["Weight"]], train.Species)
    width_pred = regression.predict(test[["Weight"]])
    species_pred = classifier.predict(test[["Weight"]])
    metrics = {"rmse": float(np.sqrt(mean_squared_error(test.Width, width_pred))),
               "r2": float(r2_score(test.Width, width_pred)),
               "accuracy": float(accuracy_score(test.Species, species_pred))}
    # Sau đánh giá độc lập, học lại trên tất cả mẫu hợp lệ để dự báo mới.
    regression = LinearRegression().fit(data[["Weight"]], data.Width)
    classifier = KNeighborsClassifier(n_neighbors=5).fit(data[["Weight"]], data.Species)
    labels = sorted(data.Species.unique())
    matrix = pd.DataFrame(confusion_matrix(test.Species, species_pred, labels=labels),
                          index=labels, columns=labels)
    return {"data": data, "model": regression, "classifier": classifier,
            "metrics": metrics, "removed": len(raw)-len(data),
            "train_size": len(train), "test_size": len(test), "confusion": matrix,
            "test": test.assign(predicted_width=width_pred, predicted_species=species_pred)}


def predict_fish(result, weight):
    """Tỷ lệ láng giềng là mức bỏ phiếu KNN, không phải xác suất đã hiệu chỉnh."""
    weight = float(weight)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Trọng lượng phải là số hữu hạn lớn hơn 0.")
    sample = pd.DataFrame({"Weight": [weight]})
    species = str(result["classifier"].predict(sample)[0])
    votes = float(result["classifier"].predict_proba(sample).max())
    width = float(result["model"].predict(sample)[0])
    note = "Loài được dự đoán chỉ từ trọng lượng; nhiều loài có trọng lượng tương tự."
    if not result["data"].Weight.min() <= weight <= result["data"].Weight.max():
        note += " Trọng lượng nằm ngoài miền dữ liệu: kết quả đang ngoại suy."
    return {"weight": weight, "width": width, "species": species, "votes": votes, "note": note}


def plot_result(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for species, group in result["data"].groupby("Species"):
        ax.scatter(group.Weight, group.Width, s=25, alpha=0.7, label=species)
    grid = pd.DataFrame({"Weight": np.linspace(result["data"].Weight.min(), result["data"].Weight.max(), 100)})
    ax.plot(grid.Weight, result["model"].predict(grid), color="black", label="Hồi quy chung")
    ax.set(title="Bài 2 - Chiều rộng cá theo trọng lượng", xlabel="Weight (g)", ylabel="Width (cm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    return fig


def save_results(result, weight=300):
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result["test"].rename_axis("source_index").to_csv(OUTPUT / "du_bao_test.csv", encoding="utf-8-sig")
    result["confusion"].rename_axis("loai_thuc_te").to_csv(OUTPUT / "confusion_matrix.csv", encoding="utf-8-sig")
    prediction = predict_fish(result, weight)
    metrics = result["metrics"]
    report = (
        "BÀI 2 - FISH: Weight -> Width; Weight -> Species\n"
        f"Mẫu hợp lệ: {len(result['data'])}; loại: {result['removed']} (thiếu/không hợp lệ).\n"
        f"Train: {result['train_size']}; test: {result['test_size']}; stratify theo loài; seed=1.\n"
        f"RMSE Width: {metrics['rmse']:.6f}; R2: {metrics['r2']:.6f}.\n"
        f"Accuracy Species: {metrics['accuracy']:.2%}.\n"
        f"Weight={weight:g}: Width={prediction['width']:.6f}; Species dự đoán={prediction['species']}.\n"
        f"Tỷ lệ phiếu của 5 láng giềng: {prediction['votes']:.0%} (không phải độ chính xác).\n"
        f"{prediction['note']}\n"
        "NOTE: Width dùng hồi quy chung trên các loài; chưa phân biệt hình dạng riêng mỗi loài.\n"
    )
    (OUTPUT / "ket_qua.txt").write_text(report, encoding="utf-8")
    fig = plot_result(result); fig.savefig(OUTPUT / "fish.png", dpi=140); plt.close(fig)
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=300)
    args = parser.parse_args()
    try:
        print(save_results(train_model(), args.weight))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
