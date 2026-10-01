"""PHẦN 5 - BÀI 1: dự báo tiền vé Titanic (fare) theo tuổi (age).

Lê Quyết Tiến - 2474802010386.
Chạy: py -3.13 bai1_titanic.py --age 30
Dataset đặt cạnh source: titanic.csv. Chỉ age được dùng làm đầu vào.
Chia train/test 60/40 để đánh giá; sau đó học lại toàn bộ dữ liệu hợp lệ
để dự báo mới. Không dùng mô hình học toàn bộ dữ liệu để báo RMSE test.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output" / "phan5" / "bai1"


def train_model(path=None):
    """Làm sạch hai cột cần dùng, đánh giá độc lập, trả mô hình để CLI/Web dùng."""
    raw = pd.read_csv(path or BASE_DIR / "titanic.csv")
    data = raw[["age", "fare"]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    # Không điền nhãn fare bị thiếu: không có nhãn thật để học/đánh giá.
    data = data[(data.age >= 0) & (data.fare >= 0)]
    if len(data) < 5 or data.age.nunique() < 2:
        raise ValueError("Cần ít nhất 5 mẫu hợp lệ và ít nhất 2 độ tuổi khác nhau.")
    train, test = train_test_split(data, test_size=0.4, random_state=1)
    evaluation_model = LinearRegression().fit(train[["age"]], train.fare)
    predicted = evaluation_model.predict(test[["age"]])
    metrics = {"rmse": float(np.sqrt(mean_squared_error(test.fare, predicted))),
               "r2": float(r2_score(test.fare, predicted))}
    model = LinearRegression().fit(data[["age"]], data.fare)
    return {"data": data, "model": model, "metrics": metrics,
            "removed": len(raw) - len(data), "train_size": len(train),
            "test_size": len(test), "test": test.assign(predicted_fare=predicted)}


def predict_fare(result, age):
    """Dự báo từ tuổi; ghi chú nếu phải ngoại suy ngoài khoảng tuổi đã quan sát."""
    age = float(age)
    if not np.isfinite(age) or not 0 <= age <= 120:
        raise ValueError("Tuổi phải là số hữu hạn từ 0 đến 120.")
    fare = float(result["model"].predict(pd.DataFrame({"age": [age]}))[0])
    low, high = result["data"].age.min(), result["data"].age.max()
    note = "Dự báo chỉ từ tuổi; giá vé còn phụ thuộc hạng vé và nhiều yếu tố khác."
    if not low <= age <= high:
        note += f" Đang ngoại suy ngoài khoảng tuổi {low:g}-{high:g} của dataset."
    return {"age": age, "fare": fare, "note": note}


def plot_result(result):
    """Vẽ điểm thật và đường hồi quy đã học trên toàn bộ dữ liệu hợp lệ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, model = result["data"], result["model"]
    grid = pd.DataFrame({"age": np.linspace(data.age.min(), data.age.max(), 100)})
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.scatter(data.age, data.fare, s=15, alpha=0.35, label="Dữ liệu thật")
    ax.plot(grid.age, model.predict(grid), color="#d34d29", label="Linear Regression")
    ax.set(xlabel="Tuổi (age)", ylabel="Tiền vé (fare, đơn vị gốc dataset)",
           title="Bài 1 - Dự báo tiền vé Titanic theo tuổi")
    ax.legend(); ax.grid(alpha=0.2)
    return fig


def save_results(result, age=30):
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result["test"].rename_axis("source_index").to_csv(OUTPUT / "du_bao_test.csv", encoding="utf-8-sig")
    prediction = predict_fare(result, age)
    model = result["model"]
    report = (
        "BÀI 1 - TITANIC: age -> fare\n"
        f"Mẫu hợp lệ: {len(result['data'])}; loại: {result['removed']}.\n"
        f"Train: {result['train_size']}; test: {result['test_size']}; random_state=1.\n"
        f"RMSE test: {result['metrics']['rmse']:.6f}; R2 test: {result['metrics']['r2']:.6f}.\n"
        f"Mô hình toàn bộ dữ liệu: fare = {model.intercept_:.6f} + {model.coef_[0]:.6f} * age.\n"
        f"Tuổi {age:g}: tiền vé dự báo {prediction['fare']:.6f}.\n{prediction['note']}\n"
        "NOTE: R2 gần 0 hoặc âm cho thấy tuổi riêng lẻ giải thích giá vé kém.\n"
    )
    (OUTPUT / "ket_qua.txt").write_text(report, encoding="utf-8")
    fig = plot_result(result); fig.savefig(OUTPUT / "titanic.png", dpi=140); plt.close(fig)
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age", type=float, default=30)
    args = parser.parse_args()
    try:
        result = train_model()
        print(save_results(result, args.age))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
