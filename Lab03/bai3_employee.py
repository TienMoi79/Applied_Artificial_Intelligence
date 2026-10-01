"""PHẦN 5 - BÀI 3: gom Employee_Data.csv thành 3 cụm.

Lê Quyết Tiến - 2474802010386.
Chạy: py -3.13 bai3_employee.py
Chọn đặc trưng: py -3.13 bai3_employee.py --columns Age Salary

NOTE: Employee_Data.csv chưa có trong dữ liệu được cung cấp lúc viết bài.
Không dùng dataset khác hoặc tự tạo dữ liệu giả để thay thế. Khi có file,
đặt cạnh source. Đề không chỉ định các cột; chương trình tự chọn cột số có
biến thiên, bỏ mã định danh phổ biến, và in chính xác các cột đã dùng.
Cần kiểm tra ý nghĩa cột; dùng --columns nếu muốn chọn đặc trưng cụ thể.

Các hàm gom cụm/xuất kết quả ở đây được dùng lại trong bài 4 và Web bài 5.
"""
import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output" / "phan5" / "bai3"


def is_identifier(column):
    """Mã nhân viên là định danh, khoảng cách giữa các mã không có ý nghĩa."""
    name = re.sub(r"[^a-z0-9]", "", column.lower())
    return name.startswith("unnamed") or name in {
        "id", "employeeid", "employeeidentifier", "employeenumber", "employeecode",
        "empid", "empno", "staffid", "number", "serialnumber", "index", "stt"}


def fit_clusters(raw, columns, dataset_name):
    """Điền thiếu bằng median -> StandardScaler -> KMeans(K=3).

    Không dùng train/test vì đây là gom nhóm các quan sát chưa có nhãn.
    Chuẩn hóa tránh một cột có đơn vị/độ lớn lớn hơn chi phối khoảng cách.
    Tâm được đổi về đơn vị gốc để có thể giải thích kết quả.
    """
    if len(columns) < 2 or len(set(columns)) != len(columns):
        raise ValueError("Cần ít nhất 2 cột đặc trưng khác nhau.")
    absent = set(columns) - set(raw.columns)
    if absent:
        raise ValueError(f"Không tìm thấy cột: {', '.join(sorted(absent))}.")
    features = raw[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Hàng thiếu toàn bộ đặc trưng không có cơ sở để phân cụm, nên bỏ.
    features = features.dropna(how="all")
    if features.empty or features.isna().all().any():
        raise ValueError("Dataset/cột đặc trưng không có giá trị số để gom cụm.")
    if features.nunique().lt(2).any():
        raise ValueError("Mỗi cột được chọn phải có ít nhất 2 giá trị khác nhau.")
    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(features)
    if len(np.unique(filled, axis=0)) < 3:
        raise ValueError("Cần ít nhất 3 quan sát khác nhau để tạo 3 cụm.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    # Bài 4 có gần 300.000 dòng: dùng triển khai thư viện để tính hiệu quả.
    # n_init=10 thử 10 khởi tạo và giữ nghiệm SSE nhỏ nhất, seed để tái lập.
    model = KMeans(n_clusters=3, n_init=10, random_state=123, max_iter=300, tol=1e-6)
    with threadpool_limits(limits=2):
        labels = model.fit_predict(scaled)
    errors = np.square(scaled - model.cluster_centers_[labels]).sum(axis=1)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=columns)
    centers.index = pd.Index([1, 2, 3], name="cluster")
    centers["count"] = np.bincount(labels, minlength=3)
    # Chỉ lưu đặc trưng đã xử lý và vị trí dòng CSV gốc để dễ đối chiếu.
    result = pd.DataFrame(filled, columns=columns)
    result.insert(0, "source_csv_row", features.index.to_numpy() + 2)
    result["cluster"] = labels + 1  # Đổi nhãn 0..2 thành 1..3 cho người đọc.
    result["squared_distance_scaled"] = errors
    return {"data": result, "centers": centers, "model": model, "scaler": scaler,
            "imputer": imputer, "columns": columns, "dataset": dataset_name,
            "raw_size": len(raw), "removed": len(raw)-len(features),
            "missing_filled": int(features.isna().sum().sum()),
            "inertia": float(errors.sum()), "iterations": int(model.n_iter_)}


def train_model(path=None, columns=None):
    path = Path(path) if path else BASE_DIR / "Employee_Data.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Thiếu {path.name}. Đặt Employee_Data.csv cạnh source rồi chạy lại bài 3. "
            "Không thay thế bằng dataset lương của bài 4.")
    raw = pd.read_csv(path)
    if columns is None:
        columns = [name for name in raw.select_dtypes(include="number").columns
                   if not is_identifier(name) and raw[name].nunique() > 1]
    return fit_clusters(raw, columns, path.name)


def predict_cluster(result, values):
    """Gán một mẫu mới vào tâm đã học; dùng đúng imputer/scaler khi huấn luyện."""
    sample = pd.DataFrame([values], columns=result["columns"])
    if not np.isfinite(sample.to_numpy(dtype=float)).all():
        raise ValueError("Các đặc trưng mới phải là số hữu hạn.")
    scaled = result["scaler"].transform(result["imputer"].transform(sample))
    with threadpool_limits(limits=2):
        return int(result["model"].predict(scaled)[0]) + 1


def plot_result(result):
    """Hình chiếu hai cột đầu; tối đa 6.000 điểm, mô hình vẫn học toàn bộ dữ liệu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = result["data"]
    view = data.sample(min(6000, len(data)), random_state=123)
    x, y = result["columns"][:2]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for label, color in [(1, "#226bb5"), (2, "#d45b24"), (3, "#288148")]:
        group = view[view.cluster == label]
        count = int(result["centers"].loc[label, "count"])
        ax.scatter(group[x], group[y], s=12, alpha=0.4, color=color, label=f"Cụm {label}: {count:,} mẫu")
        center = result["centers"].loc[label]
        ax.scatter(center[x], center[y], color=color, marker="X", s=180, edgecolors="black")
    subtitle = f"Hiển thị {len(view):,}/{len(data):,} điểm; tâm ở đơn vị gốc"
    if len(result["columns"]) > 2: subtitle += "; hình chiếu 2D"
    ax.set(title=f"K-Means: {result['dataset']}\n{subtitle}", xlabel=x, ylabel=y)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    return fig


def save_results(result, output=OUTPUT):
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result["data"].to_csv(output / "ket_qua_gom_cum.csv", index=False, encoding="utf-8-sig")
    result["centers"].to_csv(output / "tam_cum.csv", encoding="utf-8-sig")
    report = (
        f"GOM 3 CỤM: {result['dataset']}\n"
        f"Đặc trưng: {', '.join(result['columns'])}.\n"
        f"Dòng gốc: {result['raw_size']}; dùng: {len(result['data'])}; loại: {result['removed']}.\n"
        f"Ô thiếu được điền median: {result['missing_filled']}.\n"
        f"StandardScaler; KMeans K=3, n_init=10, seed=123.\n"
        f"SSE trong không gian chuẩn hóa: {result['inertia']:.6f}.\n"
        f"Số vòng của lần khởi tạo được chọn: {result['iterations']}.\n"
        f"Tâm trong đơn vị gốc và số mẫu:\n{result['centers'].to_string()}\n"
        "NOTE: Nhãn 1,2,3 là tên nhóm, không phải thứ hạng hay mức đánh giá nhân viên.\n"
        "NOTE: Gom cụm không có accuracy vì dataset không có nhãn cụm thật.\n"
        "NOTE: SSE tính trên dữ liệu chuẩn hóa; không so trực tiếp với SSE dùng đơn vị gốc.\n"
    )
    if result["iterations"] >= 300:
        report += "NOTE: Đã đạt giới hạn 300 vòng; cần xem xét tăng max_iter.\n"
    (output / "ket_qua.txt").write_text(report, encoding="utf-8")
    fig = plot_result(result); fig.savefig(output / "gom_cum.png", dpi=140); plt.close(fig)
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", nargs="+", help="Các cột số dùng làm đặc trưng.")
    parser.add_argument("--data", type=Path, help="Đường dẫn Employee_Data.csv nếu nằm nơi khác.")
    args = parser.parse_args()
    try:
        print(save_results(train_model(args.data, args.columns)))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Chưa chạy được bài 3: {error}\n")
