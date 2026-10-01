"""PHẦN 3 - HỌC KHÔNG GIÁM SÁT: tự cài đặt K-Means (mục 3.1-3.6).

Chạy: py -3.13 phan3_kmeans.py
Xem cửa sổ biểu đồ: py -3.13 phan3_kmeans.py --show

Dataset: kmeans_data.csv, 100 điểm x, y tạo như mục 3.2 với seed 123.
Không có nhãn Y để học; nhãn cụm 0, 1, 2 do thuật toán tự gán.
Không dùng sklearn.cluster.KMeans; NumPy/Pandas chỉ hỗ trợ tính toán.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "kmeans_data.csv"
OUTPUT_DIR = BASE_DIR / "output" / "phan3"
FEATURES = ["x", "y"]
SEED = 123


# 3.1. K-Means: gán điểm vào tâm gần nhất rồi cập nhật tâm bằng trung bình.
# Hàm mục tiêu là SSE = tổng bình phương khoảng cách từ điểm đến tâm cụm.
# K=3 nghĩa là yêu cầu 3 cụm, không có nghĩa dữ liệu có sẵn 3 lớp thật.


# 3.2. Tạo bộ dữ liệu, đúng thứ tự gọi np.random.rand(100) trong đề.
def create_dataset(seed=SEED):
    """Trả về 100 điểm trong [0, 1); không thay đổi bộ sinh số ngẫu nhiên toàn cục."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"x": rng.rand(100), "y": rng.rand(100)})


def load_dataset(path=DATASET_PATH):
    """Đọc đúng hai cột đặc trưng; không dùng nhãn cụm hoặc sai số làm đầu vào."""
    data = pd.read_csv(path)
    if list(data.columns) != FEATURES:
        raise ValueError("Dataset phải có đúng hai cột x,y theo thứ tự này.")
    data = data.astype(float)
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Dataset rỗng hoặc chứa giá trị thiếu/vô hạn.")
    return data


# 3.3. Khởi tạo K tâm ngẫu nhiên trong miền giá trị của dữ liệu.
def initialize_centroids(k, data, seed=SEED):
    """Mỗi dòng là một tâm; mỗi cột ứng với một đặc trưng x hoặc y."""
    rng = np.random.RandomState(seed)
    # Dùng min/max riêng cho mỗi trục để mỗi tâm nằm trong miền dữ liệu.
    values = rng.uniform(data.min().to_numpy(), data.max().to_numpy(),
                         size=(k, len(FEATURES)))
    return pd.DataFrame(values, columns=FEATURES)


# 3.4. Khoảng cách và cách gán một điểm vào tâm gần nhất.
def calculate_error(point, centroid):
    """Bình phương khoảng cách Euclid: (x2-x1)^2 + (y2-y1)^2.

    Đề gọi đại lượng này là error. Không cần căn bậc hai khi tìm tâm gần
    nhất, vì căn bậc hai không làm thay đổi thứ tự các khoảng cách.
    """
    difference = np.asarray(point, dtype=float) - np.asarray(centroid, dtype=float)
    return float(np.sum(difference ** 2))


def assign_centroid(data, centroids):
    """Trả về nhãn tâm gần nhất và sai số bình phương cho từng điểm."""
    labels, errors = [], []
    for point in data[FEATURES].to_numpy():
        distances = [calculate_error(point, center)
                     for center in centroids[FEATURES].to_numpy()]
        # argmin trả về VỊ TRÍ nhỏ nhất; amin/min trả về GIÁ TRỊ nhỏ nhất.
        # Khi hòa, argmin chọn tâm có chỉ số nhỏ nhất để kết quả ổn định.
        closest = int(np.argmin(distances))
        labels.append(closest)
        errors.append(distances[closest])
    return np.array(labels, dtype=int), np.array(errors, dtype=float)


# 3.5. Lặp các bước gán cụm -> cập nhật tâm -> kiểm tra hội tụ.
def k_means(data, k=3, seed=SEED, max_iter=300, tolerance=1e-8):
    """Gom cụm và trả về (nhãn, sai số, tâm, lịch sử SSE, đã hội tụ).

    Chỉ sử dụng x,y, giữ nguyên DataFrame đầu vào. Tâm rỗng được khởi tạo
    lại bằng điểm đang có sai số lớn; max_iter tránh vòng lặp vô hạn.
    """
    features = data[FEATURES].astype(float).copy()
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Dữ liệu phải là các số hữu hạn.")
    if not isinstance(k, int) or not 1 <= k <= len(features.drop_duplicates()):
        raise ValueError("K phải là số nguyên từ 1 đến số điểm phân biệt.")
    if max_iter < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_iter phải dương; tolerance phải hữu hạn, không âm.")

    centroids = initialize_centroids(k, features, seed)
    labels, errors = assign_centroid(features, centroids)
    history = [{"iteration": 0, "sse": float(errors.sum())}]
    converged = False

    for iteration in range(1, max_iter + 1):
        new_centroids = centroids.copy()
        empty_cluster = False
        replacement_indices = set()
        for cluster in range(k):
            members = features.loc[labels == cluster, FEATURES]
            if not members.empty:
                new_centroids.loc[cluster, FEATURES] = members.mean().to_numpy()
            else:
                # Không để mean của cụm rỗng sinh NaN hoặc làm mất một tâm.
                empty_cluster = True
                farthest = next(int(index) for index in np.argsort(-errors)
                                if int(index) not in replacement_indices)
                replacement_indices.add(farthest)
                new_centroids.loc[cluster, FEATURES] = features.iloc[farthest].to_numpy()

        new_labels, new_errors = assign_centroid(features, new_centroids)
        new_sse = float(new_errors.sum())
        history.append({"iteration": iteration, "sse": new_sse})
        shift = np.linalg.norm(new_centroids.to_numpy() - centroids.to_numpy(), axis=1).max()
        stable_labels = np.array_equal(labels, new_labels)
        stable_error = abs(history[-2]["sse"] - new_sse) <= tolerance
        centroids, labels, errors = new_centroids, new_labels, new_errors

        # Nhãn không đổi sau cập nhật: mỗi tâm là trung bình đúng của cụm.
        # Có thêm kiểm tra sai số và độ dịch tâm để hạn chế sai số số thực.
        if not empty_cluster and (stable_labels or (stable_error and shift <= tolerance)):
            converged = True
            break

    return labels, errors, centroids, pd.DataFrame(history), converged


# 3.6. Vẽ dữ liệu trước/sau gom cụm và biểu đồ SSE theo vòng lặp.
def plot_results(data, labels, centroids, history, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    axes[0].scatter(data["x"], data["y"], color="#526775", alpha=0.8)
    axes[0].set_title("3.2. Dữ liệu trước khi gom cụm")
    colors = ["#d83c3c", "#2675c7", "#23874d"]
    for cluster, center in centroids.iterrows():
        members = data.loc[labels == cluster]
        axes[1].scatter(members["x"], members["y"], color=colors[cluster],
                        alpha=0.75, label=f"Cụm {cluster}: {len(members)} điểm")
        axes[1].scatter(center["x"], center["y"], s=240, marker="X",
                        color=colors[cluster], edgecolors="black", linewidths=1.3)
    axes[1].set_title("3.6. Ba cụm và tâm cụm (dấu X)")
    axes[1].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set(xlabel="x", ylabel="y", xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.set_aspect("equal", adjustable="box")
    axes[2].plot(history["iteration"], history["sse"], marker="o", color="#6b43a8")
    axes[2].set(title="3.5. Tổng sai số SSE", xlabel="Vòng lặp (0: khởi tạo)", ylabel="SSE")
    for ax in axes:
        ax.grid(alpha=0.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "kmeans_3_cum.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Hiện cửa sổ biểu đồ.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # CSV được đóng gói sẵn. Nếu bị thiếu, tạo lại đúng 100 điểm với seed 123.
    if not DATASET_PATH.exists():
        create_dataset().to_csv(DATASET_PATH, index=False, float_format="%.17g")
    data = load_dataset()
    initial_centroids = initialize_centroids(3, data)
    initial_labels, initial_errors = assign_centroid(data, initial_centroids)
    labels, errors, centroids, history, converged = k_means(data, k=3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = data.assign(centroid=labels, error=errors)
    result.to_csv(OUTPUT_DIR / "ket_qua_gom_cum.csv", index=False, encoding="utf-8-sig")
    centroids.rename_axis("centroid").to_csv(OUTPUT_DIR / "tam_cum.csv", encoding="utf-8-sig")
    history.to_csv(OUTPUT_DIR / "lich_su_sse.csv", index=False, encoding="utf-8-sig")
    plot_results(data, labels, centroids, history, show=args.show)

    lines = [
        "PHẦN 3 - K-MEANS | Lê Quyết Tiến | 2474802010386",
        "3.1. Tự cài đặt K-Means với K = 3.",
        f"3.2. Dataset: {DATASET_PATH.name}; {len(data)} điểm x,y; seed = {SEED}.",
        data.head(10).to_string(index=False),
        "\n3.3. Tâm khởi tạo:", initial_centroids.to_string(),
        "\n3.4. Gán cụm ban đầu (5 điểm đầu):",
        data.assign(centroid=initial_labels, error=initial_errors).head().to_string(index=False),
        "\n3.5. Lịch sử SSE:", history.to_string(index=False),
        f"Trạng thái: {'Đã hội tụ' if converged else 'Đạt max_iter, chưa hội tụ'}.",
        f"Số vòng cập nhật: {len(history) - 1}; SSE cuối: {errors.sum():.8f}.",
        "Tâm cuối cùng:", centroids.to_string(),
        "Số điểm mỗi cụm:", result.groupby("centroid").size().to_string(),
        "\n3.6. Biểu đồ: output/phan3/kmeans_3_cum.png",
        "NOTE: Nhãn 0,1,2 chỉ là tên cụm, không phải thứ hạng hay nhãn có sẵn.",
        "NOTE: SSE càng nhỏ thì cụm càng chặt với cùng dữ liệu và cùng K.",
        "NOTE: K-Means phụ thuộc khởi tạo; một lần chạy không bảo đảm tối ưu toàn cục.",
    ]
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "ket_qua.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
