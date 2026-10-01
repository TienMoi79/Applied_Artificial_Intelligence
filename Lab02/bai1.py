"""Lab 2 - Phần 2: Học giám sát (mục 2.1 đến 2.8).

Tự cài đặt hồi quy tuyến tính đơn, không dùng mô hình có sẵn.
Chạy trên máy hiện tại: py -3.13 1.py
Hiện biểu đồ: py -3.13 1.py --show
Dự báo giá trị khác: py -3.13 1.py --x-new 25
"""

import argparse
import csv
import math
from pathlib import Path
from random import Random
import sys


BASE_DIR = Path(__file__).resolve().parent
DEMO_DATASET = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]


# 2.3. Tính Mean và Variance.
def mean(values):
    """Trung bình cộng: tổng các giá trị / số lượng giá trị."""
    if not values:
        raise ValueError("Không thể tính mean của danh sách rỗng.")
    return sum(values) / len(values)


def variance(values, mean_value):
    """Theo đề: tổng bình phương độ lệch, KHÔNG chia n hoặc n - 1."""
    return sum((value - mean_value) ** 2 for value in values)


# 2.4. Tính Covariance.
def covariance(x, mean_x, y, mean_y):
    """Theo đề: tổng tích độ lệch của X và Y, chưa chuẩn hóa."""
    if len(x) != len(y):
        raise ValueError("X và Y phải có cùng số phần tử.")
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))


# 2.5. Ước lượng hệ số B0 và B1 cho y = B0 + B1*x.
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    mean_x, mean_y = mean(x), mean(y)
    var_x = variance(x, mean_x)
    if var_x == 0:
        raise ValueError("Không thể ước lượng B1 khi mọi giá trị X bằng nhau.")
    b1 = covariance(x, mean_x, y, mean_y) / var_x
    b0 = mean_y - b1 * mean_x
    return b0, b1


# 2.6. Dự báo; chỉ dùng X của tập test, không đọc nhãn Y của test.
def simple_linear_regression(train, test):
    b0, b1 = coefficients(train)
    return [b0 + b1 * row[0] for row in test]


# 2.7. Đánh giá bằng Root Mean Squared Error.
def rmse_metric(actual, predicted):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Hai danh sách phải không rỗng và có cùng số phần tử.")
    return math.sqrt(
        sum((y_hat - y) ** 2 for y, y_hat in zip(actual, predicted)) / len(actual)
    )


# 2.8.2. Đọc CSV, bỏ dòng tiêu đề và dòng trống.
def load_csv(filename):
    with Path(filename).open(newline="", encoding="utf-8-sig") as file:
        rows = csv.reader(file)
        next(rows, None)
        dataset = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not dataset:
        raise ValueError(f"File không có dữ liệu: {filename}")
    return dataset


def str_column_to_float(dataset, column):
    """Chuyển từng cột từ chuỗi trong CSV sang số thực."""
    for line_number, row in enumerate(dataset, start=2):
        try:
            value = float(row[column].strip())
        except (ValueError, IndexError) as error:
            raise ValueError(f"Dữ liệu không hợp lệ ở dòng {line_number}.") from error
        if not math.isfinite(value):
            raise ValueError(f"Giá trị không hữu hạn ở dòng {line_number}.")
        row[column] = value


# 2.8.3. Chia ngẫu nhiên thành train 60% và test 40%.
def train_test_split(dataset, split=0.6, seed=1):
    if not 0 < split < 1:
        raise ValueError("Tỷ lệ train phải nằm trong khoảng (0, 1).")
    # Làm tròn lên giống vòng while len(train) < split*len(dataset) trong đề.
    train_size = math.ceil(split * len(dataset))
    if train_size < 2 or train_size >= len(dataset):
        raise ValueError("Cần ít nhất 2 mẫu train và 1 mẫu test.")
    remaining = [list(row) for row in dataset]
    train = []
    rng = Random(seed)  # Seed 1 giống đề, giúp tái lập kết quả mỗi lần chạy.
    while len(train) < train_size:
        train.append(remaining.pop(rng.randrange(len(remaining))))
    return train, remaining


# 2.7 và 2.8.4. split=None: ví dụ trên tập huấn luyện; split=0.6: test riêng.
def evaluate_algorithm(dataset, algorithm, split=None, *args, seed=1):
    if split is None:
        train, test = dataset, dataset
    else:
        train, test = train_test_split(dataset, split, seed)
    test_without_labels = [[row[0], None] for row in test]
    predicted = algorithm(train, test_without_labels, *args)
    actual = [row[1] for row in test]
    return rmse_metric(actual, predicted)


def describe_dataset(dataset):
    """Kết quả các mục 2.3, 2.4, 2.5 cho một bộ dữ liệu."""
    x, y = [row[0] for row in dataset], [row[1] for row in dataset]
    mx, my = mean(x), mean(y)
    b0, b1 = coefficients(dataset)
    return [
        f"Số mẫu: {len(dataset)}",
        f"X: Mean = {mx:.6f}; Variance (theo đề) = {variance(x, mx):.6f}",
        f"Y: Mean = {my:.6f}; Variance (theo đề) = {variance(y, my):.6f}",
        f"Covariance (theo đề) = {covariance(x, mx, y, my):.6f}",
        f"B0 = {b0:.6f}; B1 = {b1:.6f}",
        f"Phương trình: Y = {b0:.6f} + {b1:.6f} * X",
    ]


# 2.8.7. Trực quan hóa dữ liệu và đường hồi quy.
def plot_results(dataset, train, test, output_file, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # Lưu ảnh, không chặn chương trình để đợi đóng cửa sổ.
    import matplotlib.pyplot as plt

    b0_all, b1_all = coefficients(dataset)
    b0_train, b1_train = coefficients(train)
    x = [row[0] for row in dataset]
    limits = [min(x), max(x)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].scatter(x, [row[1] for row in dataset], color="#2878b5", alpha=0.75,
                    label=f"Toàn bộ dữ liệu ({len(dataset)} mẫu)")
    axes[0].plot(limits, [b0_all + b1_all * value for value in limits],
                 color="#d45d00", label="Hồi quy trên toàn bộ dữ liệu")
    axes[0].set_title("Swedish Insurance: dữ liệu và hồi quy")

    for rows, label, color, marker in (
        (train, f"Train ({len(train)} mẫu)", "#2878b5", "o"),
        (test, f"Test ({len(test)} mẫu)", "#d45d00", "x"),
    ):
        axes[1].scatter([row[0] for row in rows], [row[1] for row in rows],
                        label=label, color=color, marker=marker, alpha=0.8)
    axes[1].plot(limits, [b0_train + b1_train * value for value in limits],
                 color="#218554", label="Hồi quy chỉ học từ train")
    axes[1].set_title("Chia train/test theo tỷ lệ 60%/40%")
    for ax in axes:
        ax.set_xlabel("X - Số lần yêu cầu bảo hiểm")
        ax.set_ylabel("Y - Tổng tiền bảo hiểm chi trả")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output_file, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-new", type=float, default=18,
                        help="Số lần yêu cầu cần dự báo (mặc định: 18 như đề).")
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ biểu đồ.")
    args = parser.parse_args()
    if not math.isfinite(args.x_new) or args.x_new < 0:
        parser.error("--x-new phải là số hữu hạn không âm.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = BASE_DIR / "output" / "phan2"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "LAB 2 - PHẦN 2: HỌC GIÁM SÁT",
        "2.1. Linear Regression: Y = B0 + B1 * X.",
        "2.2. Swedish Insurance: X = số yêu cầu, Y = tổng tiền chi trả.",
        "",
        "2.3 - 2.5. VÍ DỤ 5 ĐIỂM TRONG ĐỀ",
        f"Dataset: {DEMO_DATASET}",
        *describe_dataset(DEMO_DATASET),
    ]
    demo_predictions = simple_linear_regression(DEMO_DATASET, DEMO_DATASET)
    demo_rmse = evaluate_algorithm(DEMO_DATASET, simple_linear_regression)
    lines.extend([
        "",
        "2.6. DỰ BÁO TRÊN VÍ DỤ",
        f"Y thực tế: {[row[1] for row in DEMO_DATASET]}",
        f"Y dự báo: {[round(value, 6) for value in demo_predictions]}",
        "",
        "2.7. ĐÁNH GIÁ VÍ DỤ",
        f"RMSE = {demo_rmse:.6f}",
        "Đây là lỗi trên tập huấn luyện, chưa phản ánh lỗi trên dữ liệu mới.",
    ])

    # 2.8.1 - 2.8.2. Dùng dữ liệu có sẵn cạnh file 1.py.
    dataset = load_csv(BASE_DIR / "SwedishInsurance.csv")
    if any(len(row) != 2 for row in dataset):
        raise ValueError("SwedishInsurance.csv phải có đúng hai cột X, Y.")
    for column in range(2):
        str_column_to_float(dataset, column)
    train, test = train_test_split(dataset, split=0.6, seed=1)
    b0_train, b1_train = coefficients(train)
    test_predictions = simple_linear_regression(train, [[row[0], None] for row in test])
    test_rmse = evaluate_algorithm(dataset, simple_linear_regression, 0.6, seed=1)

    lines.extend([
        "",
        "2.8.1 - 2.8.2. ĐỌC SWEDISH INSURANCE VÀ TÍNH THỐNG KÊ",
        *describe_dataset(dataset),
        "",
        "2.8.3 - 2.8.5. HUẤN LUYỆN VÀ ĐÁNH GIÁ TRÊN TEST",
        f"Seed = 1; train = {len(train)} mẫu; test = {len(test)} mẫu.",
        "Số mẫu train được làm tròn lên theo cách chia trong đề.",
        f"Hệ số học CHỈ từ train: B0 = {b0_train:.6f}; B1 = {b1_train:.6f}",
        f"RMSE test = {test_rmse:.6f}",
    ])
    predictions_path = output_dir / "du_bao_test.csv"
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y_thuc_te", "Y_du_bao", "Sai_so_du_bao_tru_thuc_te"])
        for row, prediction in zip(test, test_predictions):
            writer.writerow([row[0], row[1], prediction, prediction - row[1]])

    # 2.8.6. Đề huấn luyện lại trên toàn bộ dataset để dự báo x_new = 18.
    b0, b1 = coefficients(dataset)
    y_new = b0 + b1 * args.x_new
    lines.extend([
        "",
        "2.8.6. DỰ BÁO GIÁ TRỊ MỚI",
        "Mô hình dự báo này dùng toàn bộ dữ liệu, theo đoạn mã ở mục 2.8.6.",
        f"X mới = {args.x_new:g} -> Y dự báo = {y_new:.6f}",
        "RMSE test phía trên thuộc mô hình học từ train, không phải mô hình này.",
        "",
        "2.8.7. TRỰC QUAN HÓA",
        "Biểu đồ: output/phan2/swedish_insurance.png",
        "Dự báo từng mẫu test: output/phan2/du_bao_test.csv",
        "",
        "LƯU Ý: Variance/Covariance trong đề là tổng độ lệch chưa chuẩn hóa.",
        "RMSE cùng đơn vị với Y; giá trị nhỏ hơn biểu thị sai số thấp hơn.",
    ])
    plot_results(dataset, train, test, output_dir / "swedish_insurance.png", args.show)
    report = "\n".join(lines) + "\n"
    (output_dir / "ket_qua.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()