"""PHẦN 5 - BÀI 4: gom 3 cụm theo Salaries và Overtime.

Lê Quyết Tiến - 2474802010386.
Chạy: py -3.13 bai4_compensation.py
Dataset: KM_Employee_Compensation_SF.csv, đặt cạnh source.
Dùng lại hàm K-Means/chuẩn hóa/xuất kết quả của bai3_employee.py.

NOTE: Các dòng là bản ghi thu nhập, không nhất thiết là người khác nhau.
Dataset chứa nhiều năm và hai loại năm (Fiscal/Calendar). Bài này gom các
bản ghi theo đúng yêu cầu hai cột, không cộng tổng giữa các loại năm.
Giữ số âm trong dữ liệu kế toán gốc; không tự coi đó là dữ liệu bị lỗi.
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

from bai3_employee import fit_clusters, predict_cluster, save_results

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output" / "phan5" / "bai4"


def train_model(path=None):
    """Đọc đúng hai cột để giảm bộ nhớ, fit trên tất cả bản ghi hợp lệ."""
    path = Path(path) if path else BASE_DIR / "KM_Employee_Compensation_SF.csv"
    raw = pd.read_csv(path, usecols=["Salaries", "Overtime"])
    return fit_clusters(raw, ["Salaries", "Overtime"], path.name)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salary", type=float, help="Lương của một mẫu mới (tùy chọn).")
    parser.add_argument("--overtime", type=float, help="Tiền làm thêm của mẫu mới.")
    args = parser.parse_args()
    if (args.salary is None) != (args.overtime is None):
        parser.error("Cần truyền cả --salary và --overtime.")
    try:
        result = train_model()
        print(save_results(result, OUTPUT))
        print("NOTE: Giữ các khoản âm trong dữ liệu gốc; mỗi dòng là một bản ghi thu nhập.")
        if args.salary is not None:
            label = predict_cluster(result, [args.salary, args.overtime])
            print(f"Mẫu mới Salaries={args.salary:g}, Overtime={args.overtime:g}: cụm {label}.")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Lỗi: {error}\n")
