# Phần 5 - Năm bài tập

**Sinh viên:** Lê Quyết Tiến - **MSSV:** 2474802010386.

Mỗi bài có một file Python. Bài 5 import các module bài 1-4; bài 4 dùng lại
hàm xử lý/gom cụm trong bài 3. Vì vậy hãy giữ các file trong cùng thư mục.
Các chương trình tìm dataset theo vị trí source, không theo thư mục terminal.
Source có docstring, giải thích xử lý dữ liệu, mô hình và các giới hạn kết quả.

## File và trạng thái

| Bài | Source | Dataset | Trạng thái |
| --- | --- | --- | --- |
| 1 | `bai1_titanic.py` | `titanic.csv` | Đã chạy và xuất kết quả |
| 2 | `bai2_fish.py` | `Fish.csv` | Đã chạy và xuất kết quả |
| 3 | `bai3_employee.py` | `Employee_Data.csv` | Có source; thiếu dataset gốc để chạy và kiểm chứng kết quả |
| 4 | `bai4_compensation.py` | `KM_Employee_Compensation_SF.csv` | Đã chạy toàn bộ 291.825 dòng |
| 5 | `bai5_web.py` | Dùng dataset của bài 1-4 | Web hoạt động; bài 3 hiển thị thông báo thiếu file |

Ba CSV hiện có đã được chép nguyên bản vào thư mục bài nộp. **Employee_Data.csv
chưa được cung cấp**; không thay bằng dữ liệu lương của bài 4 hay dữ liệu giả.
Khi có file gốc, đặt cạnh source rồi chạy bài 3 hoặc tải lại trang bài 3 trên Web.

## Chạy bài

Mở terminal trong `Lab3_LeQuyetTien_2474802010386`:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 bai1_titanic.py --age 30
py -3.13 bai2_fish.py --weight 300
py -3.13 bai3_employee.py
py -3.13 bai4_compensation.py
py -3.13 bai5_web.py
```

Bài 3 hiện báo thiếu file và thoát với mã 1; đó là tình trạng thiếu đầu vào,
không phải kết quả gom cụm. Nếu dataset nằm nơi khác hoặc muốn chọn cột cụ thể:

```powershell
py -3.13 bai3_employee.py --data "D:/duong_dan/Employee_Data.csv"
py -3.13 bai3_employee.py --columns Age Salary
```

`Age Salary` chỉ là ví dụ cú pháp; phải thay bằng tên cột thực tế trong file.
Bài 3 tự chọn các cột số có biến thiên và bỏ mã định danh phổ biến nếu không
truyền `--columns`. Luôn kiểm tra danh sách cột in trong báo cáo.

Dự đoán cụm một bản ghi lương mới:

```powershell
py -3.13 bai4_compensation.py --salary 60000 --overtime 3000
```

Nếu dùng một Python khác, thay `py -3.13` bằng lệnh của interpreter đó và cài
requirements trong cùng môi trường. Máy hiện tại đã có đủ thư viện ở Python 3.13.

## Bài 1 - Titanic

- Đầu vào: `age`; nhãn cần dự báo: `fare`.
- Chuyển hai cột sang số; bỏ thiếu, vô hạn và giá trị âm. Không điền nhãn fare.
- Có 1.045 dòng hợp lệ từ 1.310 dòng ban đầu; 627 train và 418 test.
- Linear Regression; test 40%, `random_state=1`.
- RMSE test **63.321757**, R² test **0.024236**.
- Sau đánh giá, học lại toàn bộ dữ liệu hợp lệ: `fare = 16.019756 + 0.692297*age`.
- Tuổi 30 cho tiền vé dự báo **36.788655** theo đơn vị gốc của dataset.

R² thấp cho thấy tuổi riêng lẻ giải thích giá vé kém; không diễn giải RMSE
thành phần trăm chính xác. Có ghi chú ngoại suy khi tuổi nằm ngoài dữ liệu đã học.

## Bài 2 - Fish

- Linear Regression: `Weight` → `Width`, một mô hình chung cho các loài.
- KNN với 5 láng giềng: `Weight` → `Species` (phân loại, không phải hồi quy).
- Bỏ 1 mẫu Weight=0; còn 158 mẫu, 94 train và 64 test.
- Chia 60/40 có stratify theo loài, `random_state=1`. Hai mô hình chỉ dùng Weight.
- RMSE Width test **0.822851 cm**, R² **0.740086**.
- Accuracy Species test **48.44%**; không khẳng định trọng lượng xác định được loài.
- Mô hình dự báo mới học lại từ 158 mẫu. Với 300 g: Width **4.002015 cm**,
  loài dự đoán **Perch**, được 2/5 láng giềng bỏ phiếu (40%).

Tỷ lệ phiếu KNN là sự đồng thuận cục bộ, không phải xác suất đã hiệu chỉnh hay
accuracy của toàn mô hình. Các loài có khoảng trọng lượng chồng lấn đáng kể.
`confusion_matrix.csv` có hàng là loài thật, cột là loài dự đoán.

## Bài 3 và 4 - Gom cụm

1. Chọn cột số; loại hàng thiếu toàn bộ đặc trưng.
2. Điền các ô thiếu còn lại bằng median của từng cột.
3. Dùng StandardScaler để đưa mỗi cột về thang đo so sánh được.
4. KMeans K=3, `n_init=10`, seed 123, tối đa 300 vòng, `tol=1e-6`.
5. Đổi tâm về đơn vị gốc, xuất số mẫu mỗi cụm và SSE trong không gian chuẩn hóa.

Phần 3 trước đó đã tự cài K-Means để học thuật toán. Các bài tập này dùng
scikit-learn để xử lý hiệu quả dữ liệu lớn. Không cần train/test khi mục tiêu
là gom nhóm các bản ghi chưa có nhãn. Không có accuracy nếu thiếu nhãn cụm thật.
Các nhãn 1, 2, 3 chỉ đặt tên cụm, không xếp hạng người lao động.

Bài 4 chỉ dùng `Salaries`, `Overtime`, giữ các giá trị âm gốc vì đây có thể là
điều chỉnh kế toán. Mỗi dòng là một **bản ghi thu nhập**, có thể trùng người ở
các năm/loại năm khác nhau. Không cộng dồn giữa Fiscal và Calendar.

Kết quả bài 4 (291.825 bản ghi, không loại dòng, không có ô thiếu):

| Cụm | Salaries tại tâm | Overtime tại tâm | Số bản ghi |
| --- | ---: | ---: | ---: |
| 1 | 98.255,592835 | 3.330,563692 | 129.078 |
| 2 | 26.913,889547 | 822,768365 | 144.052 |
| 3 | 100.803,323679 | 39.439,486104 | 18.695 |

SSE chuẩn hóa: **192511.651647**; lần khởi tạo được chọn chạy 15 vòng.
Biểu đồ bài 4 chỉ hiển thị mẫu ngẫu nhiên tối đa 6.000 điểm để dễ đọc;
huấn luyện và file kết quả vẫn chứa đủ 291.825 bản ghi.

## Bài 5 - Ứng dụng Flask

Chạy `py -3.13 bai5_web.py`, mở **http://127.0.0.1:5001**:

| Đường dẫn | Chức năng |
| --- | --- |
| `/` | Tổng quan bốn bài và tình trạng dataset |
| `/bai/1` | Nhập tuổi, dự báo tiền vé; xem RMSE/R² và biểu đồ |
| `/bai/2` | Nhập trọng lượng, dự báo chiều rộng/loài; xem đánh giá và biểu đồ |
| `/bai/3` | Chọn cột số và gom cụm nhân viên khi có Employee_Data.csv |
| `/bai/4` | Xem ba cụm lương; nhập Salaries/Overtime để gán mẫu mới |

Mô hình được tính ở lần mở bài đầu tiên và lưu trong bộ nhớ, không học lại
cho từng dự báo. Trang 4 có thể mất vài giây khi mở lần đầu. CSV thay đổi
thời gian cập nhật hoặc kích thước sẽ làm mô hình được tính lại.
Web kiểm tra giá trị rỗng/không hợp lệ/NaN/vô hạn và báo lỗi trên trang.
Chưa có Employee_Data.csv thì bài 3 báo thiếu file, các bài khác vẫn dùng được.

Dừng server bằng Ctrl+C. Nếu cổng 5001 đang bận:

```powershell
py -3.13 bai5_web.py --port 5002
```

Server chỉ mở ở `127.0.0.1` trên máy cá nhân; giao diện và CSS ở ngay trong
`bai5_web.py`, không cần mạng ngoài, template HTML riêng hay dịch vụ lưu trữ.

## Kết quả và kiểm tra đã thực hiện

Kết quả CLI nằm trong `output/phan5/bai1`, `bai2`, `bai4`: báo cáo `.txt`,
biểu đồ `.png`, bảng dự báo hoặc bảng gán cụm `.csv`. Bài 3 sẽ tạo thư mục
kết quả tương ứng sau khi nhận đúng dataset. `source_index` là chỉ số dòng
DataFrame gốc (từ 0); `source_csv_row` là vị trí bản ghi cộng dòng tiêu đề (từ 2),
giả sử mỗi bản ghi CSV nằm trên một dòng.

Đã đối chiếu hệ số hồi quy với công thức covariance/variance và tính lại RMSE
từ dự báo test. Đã kiểm tra biểu mẫu Web, trả ảnh PNG, dữ liệu nhập không hợp lệ,
tình trạng thiếu file và số bản ghi/nhãn của bài 4. Logic bài 3 được kiểm tra
trên fixture nhỏ trong bộ nhớ; **chưa kiểm chứng trên Employee_Data.csv thật**.
