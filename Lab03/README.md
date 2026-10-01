# Lab 3 - Lê Quyết Tiến - 2474802010386

Lời giải **mục 3: Học không giám sát**, **mục 4: Tìm hiểu Flask** và
**mục 5: Bài tập** trong
`Lab_2_Machine_Learning_Basics.pdf`, trang 9-18. Mục 3 và 4 mỗi mục có một file
Python độc lập; mục 5 có năm file tương ứng năm bài tập. Source có docstring,
ghi chú và comments tiếng Việt theo từng mục của đề.

**Phần 5:** xem [README_PHAN5.md](README_PHAN5.md) để chạy 5 bài tập mới
(`bai1_titanic.py` đến `bai5_web.py`). Ba dataset hiện có đã được chép vào
thư mục này; bài 3 còn thiếu `Employee_Data.csv` gốc.

## Các file trong thư mục

| File | Nội dung |
| --- | --- |
| `phan3_kmeans.py` | Mục 3.1-3.6: tự cài đặt K-Means, gom 3 cụm, vẽ biểu đồ |
| `kmeans_data.csv` | Dataset 100 điểm x,y tạo đúng mục 3.2 với seed 123 |
| `phan4_flask.py` | Mục 4.1-4.5: Hello World, route, biến URL, chuyển hướng |
| `requirements.txt` | Thư viện cần để chạy hai bài |
| `output/phan3/` | Kết quả gom cụm, tâm cụm, lịch sử SSE, báo cáo và biểu đồ |

Dataset phần 3 là dữ liệu mô phỏng tạo bởi hai lần gọi `np.random.rand(100)`
cho x rồi y, với seed 123 như đề. CSV đã được đóng gói sẵn; nếu thiếu file,
chương trình tự tạo lại. Đây không phải dataset nhân viên của phần bài tập
cuối tài liệu. Mục 4 là các ví dụ Flask cơ bản nên **không cần dataset**.

## Cài thư viện và chạy

Mở terminal tại thư mục này. Máy hiện tại đã có Python 3.13 và các thư viện:

```powershell
py -3.13 phan3_kmeans.py
py -3.13 phan3_kmeans.py --show
py -3.13 phan4_flask.py
```

Lệnh đầu tính và lưu kết quả; lệnh thứ hai chạy lại đồng thời mở cửa sổ biểu đồ.
Lệnh thứ ba khởi động Flask và giữ terminal hoạt động để phục vụ trình duyệt.
Dừng Flask bằng **Ctrl+C**. Hai file có thể chạy riêng, không phụ thuộc lẫn nhau.

Nếu chưa có thư viện:

```powershell
py -3.13 -m pip install -r requirements.txt
```

Nếu máy chỉ có lệnh `python`, thay `py -3.13` bằng `python` trong các lệnh trên.
Trong VS Code trên máy hiện tại, chọn interpreter `E:\Python\python.exe`.
Đường dẫn dataset và kết quả được tính từ vị trí source, không phụ thuộc thư mục
làm việc khi nhấn Run.

## Phần 3: trình tự thuật toán và ghi chú

1. **3.1:** Đặt K = 3. Mỗi điểm chỉ có hai đặc trưng x,y, không có nhãn lớp.
2. **3.2:** Đọc 100 điểm trong `kmeans_data.csv`; xem 10 dòng đầu trong kết quả.
3. **3.3:** Khởi tạo 3 tâm ngẫu nhiên với seed 123 trong miền min/max của từng trục.
4. **3.4:** Tính `error = (x-cx)^2 + (y-cy)^2`, gán điểm vào tâm có error nhỏ nhất.
5. **3.5:** Thay mỗi tâm bằng trung bình các điểm trong cụm, gán lại và lặp tới hội tụ.
6. **3.6:** Vẽ dữ liệu ban đầu, ba cụm cùng các tâm và đường SSE theo vòng lặp.

`calculate_error()` dùng khoảng cách Euclid **bình phương** như công thức error
trong đề. Không lấy căn vẫn chọn cùng tâm gần nhất. SSE là tổng các error,
không phải RMSE và không phải phần trăm chính xác.

`argmin` lấy vị trí nhỏ nhất để làm nhãn; `min`/`amin` lấy giá trị nhỏ nhất.
Các cột `centroid` và `error` là kết quả, tuyệt đối không đưa vào đặc trưng cho
vòng lặp tiếp theo. Source xử lý cụm rỗng bằng điểm có sai số lớn, giữ nguyên
K tâm, giới hạn 300 vòng lặp và báo rõ nếu chưa hội tụ. Kiểm tra nhãn ổn định
hoặc sai số cùng độ dịch tâm nhỏ hơn `1e-8` thay cho chỉ làm tròn SSE đến 3 số.

So với ví dụ PDF, min/max khởi tạo được tính riêng từng trục; bộ sinh số ngẫu
nhiên của dữ liệu và của tâm được tách riêng để mỗi lần chạy đều tái lập.
Do đó nhãn/tâm có thể khác một lần chạy trực tiếp mã mẫu. Nhãn 0,1,2 chỉ là
tên cụm. Dữ liệu ngẫu nhiên đều không có ba lớp thật để đo accuracy; K-Means
vẫn chia thành ba vùng theo yêu cầu, không bảo đảm nghiệm tối ưu toàn cục.

Kết quả lưu ở `output/phan3/`:

- `ket_qua_gom_cum.csv`: 100 điểm x,y cùng nhãn `centroid` và sai số `error`.
- `tam_cum.csv`: tọa độ tâm cuối cùng của 3 cụm.
- `lich_su_sse.csv`: SSE từ khởi tạo (vòng 0) qua từng vòng cập nhật.
- `ket_qua.txt`: báo cáo các mục 3.1-3.6.
- `kmeans_3_cum.png`: hình trước/sau gom cụm và đường hội tụ.

## Phần 4: các URL cần thử

Chạy `py -3.13 phan4_flask.py`, sau đó mở các URL sau trong trình duyệt:

| Mục | URL | Kết quả |
| --- | --- | --- |
| 4.2 | http://127.0.0.1:5000/ | `Hello World` |
| 4.3 | http://127.0.0.1:5000/hello_vlu | `hello world Vanlang University` |
| 4.4 | http://127.0.0.1:5000/hello/VANLANG | `hello VANLANG!` |
| 4.4 | http://127.0.0.1:5000/blog/11 | `Blog Number 11` |
| 4.4 | http://127.0.0.1:5000/rev/1.5 | `Revision Number 1.500000` |
| 4.5 | http://127.0.0.1:5000/admin | `Hello Admin` |
| 4.5 | http://127.0.0.1:5000/guest/Tien | `Hello Tien as Guest` |
| 4.5 | http://127.0.0.1:5000/user/admin | HTTP 302 tới `/admin`, rồi `Hello Admin` |
| 4.5 | http://127.0.0.1:5000/user/Tien | HTTP 302 tới `/guest/Tien`, rồi `Hello Tien as Guest` |

`<int:postID>` và `<float:revNo>` là bộ chuyển đổi kiểu URL của Flask. Ví dụ
`/blog/abc` và `/rev/1` trả về 404 vì không khớp kiểu. Với float, dùng `/rev/1.0`.
`url_for()` tạo URL từ tên hàm, còn `redirect()` trả phản hồi chuyển hướng.
Tên `admin` trong bài chỉ minh họa câu lệnh if/else, không thực hiện đăng nhập.

Nếu cổng 5000 đang được dùng, chạy `py -3.13 phan4_flask.py --port 5001`
và thay `5000` thành `5001` trong các URL trên.
