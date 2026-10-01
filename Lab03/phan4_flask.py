"""PHẦN 4 - TÌM HIỂU FLASK: các mục 4.1-4.5 trong đề, trang 13-17.
4.1. Cài đặt: py -3.13 -m pip install Flask
Chạy: py -3.13 phan4_flask.py
Mở trình duyệt: http://127.0.0.1:5000
Dừng server: Ctrl+C trong terminal đang chạy chương trình.

File này chạy độc lập, không cần dataset hay file HTML bên ngoài.
Gộp các ví dụ phát triển dần của đề vào một ứng dụng với đầy đủ route.
"""

import argparse
from flask import Flask, redirect, url_for
from markupsafe import escape


# __name__ giúp Flask xác định module chứa ứng dụng.
# Biến app chỉ được tạo khi import; server chỉ chạy trong khối __main__.
app = Flask(__name__)


# 4.2. Kiểm tra Flask: mở http://127.0.0.1:5000/
@app.route("/")
def hello_world():
    """Trang gốc, trả về chuỗi Hello World đúng ví dụ trong đề."""
    return "Hello World"


# 4.3. Dùng route() để gắn một đường dẫn với một hàm xử lý.
@app.route("/hello_vlu")
def hello_world_vlu():
    """Mở http://127.0.0.1:5000/hello_vlu."""
    return "hello world Vanlang University"


# 4.4. Truyền biến từ URL: <name> mặc định nhận chuỗi.
@app.route("/hello/<name>")
def hello_name(name):
    """Ví dụ /hello/VANLANG -> hello VANLANG!."""
    # escape giữ tên nhập vào là văn bản, không bị trình duyệt hiểu là HTML.
    return f"hello {escape(name)}!"


@app.route("/blog/<int:postID>")
def show_blog(postID):
    """int: nhận số nguyên, ví dụ /blog/11 -> Blog Number 11."""
    return f"Blog Number {postID:d}"


@app.route("/rev/<float:revNo>")
def revision(revNo):
    """float: dùng dấu chấm thập phân, ví dụ /rev/1.5 -> Revision Number 1.500000.

    /rev/1 không khớp converter float; /blog/abc không khớp converter int.
    Khi URL không khớp bất kỳ route nào, Flask tự trả mã HTTP 404.
    """
    return f"Revision Number {revNo:f}"


# 4.5. url_for() tạo URL từ TÊN HÀM (endpoint), redirect() chuyển trình duyệt.
@app.route("/admin")
def hello_admin():
    """Trang đích cho ví dụ tên admin; đây không phải cơ chế đăng nhập."""
    return "Hello Admin"


@app.route("/guest/<guest>")
def hello_guest(guest):
    """Trang đích cho các tên khác admin."""
    return f"Hello {escape(guest)} as Guest"


@app.route("/user/<name>")
def hello_user(name):
    """/user/admin -> /admin; /user/Tien -> /guest/Tien.

    redirect() trả HTTP 302, rồi trình duyệt gửi yêu cầu mới đến Location.
    url_for() nhận tên hàm hello_admin/hello_guest, không nhận '/admin'.
    So sánh có phân biệt hoa thường, đúng ví dụ: chỉ 'admin' đi tới /admin.
    """
    if name == "admin":
        return redirect(url_for("hello_admin"))
    return redirect(url_for("hello_guest", guest=name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000,
                        help="Cổng chạy Flask; đổi sang 5001 nếu cổng 5000 đang bận.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Cổng phải nằm trong khoảng 1-65535.")
    # Server phục vụ bài thực hành trên máy cá nhân. Không tự bật reloader
    # để tránh chạy chương trình hai lần; không dùng server này để triển khai thật.
    app.run(host="127.0.0.1", port=args.port, debug=False)
