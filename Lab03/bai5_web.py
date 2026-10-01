"""PHẦN 5 - BÀI 5: triển khai bài 1-4 bằng Flask, tất cả UI trong một file.

Lê Quyết Tiến - 2474802010386.
Chạy: py -3.13 bai5_web.py
Truy cập http://127.0.0.1:5001, dừng bằng Ctrl+C.

Dùng lại bốn module bài tập, không sao chép công thức/thuật toán vào Web.
Mô hình được tính khi mở bài lần đầu và lưu trong bộ nhớ. Khi CSV thay đổi,
khóa cache đổi theo thời gian cập nhật và kích thước file, mô hình được học lại.
File Employee_Data.csv thiếu không làm các bài khác ngừng hoạt động.
"""
import argparse
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, Response, render_template_string, request, url_for

import bai1_titanic as titanic
import bai2_fish as fish
import bai3_employee as employee
import bai4_compensation as compensation

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
PLOT_LOCK = Lock()  # Matplotlib không an toàn khi nhiều request cùng vẽ.
EXERCISES = {
    1: {"title": "Tiền vé Titanic", "subtitle": "Dự báo tiền vé từ độ tuổi", "dataset": "titanic.csv", "module": titanic, "tag": "Hồi quy"},
    2: {"title": "Kích thước & loài cá", "subtitle": "Dự báo chiều rộng và loài từ trọng lượng", "dataset": "Fish.csv", "module": fish, "tag": "Hồi quy + phân loại"},
    3: {"title": "Nhóm nhân viên", "subtitle": "Gom Employee Data thành ba cụm", "dataset": "Employee_Data.csv", "module": employee, "tag": "K-Means"},
    4: {"title": "Lương & làm thêm", "subtitle": "Gom ba cụm theo Salaries và Overtime", "dataset": "KM_Employee_Compensation_SF.csv", "module": compensation, "tag": "K-Means"},
}

# Template là hằng chuỗi do lập trình viên viết. Dữ liệu biểu mẫu đi qua
# Jinja autoescape; không ghép input của người dùng vào mã template.
TEMPLATE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · Lab 3</title><style>
:root{--ink:#18323d;--muted:#526772;--accent:#087b76;--line:#dbe5e8;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#f3f7f8;color:var(--ink);font:16px/1.65 system-ui,sans-serif}
header{background:#163d49;color:white}header .wrap{padding:23px 26px;display:flex;justify-content:space-between;gap:18px;align-items:center}
.wrap{max-width:1180px;margin:auto;padding:30px 26px}a{color:var(--accent)}header a{color:white;text-decoration:none}
.brand{font-size:21px;font-weight:750}.identity{font-size:13px;color:#cde0e5}nav{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:28px}
nav a{padding:8px 14px;background:white;border:1px solid var(--line);border-radius:8px;text-decoration:none;font-size:14px}
nav a.active{background:var(--accent);border-color:var(--accent);color:white}.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--accent);font-weight:750}
h1{font-size:clamp(27px,4vw,39px);line-height:1.2;margin:9px 0 12px}h2{font-size:20px;margin:0 0 13px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-top:26px}.card,.panel{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:25px}
.card{display:flex;flex-direction:column;gap:8px}.card h2{margin:3px 0}.card p{margin:0}.card .link{margin-top:16px;font-weight:650;text-decoration:none}
.tag{align-self:start;font-size:12px;background:#e8f5f1;padding:4px 10px;border-radius:20px;color:#12645b}.missing{color:#965014;background:#fff0d9}
.grid{display:grid;grid-template-columns:310px 1fr;gap:22px;margin-top:25px;align-items:start}.panel{min-width:0}label{display:block;font-size:14px;font-weight:650;margin:12px 0 6px}
input{width:100%;border:1px solid #adc1c8;padding:11px;border-radius:7px;font:inherit}button{border:0;background:var(--accent);color:white;border-radius:8px;padding:12px 18px;font:inherit;font-weight:650;margin-top:18px;cursor:pointer}
.result{margin-top:22px;padding:17px;background:#e8f5f1;border-radius:10px}.result strong{display:block;font-size:25px}.small{font-size:13px}.warning{background:#fff1db;border:1px solid #eacb94;padding:17px;border-radius:10px;margin:18px 0;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}.metric{background:#f1f6f7;padding:13px;border-radius:9px}.metric b{display:block;font-size:21px}.metric span{font-size:12px;color:var(--muted)}
img.chart{width:100%;height:auto;display:block;border-radius:8px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#f1f6f7}
footer{margin-top:25px;color:var(--muted);font-size:12px}.mono{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
@media(max-width:800px){.grid,.cards{grid-template-columns:1fr}header .wrap{align-items:start;flex-direction:column}.metrics{grid-template-columns:repeat(2,1fr)}.wrap{padding:22px 16px}.panel,.card{padding:19px}}
</style></head><body><header><div class="wrap"><a class="brand" href="{{ url_for('index') }}">ML LAB / PHẦN 5</a>
<div class="identity">Lê Quyết Tiến · 2474802010386</div></div></header><main class="wrap">
<nav><a href="{{ url_for('index') }}" {% if not number %}class="active"{% endif %}>Tổng quan</a>
{% for i,e in exercises.items() %}<a href="{{ url_for('exercise', number=i) }}" {% if number==i %}class="active"{% endif %}>Bài {{ i }} · {{ e.title }}</a>{% endfor %}</nav>
{% if not number %}
<div class="eyebrow">BÀI 5 · ỨNG DỤNG FLASK</div><h1>Thực hành học máy</h1><p class="muted">Chọn bài để dự báo, xem đánh giá mô hình hoặc khám phá kết quả gom cụm.</p>
<div class="cards">{% for i,e in exercises.items() %}<section class="card"><span class="tag">{{ e.tag }}</span><h2>{{ i }}. {{ e.title }}</h2><p class="muted">{{ e.subtitle }}</p>
<p class="small mono">{{ e.dataset }}</p>{% if not available[i] %}<span class="tag missing">Chưa có dataset</span>{% endif %}
<a class="link" href="{{ url_for('exercise', number=i) }}">Mở bài {{ i }} →</a></section>{% endfor %}</div>
{% else %}
<div class="eyebrow">BÀI {{ number }} · {{ entry.tag }}</div><h1>{{ entry.title }}</h1><p class="muted">{{ entry.subtitle }} · <span class="mono">{{ entry.dataset }}</span></p>
{% if error %}<div class="warning" role="alert">{{ error }}</div>{% endif %}
<div class="grid"><section class="panel"><h2>{{ 'Dự báo mẫu mới' if number!=3 else 'Chọn đặc trưng' }}</h2>
<form method="post">
{% if number==1 %}<label for="age">Tuổi hành khách</label><input id="age" name="age" type="number" min="0" max="120" step="any" value="{{ form.get('age','30') }}" required>
{% elif number==2 %}<label for="weight">Trọng lượng cá (g)</label><input id="weight" name="weight" type="number" min="0.001" step="any" value="{{ form.get('weight','300') }}" required>
{% elif number==3 %}<label for="columns">Tên cột, ngăn cách bằng dấu phẩy</label><input id="columns" name="columns" value="{{ form.get('columns','') }}" placeholder="Để trống: tự chọn cột số"><p class="small muted">Bỏ các cột mã định danh. Kiểm tra danh sách đặc trưng hiển thị bên phải.</p>
{% else %}<label for="salary">Salaries</label><input id="salary" name="salary" type="number" step="any" value="{{ form.get('salary','60000') }}" required>
<label for="overtime">Overtime</label><input id="overtime" name="overtime" type="number" step="any" value="{{ form.get('overtime','3000') }}" required>{% endif %}
<button type="submit">{{ 'Gom 3 cụm' if number==3 else 'Dự báo' }}</button></form>
{% if prediction %}<div class="result">
{% if number==1 %}<span>Tiền vé dự báo</span><strong>{{ '%.2f'|format(prediction.fare) }}</strong><span class="small">Đơn vị gốc của cột fare</span>
{% elif number==2 %}<span>Chiều rộng dự báo</span><strong>{{ '%.3f'|format(prediction.width) }} cm</strong><p>Loài dự đoán: <b>{{ prediction.species }}</b></p><p class="small">Phiếu KNN: {{ '%.0f'|format(prediction.votes*100) }}% trong 5 láng giềng; không phải độ chính xác.</p>
{% else %}<span>Mẫu mới thuộc</span><strong>Cụm {{ prediction.cluster }}</strong>{% endif %}</div>{% if prediction.note %}<p class="small muted">{{ prediction.note }}</p>{% endif %}{% endif %}
<p class="small muted">{{ explanation }}</p></section>
<section class="panel"><h2>Kết quả mô hình</h2>{% if result %}
<div class="metrics"><div class="metric"><b>{{ '{:,}'.format(result.data|length) }}</b><span>Mẫu được sử dụng</span></div>
{% if number in [1,2] %}<div class="metric"><b>{{ '%.3f'|format(result.metrics.rmse) }}</b><span>RMSE trên test</span></div><div class="metric"><b>{{ '%.3f'|format(result.metrics.r2) }}</b><span>R² trên test</span></div>
{% if number==2 %}<div class="metric"><b>{{ '%.1f'|format(result.metrics.accuracy*100) }}%</b><span>Accuracy loài trên test</span></div>{% endif %}
{% else %}<div class="metric"><b>3</b><span>Số cụm K-Means</span></div><div class="metric"><b>{{ '%.2f'|format(result.inertia) }}</b><span>SSE đã chuẩn hóa</span></div>{% endif %}</div>
{% if number in [1,2] %}<p class="small muted">Train {{ result.train_size }} / test {{ result.test_size }}; loại {{ result.removed }} dòng thiếu hoặc không hợp lệ. Đánh giá trên test riêng; mô hình dự báo mới được học lại trên toàn bộ mẫu hợp lệ.</p>
{% else %}<p class="small muted">Đặc trưng: {{ result.columns|join(', ') }}. Điền thiếu: {{ result.missing_filled }} ô; loại {{ result.removed }} dòng. Nhãn cụm là tên nhóm, không phải thứ hạng.</p>{% endif %}
<img class="chart" src="{{ chart_url }}" alt="Biểu đồ kết quả bài {{ number }}">
{% if center_rows %}<h2>Tâm cụm ở đơn vị gốc</h2><div class="table-scroll"><table><thead><tr>{% for name in center_headers %}<th>{{ name }}</th>{% endfor %}</tr></thead><tbody>{% for row in center_rows %}<tr>{% for value in row %}<td>{{ value }}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div>{% endif %}
{% else %}<p class="muted">Chưa có kết quả để hiển thị. Bổ sung dataset hoặc kiểm tra thông báo phía trên, sau đó mở lại bài.</p>{% endif %}</section></div>
{% endif %}<footer>Lab 3 · Phần 5 · Các kết quả phục vụ bài thực hành học máy.</footer></main></body></html>"""


@lru_cache(maxsize=8)
def cached_model(number, mtime, size, columns):
    """mtime/size tham gia khóa cache dù không truyền vào hàm huấn luyện."""
    module = EXERCISES[number]["module"]
    if number == 3:
        return module.train_model(columns=list(columns) if columns else None)
    return module.train_model()


def get_model(number, columns=()):
    path = BASE_DIR / EXERCISES[number]["dataset"]
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có {path.name}. Đặt file đúng tên trong thư mục bài nộp rồi tải lại trang. Các bài khác vẫn sử dụng bình thường.")
    stat = path.stat()
    return cached_model(number, stat.st_mtime_ns, stat.st_size, columns)


def parse_columns(value):
    return tuple(column.strip() for column in value.split(",") if column.strip())


@app.route("/")
def index():
    available = {i: (BASE_DIR / e["dataset"]).is_file() for i, e in EXERCISES.items()}
    return render_template_string(TEMPLATE, title="Tổng quan", number=None,
                                  exercises=EXERCISES, available=available)


@app.route("/bai/<int:number>", methods=["GET", "POST"])
def exercise(number):
    if number not in EXERCISES:
        return "Không có bài tập này.", 404
    result, prediction, error = None, None, None
    columns = parse_columns(request.form.get("columns", "")) if number == 3 else ()
    try:
        result = get_model(number, columns)
        if request.method == "POST":
            if number == 1:
                prediction = titanic.predict_fare(result, request.form.get("age", ""))
            elif number == 2:
                prediction = fish.predict_fish(result, request.form.get("weight", ""))
            elif number == 4:
                values = [float(request.form.get("salary", "")), float(request.form.get("overtime", ""))]
                prediction = {"cluster": employee.predict_cluster(result, values),
                              "note": "Gán theo khoảng cách đến các tâm trong không gian chuẩn hóa."}
    except FileNotFoundError as exc:
        error = str(exc)
    except (ValueError, KeyError) as exc:
        error = f"Dữ liệu chưa hợp lệ: {exc}"

    headers, rows = [], []
    if result is not None and number in (3, 4):
        headers = ["Cụm", *result["columns"], "Số mẫu"]
        for label, center in result["centers"].iterrows():
            rows.append([int(label), *[f"{center[c]:,.2f}" for c in result["columns"]], f"{int(center['count']):,}"])
    explanation = {
        1: "Tuổi riêng lẻ có thể dự báo giá vé kém. Xem R² để đánh giá khả năng giải thích của mô hình.",
        2: "Chiều rộng dùng Linear Regression chung; loài dùng KNN với 5 láng giềng. Trọng lượng không xác định duy nhất một loài cá.",
        3: "K-Means dùng StandardScaler để các đặc trưng có thể so sánh về thang đo.",
        4: "Dùng toàn bộ bản ghi, giữ các giá trị âm gốc. Biểu đồ chỉ lấy tối đa 6.000 điểm để dễ quan sát.",
    }[number]
    return render_template_string(TEMPLATE, title=EXERCISES[number]["title"], number=number,
                                  entry=EXERCISES[number], exercises=EXERCISES,
                                  result=result, prediction=prediction, error=error,
                                  form=request.form, explanation=explanation,
                                  center_headers=headers, center_rows=rows,
                                  chart_url=url_for("chart", number=number, columns=",".join(columns)))


@app.route("/chart/<int:number>.png")
def chart(number):
    if number not in EXERCISES:
        return "Không có bài tập này.", 404
    try:
        columns = parse_columns(request.args.get("columns", "")) if number == 3 else ()
        result = get_model(number, columns)
    except (FileNotFoundError, ValueError, KeyError) as error:
        return Response(str(error), status=404, mimetype="text/plain")
    # Tạo PNG trong bộ nhớ; không phát sinh file tạm khi người dùng xem hình.
    with PLOT_LOCK:
        module = EXERCISES[number]["module"] if number in (1, 2) else employee
        fig = module.plot_result(result)
        buffer = BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=120)
        finally:
            plt.close(fig)
        return Response(buffer.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5001,
                        help="Mặc định 5001, khác cổng ví dụ phần 4 (5000).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Cổng phải từ 1 đến 65535.")
    app.run(host="127.0.0.1", port=args.port, debug=False)
