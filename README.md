# Dive into Deep Learning — vở học tiếng Việt

Đây là khung dự án để biến `didl.pdf` thành một quyển vở học Machine Learning/Deep Learning dễ đọc cho người mới. Hiện tại dự án mới ở giai đoạn **khởi tạo**: 21 chương và 2 phụ lục đã có đúng tên theo mục lục PDF, nhưng chưa dịch nội dung.

## Chạy notebook reader

Yêu cầu: Python 3.10 trở lên.

```bash
python -m pip install -r requirements.txt
python app.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Có thể đổi cổng bằng `python app.py --port 8080`.

Reader có:

- mục lục theo đúng thứ tự chương trong PDF;
- tìm kiếm chương, phím `←`/`→` để lật trang, và hiệu ứng lật trang;
- ghi nhớ chương đã học ngay trên trình duyệt;
- tooltip cho thuật ngữ tiếng Anh khi rê chuột hoặc dùng bàn phím focus;
- chú thích nguồn cho ảnh, phóng to ảnh và giao diện mobile;
- hỗ trợ công thức TeX qua MathJax (cần Internet để tải MathJax ở lần mở trang).

## Kiểm tra dự án

```bash
python app.py --check
python -m unittest discover -s tests -v
```

Lệnh `--check` kiểm tra metadata, tên file, phạm vi trang, thứ tự chương và các khóa glossary được dùng trong Markdown.

## Quy ước nội dung

- Mỗi chương nằm trong `content/chapters/NN - Exact Chapter Title.md`.
- Nội dung sẽ được viết bằng tiếng Việt, nhưng tên chương/file giữ nguyên tiếng Anh đúng như PDF.
- Thuật ngữ đặc biệt dùng cú pháp `{{term:tensor|Tensor}}`; phần giải thích nằm trong `content/glossary.json`.
- Ảnh gốc từ sách nằm trong `content/assets/chapter-NN/` và luôn có thông tin trang nguồn.
- Sơ đồ do dự án tạo thêm nằm trong thư mục con `generated/` và phải ghi rõ là sơ đồ bổ sung.

Xem [AGENT.md](AGENT.md) để biết đầy đủ tiêu chuẩn dịch, giảng giải, bài tập, hình ảnh và kiểm tra chất lượng cho các lần biên soạn tiếp theo.

## Trích hình chính xác từ PDF

Ví dụ render và crop một vùng theo phần trăm của **trang PDF vật lý**:

```bash
python tools/extract_pdf_figure.py \
  --pdf ../didl.pdf \
  --pdf-page 275 \
  --book-page 235 \
  --crop 8,18,84,48 \
  --figure-id "Figure 7.1" \
  --caption "Correlation and convolution" \
  --output content/assets/chapter-07/figure-7-1.png
```

Script tạo ảnh PNG và file `.source.json` bên cạnh để lưu PDF nguồn, checksum, trang, DPI và vùng crop. Cần kiểm tra trực quan ảnh sau khi trích.
