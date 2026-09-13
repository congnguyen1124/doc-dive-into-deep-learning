# Dive into Deep Learning — vở học tiếng Việt

Đây là dự án biến `didl.pdf` thành một quyển vở học Machine Learning/Deep Learning dễ đọc cho người mới. **Chương 1–3, 14, 15 và 21 đã được biên soạn bằng tiếng Việt**; các chương/phụ lục còn lại vẫn là khung chờ cho các lần học tiếp theo.

## Chạy notebook reader

Yêu cầu: Python 3.10 trở lên.

```bash
python -m pip install -r requirements.txt
python app.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Có thể đổi cổng bằng `python app.py --port 8080`.

Reader có:

- mục lục theo đúng thứ tự chương trong PDF;
- giao diện ứng dụng bằng tiếng Anh, còn bài học trong Markdown bằng tiếng Việt;
- hai trang giấy note đặt cạnh nhau trên desktop và một trang trên mobile; nội dung tự dàn sang tờ kế tiếp, không cuộn dọc bên trong giấy;
- hiệu ứng lật có mặt trước, mặt sau và trang nằm dưới để nhìn thấy bài tiếp theo ngay khi kéo; vẫn dùng được nút hoặc phím `←`/`→`;
- **Style library**: sáu giao diện cổ trang Trung Hoa chọn được ngay trên thanh trên cùng — Secret Manual (武 bí kíp gỗ mun),
  Xuan Ink Wash (宣 giấy tuyên thuỷ mặc), Bamboo Slips (簡 thẻ tre), Blue and White (青 gốm thanh hoa),
  Dunhuang Fresco (煌 bích hoạ Đôn Hoàng) và Imperial Edict (敕 thánh chỉ); lựa chọn được ghi nhớ trên trình duyệt;
- mỗi giao diện có nhịp lật trang riêng (dựng giấy, cuộn thẻ tre, men gốm, lụa, gấm) với bóng nếp gấp,
  vệt sáng quét ngang mặt giấy và bóng đổ xuống tờ bên dưới;
- nút Light/Dark theme dùng chung cho mọi giao diện, tự nhớ lựa chọn trên trình duyệt;
- đánh dấu trang đang đọc và quay lại dấu gần nhất, kể cả sau khi mở lại trình duyệt;
- syntax highlighting Python chạy offline bằng Pygments, kèm quy ước tên/shape/comment dễ học;
- ghi nhớ chương đã học ngay trên trình duyệt;
- tooltip cho thuật ngữ tiếng Anh khi rê chuột hoặc dùng bàn phím focus;
- chú thích nguồn cho ảnh, phóng to ảnh và giao diện mobile;
- hỗ trợ công thức TeX qua MathJax (cần Internet để tải MathJax ở lần mở trang).

Các đoạn thực hành trong Chương 2–3 dùng PyTorch. Reader không cần PyTorch để mở; để chạy code học tập, cài bản PyTorch phù hợp với hệ điều hành/CUDA của bạn theo hướng dẫn chính thức của PyTorch.

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
- Dùng `<!-- pagebreak -->` để chia một file Markdown thành các cụm trang có chủ đích; reader tiếp tục tự phân trang nếu nội dung dài hơn một tờ giấy.
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
