# AGENTS.md

## Lưu ý cho Codex

- Dự án này dùng tiếng Việt trong notebook và tài liệu.
- Cẩn thận lỗi encoding tiếng Việt khi tạo hoặc sửa file bằng PowerShell/script.
- Không ghi notebook `.ipynb` bằng chuỗi JSON thủ công vì dễ làm hỏng xuống dòng hoặc biến ký tự tiếng Việt thành dấu hỏi.
- Khi sửa file có tiếng Việt, phải kiểm tra lại nội dung bằng Python đọc UTF-8 và xác minh chữ có dấu vẫn đúng.
- Với notebook, ưu tiên tạo hoặc sửa bằng `json` trong Python, ghi file bằng `encoding='utf-8'`, rồi kiểm tra JSON hợp lệ.
