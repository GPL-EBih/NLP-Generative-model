# 1. Sử dụng hình ảnh cơ sở từ Python
FROM nlp-lstm:latest

# 2. Thiết lập thư mục làm việc trong container
WORKDIR /app

# 3. Sao chép tệp yêu cầu và cài đặt các phụ thuộc
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Sao chép mã nguồn vào thư mục làm việc
COPY  . .

# 5. Mở cổng ứng dụng (nếu cần)
EXPOSE 5000

# 6. Lệnh để chạy ứng dụng khi container khởi động
CMD ["python", "app.py"]
