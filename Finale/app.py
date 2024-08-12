# from solver import Problem_solver, LSTMConfig  # Import LSTMConfig để đảm bảo nó có sẵn
# from flask import Flask, render_template, request, jsonify
# import os

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     data = request.json
#     text_input = data.get('text', '')
#     print(f"Received text: {text_input}")

#     # Đảm bảo rằng LSTMConfig có sẵn trong ngữ cảnh toàn cục
#     if 'LSTMConfig' not in globals():
#         from solver import LSTMConfig  # Import lại nếu cần
#     result = " "
#     result = Problem_solver(text_input)
#     print("Kết quả ở main nhận: ", result)
#     return jsonify(answer=result)

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)


from solver import Problem_solver, LSTMConfig  # Import LSTMConfig từ solver.py
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    text_input = data.get('text', '')
    print(f"Received text: {text_input}")

    # Gọi hàm Problem_solver để xử lý đầu vào
    result = Problem_solver(text_input)
    print("Kết quả ở main nhận: ", result)
    return jsonify(answer=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
