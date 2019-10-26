from flask import Flask, request
import os

app = Flask(__name__)

# Setup folder
save_dir = os.path.join(os.path.dirname(__file__), 'imgs')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/send', methods=['POST'])
def send():
    file = request.files['image']
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)

    return 'Image saved.'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
