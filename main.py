import os
import json
from flask import Flask, request
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model
import cv2

# Helper class to convert Numpy array to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Make image aspect ratio 1:1
def squarify(img):
    y, x, c = img.shape
    smol = min(x, y)
    start_x = x // 2 - smol // 2
    start_y = y // 2 - smol // 2
    return img[start_y:start_y + smol, start_x: start_x + smol]

# Setup folder
base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir, 'imgs')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Setup session
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

# Setup model
model_path = os.path.join(base_dir, 'asl_classifier_v1.h5')
model = load_model(model_path)

# Setup Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/send', methods=['POST'])
def send():
    with graph.as_default():
        set_session(sess)

        files = request.files.getlist('images')
        imgs = []
        outputs = []
        for file in files:
            # Save image to server storage
            save_path = os.path.join(save_dir, file.filename)
            file.save(save_path)

            # Read and resize image
            img = cv2.imread(save_path, cv2.IMREAD_COLOR)
            img = squarify(img)
            img = cv2.resize(img, (100, 100))
            cv2.imwrite('x' + file.filename, img)
            img = np.resize(img, (1, 100, 100, 3))

            # Predict image and add to output
            output = model.predict(img)
            outputs.append(output)

        # Convert output to array of strings
        s = []
        for output in outputs:
            x = int(np.argmax(output))
            s.append(chr(x + 65)) # 65 = A
        print(''.join(s))

        # Return the JSON
        return json.dumps(''.join(s), cls=NumpyEncoder)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=True)
