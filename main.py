import os
import json
from flask import Flask, request
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model
import numpy as np
import cv2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

        # Save image to server storage
        files = request.files.getlist('images')
        imgs = []
        outputs = []
        for file in files:
            save_path = os.path.join(save_dir, file.filename)
            file.save(save_path)

            # Read and resize image
            img = cv2.imread(save_path, cv2.IMREAD_COLOR)
            y, x, z = img.shape
            smol = min(x, y)
            cropx, cropy = smol, smol
            startx = x // 2 - cropx // 2
            starty = y // 2 - cropy // 2
            img = img[starty:starty + cropy, startx:startx + cropx]

            img = cv2.resize(img, (100, 100))
            cv2.imwrite('x' + file.filename, img)
            # imgs.append(img)
            img = np.resize(img, (1, 100, 100, 3))
            output = model.predict(img)
            outputs.append(output)

        # imgs = np.resize(imgs, (len(imgs), 100, 100, 3))

        # Predict and return result
        # outputs = model.predict(imgs)
        # s = []
        # m = { 0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
        # for output in outputs:
        #     s.append(m[np.argmax(output)])
        # return json.dumps(' '.join(s), cls=NumpyEncoder)

        # Predict and return result
        # outputs = model.predict(imgs)
        s = []
        for output in outputs:
            x = int(np.argmax(output))
            s.append(chr(x + 65))
        print(''.join(s))
        return json.dumps(''.join(s), cls=NumpyEncoder)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=True)
