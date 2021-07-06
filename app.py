from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from keras.models import load_model as lm
import numpy as np
from PIL import Image
import cv2

def load_model():
    global model
    model = lm('model.h5')
    model.make_predict_function()  

def model_predict(img_path, model):

    image = Image.open(img_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = np.asarray(image)
    image = cv2.resize(image, (64,64))
    
    image = image / 255
    image = np.expand_dims(image, axis = 0)

    preds = model.predict(image)

    if preds[0]<0:
        return "Female"
    else:
        return "Male"


app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        return preds
    return None

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
