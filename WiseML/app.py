import numpy as np

import tensorflow
from tensorflow import keras
from flask import Flask, render_template, request

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential

app = Flask(__name__)
model = keras.models.load_model('weapon_detection_model.h5')

labels_dict = ['gun', 'knife']

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def submit():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size = (100,100))
    image = img_to_array(image)
    #print(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    yhat = model.predict(image)
    label = np.argmax(yhat,axis = 1)[0]
    #print(label)
    classification = labels_dict[label]
    #if classification != 'gun' and classification != 'knife':
     #   classification = labels_dict[2]
    return render_template('index.html', prediction = classification)

if __name__ == "__main__":
    app.run(port = 3001, debug=True)