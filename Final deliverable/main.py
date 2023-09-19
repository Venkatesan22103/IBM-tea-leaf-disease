from flask import Flask, request, render_template,send_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys
import os
import glob
import re
import numpy as np
import os



# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
def model_predict(img_path, model):
    xtest_image = load_img(img_path, target_size=(224, 224))
    xtest_image = img_to_array(xtest_image)
    xtest_image = np.expand_dims(xtest_image, axis = 0)
    preds = model.predict(xtest_image)
    return preds
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/1.jpg')
def get_image():
    return send_from_directory('static', 'images/1.jpg')
@app.route('/home.html')
def home_():
    return render_template("home.html")
@app.route('/base.html')
def home__():
    return render_template("base.html",name = "")
@app.route('/pre.html')
def home___():
    return render_template("pre.html")
@app.route('/uploads', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        # Save the file to a desired location
        file.save('uploads/' + file.filename)
        import tensorflow as tf

# Load the model from a saved file
        model = load_model('C:\\Users\\venki\\Documents\\NanMudhalvan\\NanMudhalvan\\Final deliverable\\my_model.h5')

       
        c = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']
        # Preprocess the input image
        predictions =  model_predict('uploads/' + file.filename, model)
        print(predictions)
        predictions = list(predictions[0])
        return render_template("base.html",name = c[predictions.index(max(predictions))])
            
    

    return 'File uploaded successfully!'

if __name__ == '__main__':
    app.run(debug=True)