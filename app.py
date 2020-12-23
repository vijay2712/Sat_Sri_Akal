from flask import Flask , render_template , request, redirect
from datetime import datetime
from model_files import NRI_NDVI, RGB_VARI, stiching
import cv2 , os
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pandas as pd
import numpy as np
from keras.preprocessing import image
from glob import glob

app = Flask(__name__)
app.config["NRI_UPLOADS"] = "NRI_image/"
app.config["RGB_UPLOADS"] = "RGB_image/"
app.config['PLANT_UPLOAD'] = "Plant_image/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.route('/', methods=['POST' , 'GET'])
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/NRI_NDVI', methods=['POST' , 'GET'])
def nir_ndvi():
    if request.method == 'POST':
        image = request.files["nriImage"]
        if image and allowed_file(image.filename):
            image.save(os.path.join(app.config["NRI_UPLOADS"], image.filename))
            img = read_image('NRI_image' , image.filename )
            output_image = NRI_NDVI.Ndvi(img , image.filename)
            try:
                return render_template('/NRI_NDVI.html' , image_name = output_image)
            except Exception as e:
                print(e)
                return 'There was an error adding your image {}'.format(e)
    else:
        return render_template('/NRI_NDVI.html')


@app.route('/Rgb_Vari', methods=['POST' , 'GET'])
def rgb_vari():
    if request.method == 'POST':
        image = request.files["rgbImage"]
        if image and allowed_file(image.filename):
            image.save(os.path.join(app.config["RGB_UPLOADS"], image.filename))
            img = read_image('RGB_image' ,image.filename )
            output_image = RGB_VARI.RGB(img , image.filename)
            print(output_image)
            try:
                return render_template('/Rgb_Vari.html' , image_name = output_image)
            except Exception as e:
                print(e)
                return 'There was an error adding your image {}'.format(e)
    else:
        return render_template('/Rgb_Vari.html')


def read_image(folder , image):
    img = cv2.imread('{}\{}'.format(folder,image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@app.route('/DiseaseClassifier')
def classifier():
    return render_template('/DiseaseClassifier.html')

def load_model():
    file = open('./model_files/layers.txt', 'r')
    model_json = file.read()
    file.close()

    loaded_model = model_from_json(model_json)
    print("checking")
    # load weights
    loaded_model.load_weights('./final_model.h5')

    return loaded_model


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['filename']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config["PLANT_UPLOAD"], file.filename))
            img = image.load_img('Plant_image/{}'.format(file.filename), target_size=(224, 224))
            x = image.img_to_array(img)
            x = x / 255
            x = np.vstack([x])
            img = np.expand_dims(x, axis=0)
            model = load_model()
            classes = model.predict(img)
            if classes < 0.5:
                result = 'DISEASED'
            else:
                result = 'HEALTHY'

        return render_template('/DiseaseClassifier.html', prediction=result)
    else:
        return render_template('/DiseaseClassifier.html')


@app.route('/stitching', methods=['GET', 'POST'])
def image_stitch():
    if request.method == 'POST':
        folder_name = request.form['folder_name']
        folder_ =  'Images_to_stitch/'+folder_name+'/'
        if not os.path.exists(folder_):
            os.makedirs(folder_)

        uploaded_files = request.files.getlist("images")
        for image in uploaded_files:
            if image and allowed_file(image.filename):
                image.save(os.path.join(folder_, image.filename))
        cv_img = []
        for img in glob('{}*.jpg'.format(folder_)):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            cv_img.append(img)
        output=stiching.Stich(cv_img , folder_name)
        print(output)
        if output.find('.jpg') == -1:
            error = output
            return render_template('Stitch.html',error = error)
        else:
            output_image = output
            return render_template('Stitch.html',output_image = output_image)
    return render_template('Stitch.html')


if __name__=="__main__":
    app.run(debug = True)