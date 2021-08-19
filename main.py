import os

import tensorflow as tf
import tensorflow_hub as hub
from flask import flash, request, redirect, url_for, render_template
from nudenet import NudeDetector
from werkzeug.utils import secure_filename
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig

from app import app
from image import evaluate_image
from audio import evaluate_audio

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_AUDIO_EXTENSIONS = set(['wav', 'mp3', 'm4a'])

image_model = tf.keras.models.load_model('models/imageV1.h5', custom_objects={'KerasLayer': hub.KerasLayer})
nudenet_model = NudeDetector()
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
audio_model = None

def load_language_model(model_save_path):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    language_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    language_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    language_model.load_weights(model_save_path)
    return language_model


language_model = load_language_model("models/bert_model.h5")


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


@app.route('/image')
def upload_form():
    return render_template('image.html')


@app.route('/image', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_image_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            score, reason = evaluate_image(filepath, image_model, nudenet_model)
            flash(filename + ":")
            flash("Score: " + str(score))
            flash("Reasons: " + reason)

        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    return render_template('image.html', filenames=file_names)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/audio')
def audio_page():
    return render_template('audio.html')


@app.route('/audio', methods=['POST'])
def upload_audio():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    file_names = []
    for file in files:
        if file and allowed_audio_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text, score, reason = evaluate_audio(filepath, language_model, yamnet_model, audio_model)
            flash(filename + ":")
            flash("Text: " + text)
            flash("Score: " + str(score))
            flash("Reasons: " + reason)

        else:
            flash('Allowed audio types are -> wav, mp3, m4a')
            return redirect(request.url)

    return render_template('audio.html', filenames=file_names)


@app.route('/text', methods=['GET'])
def evaluate_text():
    return render_template('text.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
