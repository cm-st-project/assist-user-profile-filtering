import io
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from better_profanity import profanity
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
import csv
from scipy.io import wavfile

from google.cloud import speech
import speech_recognition as sr

# Instantiates a speech recognition client
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="static/SpeechRecognition-4dd7671ba43f.json"
speech_client = speech.SpeechClient()

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

audio_classes = ['low', 'high']


def evaluate_keyword(text):
    print(profanity.contains_profanity(text))


def format_audio(input_path):
    output_path = 'static/tmp/tmp.wav'
    cmd = 'ffmpeg -i %s -ar 16000 -ac 1 -y %s' % (input_path, output_path)
    os.system(cmd)
    return output_path


def speech_recognition(filename):
    # The name of the audio file to transcribe
    with io.open(filename, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Detects speech in the audio file
    response = speech_client.recognize(config=config, audio=audio)

    file = sr.AudioFile(filename)
    with file as source:
        duration = file.DURATION

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    transcript = ""
    for result in response.results:
        transcript = transcript + result.alternatives[0].transcript
    return transcript, duration


# input is an array of text
def evaluate_text(text, language_model):
    ids = []
    masks = []
    sentences = [text]
    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, pad_to_max_length=True,
                                              return_attention_mask=True)
        ids.append(bert_inp['input_ids'])
        masks.append(bert_inp['attention_mask'])

        ids = np.asarray(ids)
        masks = np.array(masks)

        preds = language_model.predict([ids, masks], batch_size=32)
        pred_labels = np.argmax(preds.logits, axis=1)
        print("transcript:" + sent)
        print(preds)
        if pred_labels == 0:
            return 10, 'Please introduce yourself with details.'
        else:
            return 50, ''


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names


def evaluate_voice(audio_file, yamnet_model, audio_model):
    sample_rate, testing_wav_data = wavfile.read(audio_file, 'rb')
    scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
    mean_embeddings = np.array(tf.reduce_mean(embeddings, 0)).reshape(1, 1024)
    # result = audio_model(mean_embeddings).numpy()
    #
    # print(result.mean(axis=0))
    # inferred_class = audio_classes[result.mean(axis=0).argmax()]
    # print(f'The class is: {inferred_class}')
    inferred_class = 1

    scores_np = scores.numpy()
    class_names = class_names_from_csv(yamnet_model.class_map_path().numpy())
    audio_type = class_names[scores_np.mean(axis=0).argmax()]
    print(f'The main sound is: {audio_type}')
    # cannot detect human voice
    if audio_type != 'Speech':
        return 0, 'Cannot detect human voice.'
    elif inferred_class == 0:
        return 0, 'Please reduce background noise and/or improve your audio quality.'
    else:
        return 20, ''


# return a score of 0 to 100
def evaluate_audio(input_path, language_model, yamnet_model, audio_model):
    score = 0
    reason = ''
    # convert audio to WAV and 16000HZ mono
    audio_file = format_audio(input_path)
    # transcribe
    text, duration = speech_recognition(audio_file)
    duration_score = 0
    if duration < 3:
        return 10, 'Please talk about relevant skill/preference.'
    elif duration > 10:
        duration_score = 30
    else:
        duration_score = 30*(duration - 3)/(10 - 3)

    evaluate_keyword(text)
    text_score, text_reason = evaluate_text(text, language_model)
    audio_score, audio_reason = evaluate_voice(audio_file, yamnet_model, audio_model)

    score = round(text_score + audio_score + duration_score)
    reason = text_reason + audio_reason
    return text, score, reason
