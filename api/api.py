#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:38:06 2018

@author: anubhav
"""

# =============================================================================
# import flask
# 
# app = flask.Flask(__name__)
# app.config["DEBUG"] = True
# 
# 
# @app.route('/', methods=['GET'])
# def home():
#     return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"
# 
# app.run()
# 
# =============================================================================

from __future__ import print_function

from flask import Flask, jsonify, request
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import pdb
import tensorflow as tf

import audioread
import sys
import os
import wave
import contextlib
import matplotlib.pyplot as plt
import wave
from scipy.signal import butter, lfilter
from scipy import signal
import calendar;
import time;

from werkzeug.utils import secure_filename

from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
#app.config["DEBUG"] = True

# =============================================================================
# Load Saved CNN Model
# =============================================================================
model = load_model('phase3_model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
graph = tf.get_default_graph()

classHash = {
    0 : 'Open Fist',
    1 : 'Closed Fist'
}


# =============================================================================
# Decode .AAC Audio into .WAV Audio File
# filename => input .aac audio file
# =============================================================================
def decode(filename, isCollect):
#    filename = os.path.abspath(os.path.expanduser(filename))
    if not filename:
        print("File not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with audioread.audio_open(os.path.join("collected-sample/" if isCollect else "decoded-input/" , filename)) as f:
            print('Input file: %i channels at %i Hz; %.1f seconds.' %
                  (f.channels, f.samplerate, f.duration),
                  file=sys.stderr)
            print('Backend:', str(type(f).__module__).split('.')[1],
                  file=sys.stderr)
            filename = filename.split('/')
            newFileName = os.path.join("collected-sample/" if isCollect else "decoded-input/" , filename[-1])
            with contextlib.closing(wave.open(newFileName + '.wav', 'w')) as of:
                of.setnchannels(f.channels)
                of.setframerate(f.samplerate)
                of.setsampwidth(2)

                for buf in f:
                    of.writeframes(buf)

    except audioread.DecodeError:
        print("File could not be decoded.", file=sys.stderr)
        sys.exit(1)
# =============================================================================
# End
# =============================================================================
     
        
# =============================================================================
# Butter Band Pass Filter 
# data => Input signal
# lowcut => Lower CutOff Frequency
# highcut => Higher CutOff Frequency
# fs => Sampling Frequency
# order => Filter Order
# =============================================================================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
# =============================================================================
# End
# =============================================================================


# =============================================================================
# Process input audio and creates a spectrogram
# =============================================================================
def process_audio(inputAudio, isCollect):
        print("\n\n=============================================================================")
        print("----------Processing Start---------")
        # Set Sampling, lower cutoff, higher cutoff Frequencies
        fs = 48e3
        lowcut = 20000.0
        highcut = 22000.0
        
        decode(inputAudio, isCollect)                
        # Direct WAV Audio Processing
        directSig = wave.open('no-obstacle.aac.wav','r')
        directSig = directSig.readframes(-1)
        directSig = np.fromstring(directSig, 'Int16')
        directSigFiltered = butter_bandpass_filter(directSig, lowcut, highcut, fs, order=6)
        
        # Test WAV Audio Processing
        testSig = wave.open(os.path.join("collected-sample/" if isCollect else "decoded-input/", inputAudio) + '.wav','r')
        testSig = testSig.readframes(-1)
        testSig = np.fromstring(testSig, 'Int16')                
        testSigFiltered = butter_bandpass_filter(testSig, lowcut, highcut, fs, order=6)
        
        # Cross Correlation of Test Signal w.r.t Direct Signal
        correlated = signal.correlate(testSigFiltered, directSig, mode='same')
        
        fig = plt.figure(1)
        # Plot Spectrogram of Correlated Signal
        Pxx, freqs, bins, im = plt.specgram(correlated, NFFT=128, Fs=1000, 
                                            window=np.hanning(128), 
                                            noverlap=127)
        
        plt.title('Spectrogram of Correlated Signal')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        # Save the Plot of Spectrogram of Correlated Signal
        fig.savefig(os.path.join('collected-spectrograms/' if isCollect else 'spectrograms/' , inputAudio) + '-spectrogram.jpg' , dpi=1024)



        print("----------Processing Done---------")
        print("=============================================================================")
        
        # Return Parameters
        result = {
            'fileName' : os.path.join('collected-spectrograms/' if isCollect else 'spectrograms/' , inputAudio) + '-spectrogram.jpg',
            'success' : True
        }
        return result
# =============================================================================
# End
# =============================================================================


# =============================================================================
# POST API for predicting the gesture corresponding to the input .aac audio file
# =============================================================================
@app.route('/predict', methods=["POST"])
def predict_gesture():
        # Preprocess the image so that it matches the training input
        print("\n\n=============================================================================")
        print("----------Prediction Start---------")
#        audio = request.files.to_dict()
#        audio = audio['audio']
        audio = request.files['audio']
        print("Audio name : ")
        ts = calendar.timegm(time.gmtime())
        filename = secure_filename("sample-" + str(ts) + "-" +audio.filename)
        print(filename)
        print(audio)
                
        audio.save(os.path.join('decoded-input/', filename))  
        # Preprocess the image so that it matches the training input
        response = process_audio(filename, False)
        print(response)
        if(response['success']):            
            image = Image.open(response['fileName'])
            print("----------Image Read Successful---------")
            image = np.asarray(image.resize((128,128)))
            image = image.reshape(1,128,128,3)
            print("----------Image Resize Successful---------")
            # Use the loaded model to generate a prediction.
            global graph
            with graph.as_default():
                pred = model.predict_classes(image)
            # Prepare and send the response.
            print("Preditions Output")
            print(pred)
            prediction = {
                'class' : classHash[pred[0][0]],
                'success' : True
            }
        else:    
            prediction = {
                'success' : False
            }
        print("----------Prediction Done---------")
        print("=============================================================================")
        return jsonify(prediction)
# =============================================================================
# End
# =============================================================================    
    
    
# =============================================================================
# POST API for uploading the sample input .aac audio file
# =============================================================================
@app.route('/audio-uploader', methods=["POST"])
def upload_audio():
        # Preprocess the image so that it matches the training input
        print("\n\n=============================================================================")
        print("----------Upload Start---------")
        audio = request.files['audio']
        print("Audio name : ")
        ts = calendar.timegm(time.gmtime())
        filename = secure_filename("sample-" + str(ts) + "-" +audio.filename)
        print(filename)
        print(audio)
        try:
            audio.save(os.path.join('collected-sample/', filename))
            response = process_audio(filename, True)
        except:
            response = {
                    'success' : False
                    }
        print("----------Upload Done---------")
        print("=============================================================================")
        return jsonify(response)
# =============================================================================
# End
# =============================================================================
       
        
# =============================================================================
# Test API 
# =============================================================================
@app.route('/testing', methods=["GET"])
def testing():
        # Preprocess the image so that it matches the training input
        print("\n\n=============================================================================")
        print("----------Test Start---------")
        response = {
                'success' : True
                }
        print("----------Test Done---------")
        print("=============================================================================")
        return jsonify(response)
# =============================================================================
# End
# =============================================================================


if __name__ == "__main__":
        app.run(host='10.1.203.241')