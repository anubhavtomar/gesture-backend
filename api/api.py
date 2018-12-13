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


from flask import Flask, jsonify, request
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import pdb
import tensorflow as tf

from __future__ import print_function
import audioread
import sys
import os
import wave
import contextlib
import matplotlib.pyplot as plt
import wave
from scipy.signal import butter, lfilter
from scipy import signal

app = Flask(__name__)

model = load_model('python_model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
graph = tf.get_default_graph()

classHash = {
    0 : 'Sitting',
    1 : 'Standing'
}


# =============================================================================
# Decode .AAC Audio into .WAV Audio File
# filename => input .aac audio file
# =============================================================================
def decode(filename):
    filename = os.path.abspath(os.path.expanduser(filename))
    if not os.path.exists(filename):
        print("File not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with audioread.audio_open(filename) as f:
            print('Input file: %i channels at %i Hz; %.1f seconds.' %
                  (f.channels, f.samplerate, f.duration),
                  file=sys.stderr)
            print('Backend:', str(type(f).__module__).split('.')[1],
                  file=sys.stderr)
            filename = filename.split('/')
            newFileName = os.path.join("../../decoded-input/" , filename[-1])
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
# Post API for predicting the gesture corresponding to the input .aac audio file
# =============================================================================
@app.route('/predict', methods=["POST"])
def predict_image():
    
        # Preprocess the image so that it matches the training input
        print("\n\n=============================================================================")
        print("----------Prediction Start---------")
        audio = request.files.to_dict()
        audio = image['audio']
        print("Audio name : ")
        image = process_audio(audio)
        decode(os.path.join("../../decoded-input/" , file))
        print(image)
        
        image = Image.open(image)
        print("----------Image Read Successful---------")
        image = np.asarray(image.resize((128,128)))
#        pdb.set_trace()
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
        print("----------Prediction Done---------")
        print("=============================================================================")
        return jsonify(prediction)
# =============================================================================
# End
# =============================================================================
    
    
# =============================================================================
# Process input audio and creates a spectrogram
# =============================================================================
def process_audio():
        # Preprocess the image so that it matches the training input
        print("\n\n=============================================================================")
        print("----------Processing Start---------")


        for file in os.listdir("../../decoded-input"):
            os.remove(os.path.join("../../decoded-input/" , file))
            
        for file in os.listdir("../../audio-recording-input"):
            if file.endswith(".aac"):
                print(os.path.join(file))
                
                decode(os.path.join("../../audio-recording-input/" , file))
                
                #Fetch WAV Audio
                directSig = wave.open('../no-obstacle.aac.wav','r')
                 
                directSig = directSig.readframes(-1)
                directSig = np.fromstring(directSig, 'Int16')
                                
                fs = 48e3
                lowcut = 20000.0
                highcut = 22000.0
                
                directSigFiltered = butter_bandpass_filter(directSig, lowcut, highcut, fs, order=6)
                
                testSig = wave.open(os.path.join("../../decoded-input/" , file) + '.wav','r')
                
                testSig = testSig.readframes(-1)
                testSig = np.fromstring(testSig, 'Int16')                
                
                testSigFiltered = butter_bandpass_filter(testSig, lowcut, highcut, fs, order=6)
                                
                correlated = signal.correlate(testSigFiltered, directSigFiltered, mode='same')
                fig = plt.figure(1)
                Pxx, freqs, bins, im = plt.specgram(correlated, NFFT=128, Fs=fs, 
                                                    window=np.hanning(128), 
                                                    noverlap=127)
                
                plt.title('Spectrogram of Correlated Signal')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
        
                fig.savefig(os.path.join(file) + '-spectrogram.jpg' , dpi=128)



        print("----------Processing Done---------")
        print("=============================================================================")
        return 

if __name__ == "__main__":
        app.run()