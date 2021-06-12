#! /usr/bin/env python3

import sys
from PyQt5.QtCore import Qt, QCoreApplication, QSize
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

import sounddevice as sd
import soundfile as sf
import time
from playsound import playsound
import librosa
import numpy as np  
from tensorflow import keras

  
  
def record():
    lb_start.setText("Start speaking!")
    app.processEvents()
    mydata = sd.rec(int(16000), samplerate=16000, channels=1, blocking=True) 
    sd.wait()
    lb_start.setText("Stop speaking!")
    sf.write('prediction.wav', mydata, 16000)
    pass
    
def listen():
    playsound('prediction.wav')
    pass     
    
def predict():
    test, test_rate = librosa.load('prediction.wav', sr = 16000)
    test_sample = librosa.resample(test, test_rate, 8000)
    audio = test_sample
    prob=model.predict(audio.reshape(1, 8000, 1))
    index=np.argmax(prob[0])
    print(prob)
    print(index)
    lb_prediction.setText("Prediction: " + labels[index])
    pass
     
def exit():
    QCoreApplication.quit()



model = keras.models.load_model('model.hdf5')

labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']


app = QApplication(sys.argv)

app_icon = QtGui.QIcon()
app_icon.addFile('gui/icons/16x16.png', QSize(16,16))
app_icon.addFile('gui/icons/24x24.png', QSize(24,24))
app_icon.addFile('gui/icons/32x32.png', QSize(32,32))
app_icon.addFile('gui/icons/48x48.png', QSize(48,48))
app_icon.addFile('gui/icons/256x256.png', QSize(256,256))
app.setWindowIcon(app_icon)

window = QWidget()
window.setWindowTitle('MUIU - Deep Learning for Speech Recognition')
window.resize(700,300)

lb_title = QLabel('<h1>Python Speech Recognition with Keras</h1>', parent=window)
lb_title.setAlignment(Qt.AlignCenter | Qt.AlignTop)
lb_title.setStyleSheet("font-family: Arial, Helvetica, sans-serif;") 

btn_record = QPushButton('Record')
btn_record.clicked.connect(record) 
btn_record.setMaximumWidth(200) 
btn_record.setStyleSheet("width: 200px; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif;") 

lb_start = QLabel('<h2 border="1px solid black"> </h2>', parent=window)
lb_start.setAlignment(Qt.AlignCenter)
lb_start.setFixedSize(200, 30)
lb_start.setStyleSheet("border: 1px solid black; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif; color: red") 

btn_listen = QPushButton('Listen')
btn_listen.clicked.connect(listen) 
btn_listen.setMaximumWidth(200) 
btn_listen.setStyleSheet("width: 200px; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif;") 

btn_predict = QPushButton('Predict')
btn_predict.clicked.connect(predict) 
btn_predict.setMaximumWidth(200) 
btn_predict.setStyleSheet("width: 200px; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif;") 

lb_prediction = QLabel('<h2 border="1px solid black"> </h2>', parent=window)
lb_prediction.setFixedSize(200, 30)
lb_prediction.setAlignment(Qt.AlignCenter)
lb_prediction.setStyleSheet("border: 1px solid black; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif;") 

btn_exit = QPushButton('Exit')
btn_exit.clicked.connect(exit) 
btn_exit.setMaximumWidth(200) 
btn_exit.setStyleSheet("width: 200px; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif;") 

lb_about = QLabel('<p>Made by Edvin <b>Teskeredzic</b> and Hana <b>Bezdrob</b>, 2021. <a href=\"https://eteskeredzic.github.io/">Click here to view the code</a></p>', parent=window)
lb_about.setOpenExternalLinks(True)

layout = QVBoxLayout()

layout.addWidget(lb_title, alignment=Qt.AlignCenter)
layout.addWidget(btn_record, alignment=Qt.AlignCenter)
layout.addStretch()
layout.addWidget(lb_start, alignment=Qt.AlignCenter)
layout.addStretch()
layout.addWidget(btn_listen, alignment=Qt.AlignCenter)
layout.addStretch()
layout.addWidget(btn_predict, alignment=Qt.AlignCenter)
layout.addStretch()
layout.addWidget(lb_prediction, alignment=Qt.AlignCenter)
layout.addStretch()
layout.addWidget(btn_exit, alignment=Qt.AlignCenter)
layout.addStretch()
layout.addWidget(lb_about)

window.setLayout(layout)
window.show()

sys.exit(app.exec_())
