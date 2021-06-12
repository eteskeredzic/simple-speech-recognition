# simple-speech-recognition
Using deep learning for recognizing simple voice commands

## Model architecture
The implemented model uses a combination of convolutional and GRU (Gated Recurrent Unit) layers. It is based on current trends in deep learning for speech recognition, namely the state of the art models DeepSpeech2 and Wav2Letter++. The architecture of the model is as follows:

![image](https://github.com/eteskeredzic/simple-speech-recognition/tree/master/img/model.png)


## Data
The data used for training is the Speech Commands dataset, which contains 10 different voice commands in English (left, right, up, down, go, stop, yes, no, on, off). The commands are roughly 1 second long, and are spoken by many different people, so the model can generalize. This dataset is publicly available on Kaggle.

## GUI
A simple GUI app is also included, containing the pretrained network. This app will record your voice, and then predict the command you have given using the trained model. A preview of the GUI:


![image](https://github.com/eteskeredzic/simple-speech-recognition/tree/master/img/app.png)


