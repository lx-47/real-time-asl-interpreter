from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QTextEdit, QVBoxLayout, QGridLayout, QWidget, QLabel, QLineEdit
from PyQt5.QtGui import QFont, QPalette, QTextCursor, QIcon, QKeyEvent, QFontDatabase, QPainter, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer, QEvent, QPropertyAnimation, QEasingCurve
from time import sleep
import sys
import os
import re
from qtwidgets import AnimatedToggle
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import cv2
from speech_recognition import Recognizer, Microphone, UnknownValueError, RequestError
from pyautogui import screenshot
import pyttsx3

# function to get the absolute path to a resource
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Define the Worker class for gesture recognition
class Worker(QThread):
    update_signal = pyqtSignal(str)  # Signal for updating GUI with recognized gestures

    def __init__(self):
        super().__init__()
        try:
            # Initialize hand detector and classifier
            self.detector = HandDetector(maxHands=1)
            self.classifier = Classifier(resource_path("mmodel/keras_model.h5"), resource_path("mmodel/labels.txt"))
            self.offset = 20
            self.imgSize = 300
            self.labels = ["Yes", "I love you", "Thank you","No","Sorry","a","b","c","d","e","f","g","h","i","l","o","r","s","u","v","w","x","y","k","m","n","p","q","t","Hello"]
            self.is_recording = False  # Flag to control gesture recognition process
            self.last_prediction = None  # Store the last prediction
            self.engine = pyttsx3.init()  # Initialize TTS engine
        except Exception as e:
            print(f"Error occurred while initializing Worker: {e}")

    # Method to perform sign recognition
    def run(self):
        while self.is_recording:
            img = screenshot()  # Capture a screenshot
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert the image from RGB to BGR
            imgOutput = img.copy()
            hands, img = self.detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = self.imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                if prediction[index] > .90:
                    word = self.labels[index]
                    if word != self.last_prediction:
                        self.update_signal.emit(word)
                        print(f"Prediction: {word}, Confidence: {prediction[index] * 100:.2f}%")
                        self.last_prediction = word
                        self.text_to_speech(word)
            cv2.waitKey(1)

    def stop(self):
        self.is_recording = False
        self.quit()
        self.wait()

    # Method for text-to-speech conversion
    def text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


class TransparentWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        screen_geometry = QtWidgets.QApplication.desktop().availableGeometry()
        self.setGeometry(screen_geometry)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 100);")  # Adjust alpha value for transparency

        # Create back button
        self.back_button = QtWidgets.QPushButton("Back", self)
        self.back_button.setStyleSheet("QPushButton { background-color: #14213D; color: white; border-radius: 5px; padding: 5px 10px; }")
        self.back_button.clicked.connect(self.close)
        self.back_button.setGeometry(20, 20, 100, 40)
        self.back_button.clicked.connect(self.onBackClicked)

    backClicked = QtCore.pyqtSignal()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))  # Adjust alpha value as needed for background tint

    def onBackClicked(self):
        self.backClicked.emit()

class YourWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_recorder = AudioModule()  # Initialize the audio recorder for speech recognition
        self.audio_recorder.update_signal.connect(self.handle_audio)  # Connect signal for updating GUI with recognized speech
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)  # Initialize media player for video playback
        self.mediaPlayer.setVideoOutput(self.videoWidget)  # Set video output to the video widget   
        self.worker = Worker()  # Initialize worker for gesture recognition
        self.worker.update_signal.connect(self.update_text_area)

    def initUI(self):
        layout = QVBoxLayout()
        self.setWindowTitle("Real-time ASL Interpreter")
        self.setGeometry(0, 0, 1920, 1080)
        self.background_image = QtGui.QPixmap("hm.png")
        self.background_label = QtWidgets.QLabel(self)
        self.background_label.setPixmap(self.background_image)
        self.background_label.setScaledContents(True)
        self.background_label.setGeometry(0, 0, 1920, 1080)

        self.button = AnimatedToggle(self, pulse_checked_color="#14213D")
        self.button.setFixedSize(70, 50)
        self.button_3D = AnimatedToggle(self, pulse_checked_color="#14213D")
        self.button_3D.setFixedSize(70, 50)
        self.text_3D = AnimatedToggle(self, pulse_checked_color="#14213D")
        self.text_3D.setFixedSize(70, 50)

        self.videoWidget = QVideoWidget()
        self.videoWidget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.videoWidget.setGeometry(0, self.height() - 700, 400, 400)
        self.videoWidget.hide()

        self.button.clicked.connect(self.showSignToTextWindow)
        self.button_3D.clicked.connect(self.showAudioToASLWindow)
        self.text_3D.clicked.connect(self.showTextToASLWindow)

        id = QFontDatabase.addApplicationFont("Inter-ExtraBold.ttf")
        families = QFontDatabase.applicationFontFamilies(id)

        self.label_1 = QtWidgets.QLabel("Sign to Text", self)
        self.label_1.setFont(QtGui.QFont(families[0], 24))
        self.label_1.setStyleSheet("color: #14213D")
        self.label_2 = QtWidgets.QLabel("Audio to ASL", self)
        self.label_2.setFont(QtGui.QFont(families[0], 24))
        self.label_2.setStyleSheet("color: #14213D")
        self.label_3 = QtWidgets.QLabel("Text to ASL", self)
        self.label_3.setFont(QtGui.QFont(families[0], 24))
        self.label_3.setStyleSheet("color: #14213D")

        self.box = QtWidgets.QGroupBox(self)
        self.box.setTitle("")
        self.box.setStyleSheet("QGroupBox { background-color: white; border-radius: 10px; }")
        self.box.setFixedSize(600, 600)

        layout = QtWidgets.QGridLayout(self.box)
        layout.addWidget(self.label_1, 0, 0)
        layout.addWidget(self.button, 0, 1)
        layout.addWidget(self.label_2, 1, 0)
        layout.addWidget(self.button_3D, 1, 1)
        layout.addWidget(self.label_3, 2, 0)
        layout.addWidget(self.text_3D, 2, 1)

        shadow_effect = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow_effect.setBlurRadius(20)
        shadow_effect.setColor(QtGui.QColor("#14213D"))
        shadow_effect.setOffset(5, 5)
        self.box.setGraphicsEffect(shadow_effect)

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addStretch()
        vertical_layout.addWidget(self.box)

        self.box.setLayout(vertical_layout)

        horizontal_layout = QtWidgets.QHBoxLayout(self)
        horizontal_layout.addStretch()
        horizontal_layout.addWidget(self.box)
        horizontal_layout.addItem(QtWidgets.QSpacerItem(150, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.setLayout(horizontal_layout)
        
    # words and letters video paths for video playback
        self.words = {
            "night" : resource_path("model/words/night.mkv"),
            "became" : resource_path("model/words/became.mkv"),
            "family" : resource_path("model/words/family.mkv"),
            "question" : resource_path("model/words/question.mkv"),
            "study" : resource_path("model/words/study.mkv"),
            "tomorrow" : resource_path("model/words/tomorrow.mkv"),
            "name" : resource_path("model/words/name.mkv"),  
            "only" : resource_path("model/words/only.mkv"),
            "see" : resource_path("model/words/see.mkv")                  
        }
        self.letters = {
            "a": resource_path("model/letters/a1.mkv"),
            "b": resource_path("model/letters/b1.mkv"),
            "c": resource_path("model/letters/c1.mkv"),
            "d": resource_path("model/letters/d1.mkv"),
            "e": resource_path("model/letters/e1.mkv"),
            "f": resource_path("model/letters/f1.mkv"),
            "g": resource_path("model/letters/g1.mkv"),
            "h": resource_path("model/letters/h1.mkv"),
            "i": resource_path("model/letters/i1.mkv"),                        
            "j": resource_path("model/letters/j1.mkv"),
            "k": resource_path("model/letters/k1.mkv"),
            "l": resource_path("model/letters/l1.mkv"),
            "m": resource_path("model/letters/m1.mkv"),
            "n": resource_path("model/letters/n1.mkv"),
            "o": resource_path("model/letters/o1.mkv"),
            "p": resource_path("model/letters/p1.mkv"),
            "q": resource_path("model/letters/q1.mkv"), 
            "r": resource_path("model/letters/r1.mkv"),
            "s": resource_path("model/letters/s1.mkv"), 
            "t": resource_path("model/letters/t1.mkv"),
            "u": resource_path("model/letters/u1.mkv"),
            "v": resource_path("model/letters/v1.mkv"),
            "w": resource_path("model/letters/w1.mkv"),
            "x": resource_path("model/letters/x1.mkv"),
            "y": resource_path("model/letters/y1.mkv"), 
            "z": resource_path("model/letters/z1.mkv"),                                                  
        }         

    def toggle_recording(self):
        if not self.worker.isRunning():
            self.text_area.show()
            self.worker.is_recording = True
            self.worker.start()
        else:
            self.text_area.hide()
            self.worker.stop()

    def update_text_area(self, text):
        self.text_area.moveCursor(QTextCursor.End)  
        self.text_area.insertPlainText(" " + text)
           
    def showSignToTextWindow(self, checked):
        self.sign_to_text_window = TransparentWindow(self)
        self.sign_to_text_window.show()
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setGeometry(150, self.height() - 200, self.width() - 350, 50)
        self.text_area.hide()
        font = QFont("Arial", 25)
        self.text_area.setFont(font)
        self.text_area.setTextColor(Qt.white)
        palette = QPalette()
        palette.setColor(QPalette.Base, Qt.gray)
        self.text_area.setPalette(palette)        
        self.text_area.show()

        self.worker.is_recording = True
        self.worker.start()
        self.sign_to_text_window.backClicked.connect(self.closeSignToTextWindow)

    def closeSignToTextWindow(self):
        self.worker.stop()
        self.sign_to_text_window.close()
        self.text_area.close()
        self.button.setChecked(False)

    def handle_audio(self, text):
        lower_text = text.lower()    
        pattern = r'\b\w+\b'
        words = re.findall(pattern, lower_text)

        def spell_out_word(word_to_spell):
            for char in word_to_spell:
                if char.isalpha() and char in self.letters:
                    self.play_video(self.letters[char])
                    sleep(2)
                    self.play_video(resource_path('default.mkv'))
                    
                else:
                    self.play_video(resource_path('default.mkv'))
                    sleep(2)    

        for i, word in enumerate(words):
                if word in self.words:
                    self.play_video(self.words[word])
                    sleep(2)
                    self.play_video(resource_path('default.mkv'))
                else:                        
                    spell_out_word(word)

    def showAudioToASLWindow(self, checked):
        self.audio_to_asl_window = TransparentWindow(self)
        self.audio_to_asl_window.show()
        self.audio_recorder.start_recording()
        self.worker.is_recording = True        
        self.videoWidget.show()
        self.play_video(resource_path("default.mkv"))
        self.audio_to_asl_window.backClicked.connect(self.closeAudioToAslWindow)

    def closeAudioToAslWindow(self):
        self.audio_recorder.stop_recording()
        self.worker.stop()
        self.audio_to_asl_window.close()
        self.videoWidget.hide()
        self.button_3D.setChecked(False)

    def play_video(self, video_path):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.mediaPlayer.play()

    def handle_text(self):
        text = self.text_edit.text()
        lower_text = text.lower()    
        pattern = r'\b\w+\b'
        new_words = re.findall(pattern, lower_text)
        new_words_set = set(new_words)
        previous_words_set = set(re.findall(pattern, self.previous_text.lower()))
        added_words = list(new_words_set - previous_words_set)
        self.previous_text = text

        def spell_out_word(word_to_spell):
            for char in word_to_spell:
                if char.isalpha() and char in self.letters:
                    self.play_video(self.letters[char])
                    sleep(2)
                    self.play_video(resource_path('default.mkv'))
                else:
                    self.play_video(resource_path('default.mkv'))
                    sleep(2)    

        for word in new_words:
            if word in self.words:
                self.play_video(self.words[word])
                sleep(2)
                self.play_video(resource_path('default.mkv')) 
            else:       
                spell_out_word(word) 

    def showTextToASLWindow(self, checked):
        self.text_to_asl_window = TransparentWindow(self)
        self.text_to_asl_window.show()
        self.videoWidget.show()
        self.text_edit = QLineEdit()
        self.text_edit.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.text_edit.setGeometry(self.width() - 470, self.height() - 400, 400, 200)
        font = QFont("Arial", 16)
        self.text_edit.setFont(font)
        palette = QPalette()
        palette.setBrush(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Base, Qt.gray)
        self.text_edit.setPalette(palette)
        self.previous_text = ""
        self.text_edit.editingFinished.connect(self.handle_text)         
        self.text_edit.show()
        self.play_video(resource_path("default.mkv"))
        self.text_to_asl_window.backClicked.connect(self.closeTextToAslWindow)

    def closeTextToAslWindow(self):
        self.text_edit.close()
        self.text_to_asl_window.close()
        self.videoWidget.hide()
        self.text_3D.setChecked(False)

class AudioModule(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.recognizer = Recognizer()
        self.microphone = Microphone()

    def run(self):
        while self.is_recording:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.listen(source)
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    self.update_signal.emit(text)
                    print(text)
                except UnknownValueError:
                    print("$")
                except RequestError as e:
                    print("cant read")

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    splash_pix = QtGui.QPixmap('splashf.png')
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    opaqueness = 0.0
    step = 0.1
    splash.setWindowOpacity(opaqueness)
    splash.show()

    while opaqueness < 1:
        splash.setWindowOpacity(opaqueness)
        sleep(step)
        opaqueness += step

    sleep(1)
    splash.close()

    widget = YourWidget()
    widget.show()
    app.exec_()
