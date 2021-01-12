import librosa, librosa.display
import matplotlib.pyplot as plt

file = "blues.00000.wav"

# waveform
librosa.load(file, sr=22050)