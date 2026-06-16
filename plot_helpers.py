import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

def plot_signal(t, signal, title):
    plt.figure(figsize=(12,4))
    plt.plot(t, signal)    
    plt.xlabel("time (s)")
    plt.ylabel("Signal")
    plt.title(title)
    plt.show() 

def plot_emg_vs_joystick(t, emg, y):
    plt.figure(figsize=(12,4))
    plt.plot(t, emg)
    plt.plot(t, y)
    plt.xlabel("time (s)")
    plt.title("Smoothed EMG")
    plt.show()

def plot_frequencies(signal, fs, center=False, title=""):
    if center:
        signal = signal - np.mean(signal)

    N = len(signal)

    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    magnitude = np.abs(fft_vals)
    
    plt.figure(figsize=(10,5))
    plt.plot(freqs, magnitude)
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    if title:
        plt.title(title)
    else:
        plt.title("Signal Frequency Spectrum")
    
    plt.xlim(0, 400)
    plt.grid()
    
    plt.show()