import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt

AUDIO_FILE = "audio/cero_1.wav"
EXPECTED_FS = 11025
FRAME_SIZE = 256
OVERLAP = 0.50
ALPHA = 0.97

fs, audio = wavfile.read(AUDIO_FILE)
if fs != EXPECTED_FS:
    raise ValueError(f"Se esperaban {EXPECTED_FS} Hz, se encontró {fs} Hz.")

# Quitar offset si es uint8, pasar a float y mono
if audio.dtype == np.uint8:
    audio = audio.astype(np.float64) - 128.0
else:
    audio = audio.astype(np.float64)
if audio.ndim == 2:
    audio = audio.mean(axis=1)

peak = np.max(np.abs(audio))
if peak != 0:
    audio /= peak

# Preénfasis: y[n] = x[n] - alpha*x[n-1]
pre = lfilter([1, -ALPHA], [1], audio)

# Framing (con zero-padding al final)
hop = int(FRAME_SIZE * (1 - OVERLAP))
num_frames = int(np.ceil((len(pre) - FRAME_SIZE) / hop)) + 1
pad = (num_frames - 1) * hop + FRAME_SIZE - len(pre)
pre_p = np.pad(pre, (0, pad))

idx = np.arange(FRAME_SIZE)[None, :] + hop * np.arange(num_frames)[:, None]
frames = pre_p[idx]                       # (num_frames, FRAME_SIZE)

# Ventana Hamming
w = np.hamming(FRAME_SIZE)
wframes = frames * w

# Características por frame
energy = np.sum(wframes**2, axis=1)
power = energy / FRAME_SIZE
zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1) / FRAME_SIZE

# Espectrograma en escala de grises
spec = np.abs(np.fft.rfft(wframes, n=FRAME_SIZE, axis=1))**2   # (frames, 129)
spec_db = 10 * np.log10(spec + 1e-10)
spec_db = np.clip(spec_db, spec_db.max() - 80, None)           # rango dinámico 80 dB

img = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
img = (img * 255).astype(np.uint8).T                           # (freq, tiempo)

t_max = len(audio) / fs
plt.imshow(img, cmap="gray", origin="lower", aspect="auto",
           extent=[0, t_max, 0, fs / 2])
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.title("Espectrograma")
plt.colorbar(label="Intensidad (0-255)")
plt.show()