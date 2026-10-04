"""
Verificación independiente de analizador_audio.py
=================================================
Compara los cálculos del programa contra:
  1) Una señal sintética con resultado conocido (prueba analítica)
  2) Una implementación "ingenua" con ciclos for (definición matemática pura)
  3) Parseval (energía en tiempo = energía en frecuencia)
  4) librosa  (RMS, cruces por cero, STFT)  [opcional]
  5) Praat vía parselmouth (pitch)          [opcional]

Uso:
    python verificar_resultados.py                 # solo señal sintética
    python verificar_resultados.py audio/cero_1.wav

Instalación de opcionales:
    pip install librosa praat-parselmouth
"""

import sys
import numpy as np
from scipy.signal import get_window

import analizador_audio as A

FRAME, OVL, ALPHA = 256, 0.50, 0.97
PFRAME, FMIN, FMAX, VTHR, GATE = 512, 70, 400, 0.30, 2.0


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre} {detalle}")


# ----------------------------------------------------------------------
# 1) Prueba analítica: seno de amplitud A y frecuencia f conocidas
# ----------------------------------------------------------------------
def prueba_analitica():
    print("\n1) PRUEBA ANALÍTICA (seno de amplitud 0.5 a 1000 Hz, sin preénfasis)")
    fs, amp, f = 11025, 0.5, 1000.0
    n = np.arange(fs)                       # 1 s
    x = amp * np.sin(2 * np.pi * f * n / fs)   # (sin normalizar: ya vale 0.5)
    # alpha=0 -> el preénfasis es la identidad
    res = A.analyze(x, fs, FRAME, OVL, 0.0, PFRAME, FMIN, FMAX, VTHR, GATE)

    w = np.hamming(FRAME)
    # Potencia esperada con ventana: (A²/2) * mean(w²)  (si f >> 1/N)
    p_esp = (amp ** 2 / 2) * np.mean(w ** 2)
    p_med = np.median(res["power"][2:-3])
    check("Potencia por frame", abs(p_med - p_esp) / p_esp < 0.02,
          f"(medida {p_med:.5f}, esperada {p_esp:.5f})")
    e_esp = p_esp * FRAME
    e_med = np.median(res["energy"][2:-3])
    check("Energía por frame", abs(e_med - e_esp) / e_esp < 0.02,
          f"(medida {e_med:.4f}, esperada {e_esp:.4f})")
    # Cruces esperados: 2 * f * N / fs
    z_esp = 2 * f * FRAME / fs
    z_med = np.median(res["zc"][2:-3])
    check("Cruces por cero", abs(z_med - z_esp) <= 1,
          f"(medidos {z_med:.0f}, esperados ≈ {z_esp:.1f})")
    # Pitch con un seno de 150 Hz
    x2 = amp * np.sin(2 * np.pi * 150 * n / fs)
    r2 = A.analyze(x2, fs, FRAME, OVL, 0.0, PFRAME, FMIN, FMAX, VTHR, GATE)
    f0 = np.nanmedian(r2["f0"])
    check("Pitch de seno de 150 Hz", abs(f0 - 150) < 1.0, f"(medido {f0:.2f} Hz)")


# ----------------------------------------------------------------------
# 2) Implementación ingenua (definiciones directas con for)
# ----------------------------------------------------------------------
def prueba_ingenua(x, fs, res, pre):
    print("\n2) IMPLEMENTACIÓN INGENUA (definición matemática con ciclos for)")
    N, hop = FRAME, res["hop"]
    w = np.hamming(N)
    maxdif_e = maxdif_z = 0.0
    for m in range(res["num"]):
        seg = np.zeros(N)
        chunk = pre[m * hop: m * hop + N]
        seg[:len(chunk)] = chunk
        E = sum((seg[n] * w[n]) ** 2 for n in range(N))          # Σ [x(n)w(n)]²
        Z = sum(1 for n in range(1, N) if seg[n] * seg[n - 1] < 0)
        maxdif_e = max(maxdif_e, abs(E - res["energy"][m]))
        maxdif_z = max(maxdif_z, abs(Z - res["zc"][m]))
    check("Energía Σ[x·w]²", maxdif_e < 1e-9, f"(dif. máx {maxdif_e:.2e})")
    check("Potencia = Energía/N", np.allclose(res["power"], res["energy"] / N))
    check("Cruces por cero", maxdif_z == 0, f"(dif. máx {maxdif_z:.0f})")


# ----------------------------------------------------------------------
# 3) Parseval
# ----------------------------------------------------------------------
def prueba_parseval(pre, res):
    print("\n3) PARSEVAL (energía en tiempo vs. energía en frecuencia)")
    N, hop = FRAME, res["hop"]
    w = np.hamming(N)
    m = res["num"] // 2
    seg = pre[m * hop: m * hop + N] * w
    X = np.fft.fft(seg)
    e_t = np.sum(seg ** 2)
    e_f = np.sum(np.abs(X) ** 2) / N
    check("Σ|x|² = (1/N)Σ|X|²", np.isclose(e_t, e_f), f"({e_t:.6f} vs {e_f:.6f})")


# ----------------------------------------------------------------------
# 4) librosa
# ----------------------------------------------------------------------
def prueba_librosa(x, fs, res, pre):
    print("\n4) LIBROSA")
    try:
        import librosa
    except ImportError:
        print("  (librosa no instalado: pip install librosa)")
        return
    N, hop = FRAME, res["hop"]
    n_ok = 1 + (len(pre) - N) // hop        # librosa con center=False no rellena
    frames = res["_frames"][:n_ok]

    # RMS sin ventana: librosa.rms = sqrt(mean(x²))  ->  rms² = potencia sin ventana
    rms = librosa.feature.rms(y=pre, frame_length=N, hop_length=hop,
                              center=False)[0][:n_ok]
    rms_prop = np.sqrt(np.mean(frames ** 2, axis=1))
    check("RMS (sin ventana)", np.allclose(rms, rms_prop, atol=1e-6),
          f"(dif. máx {np.max(np.abs(rms - rms_prop)):.2e})")

    # Cruces por cero: librosa devuelve la tasa = cruces / (N-1) y cuenta el cero como positivo
    zcr = librosa.feature.zero_crossing_rate(pre, frame_length=N, hop_length=hop,
                                             center=False)[0][:n_ok]
    zc_lib = np.rint(zcr * (N - 1)).astype(int)
    difs = int(np.sum(zc_lib != res["zc"][:n_ok]))
    check("Cruces por cero", difs <= max(1, int(0.02 * n_ok)),
          f"({difs} frames de {n_ok} difieren; librosa cuenta el 0 como positivo)")

    # STFT con Hamming simétrica (np.hamming) para que sea comparable
    S = librosa.stft(pre, n_fft=N, hop_length=hop, win_length=N,
                     window=np.hamming(N), center=False)
    P_lib = np.abs(S) ** 2                                  # (freq, tiempo)
    P_prop = np.abs(np.fft.rfft(frames * np.hamming(N), axis=1)) ** 2
    check("Espectrograma |STFT|²", np.allclose(P_lib[:, :n_ok].T, P_prop, rtol=1e-4, atol=1e-8),
          f"(dif. relativa máx {np.max(np.abs(P_lib[:, :n_ok].T - P_prop)) / P_prop.max():.2e})")


# ----------------------------------------------------------------------
# 5) Praat (parselmouth)
# ----------------------------------------------------------------------
def prueba_praat(x, fs, res):
    print("\n5) PRAAT (pitch)")
    try:
        import parselmouth
    except ImportError:
        print("  (parselmouth no instalado: pip install praat-parselmouth)")
        return
    snd = parselmouth.Sound(x, sampling_frequency=fs)
    pitch = snd.to_pitch(time_step=res["hop"] / fs, pitch_floor=FMIN, pitch_ceiling=FMAX)
    f_praat = np.array([pitch.get_value_at_time(t) for t in res["t"]], dtype=float)
    ambos = ~np.isnan(f_praat) & ~np.isnan(res["f0"])
    if ambos.sum() < 3:
        print("  Muy pocos frames con pitch en ambos; no se puede comparar.")
        return
    d = np.abs(f_praat[ambos] - res["f0"][ambos])
    print(f"  Frames con pitch: programa {np.sum(~np.isnan(res['f0']))}, "
          f"Praat {np.sum(~np.isnan(f_praat))}, en común {ambos.sum()}")
    print(f"  Diferencia media {d.mean():.2f} Hz, mediana {np.median(d):.2f} Hz, "
          f"máx {d.max():.2f} Hz")
    print(f"  Media F0: programa {np.nanmean(res['f0']):.1f} Hz, "
          f"Praat {np.nanmean(f_praat):.1f} Hz")


# ----------------------------------------------------------------------
def main():
    prueba_analitica()

    if len(sys.argv) > 1:
        fs, x = A.load_audio(sys.argv[1])
        print(f"\nAudio real: {sys.argv[1]} ({fs} Hz, {len(x)/fs:.2f} s)")
    else:
        fs = 11025
        t = np.arange(fs) / fs
        f = 120 + 60 * t
        x = sum(np.sin(k * 2 * np.pi * np.cumsum(f) / fs) / k for k in range(1, 6))
        x /= np.abs(x).max()
        print("\nSin archivo: usando señal sintética con pitch 120→180 Hz")

    res = A.analyze(x, fs, FRAME, OVL, ALPHA, PFRAME, FMIN, FMAX, VTHR, GATE)
    pre = A.preemphasis(x, ALPHA)
    frames, _ = A.frame_signal(pre, FRAME, res["hop"])
    res["_frames"] = frames

    prueba_ingenua(x, fs, res, pre)
    prueba_parseval(pre, res)
    prueba_librosa(x, fs, res, pre)
    prueba_praat(x, fs, res)
    print()


if __name__ == "__main__":
    main()
