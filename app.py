"""
Analizador de audio - Proyecto PDS
==================================

Pipeline:
WAV -> mono/normalizado -> preénfasis -> framing -> Hamming

Características por segmento:
- Energía
- Potencia
- Cruces por cero
- ZCR
- Pitch (F0) mediante autocorrelación

Visualización:
- Forma de onda
- Espectrograma
- Energía
- Potencia
- Cruces por cero
- Pitch

Aplicación adaptada para Streamlit Community Cloud.
"""

import io
import csv

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter


EXPECTED_FS = 11025
EPS = 1e-12


# ======================================================================
# DSP
# ======================================================================

def load_audio(file):
    """Lee el WAV, lo pasa a float64, mono y normalizado a [-1, 1]."""
    fs, x = wavfile.read(file)

    if x.dtype == np.uint8:
        x = x.astype(np.float64) - 128.0
    else:
        x = x.astype(np.float64)

    if x.ndim == 2:
        x = x.mean(axis=1)

    peak = np.max(np.abs(x))

    if peak > 0:
        x = x / peak

    return fs, x


def preemphasis(x, alpha):
    """y[n] = x[n] - alpha * x[n-1]"""
    return lfilter([1.0, -alpha], [1.0], x)


def frame_signal(x, frame_size, hop):
    """Divide x en frames con traslape; rellena con ceros el último."""
    n = len(x)

    if n < frame_size:
        x = np.pad(x, (0, frame_size - n))
        n = frame_size

    num = int(np.ceil((n - frame_size) / hop)) + 1

    pad = (num - 1) * hop + frame_size - n
    xp = np.pad(x, (0, pad))

    idx = (
        np.arange(frame_size)[None, :]
        + hop * np.arange(num)[:, None]
    )

    return xp[idx], num


def zero_crossings(frames):
    """Número de cambios de signo por frame."""
    return np.sum(
        frames[:, 1:] * frames[:, :-1] < 0,
        axis=1
    )


def spectrogram_gray(wframes, n_fft, dyn_range_db=80.0):
    """
    Espectrograma de potencia en dB, escalado a 0-255.
    Devuelve matriz (frecuencia, tiempo).
    """
    spec = np.abs(
        np.fft.rfft(wframes, n=n_fft, axis=1)
    ) ** 2

    spec_db = 10 * np.log10(spec + 1e-10)

    spec_db = np.clip(
        spec_db,
        spec_db.max() - dyn_range_db,
        None
    )

    rng = spec_db.max() - spec_db.min()

    img = (
        (spec_db - spec_db.min())
        / (rng if rng > 0 else 1.0)
    )

    return (img * 255).astype(np.uint8).T


def estimate_pitch(
    x,
    fs,
    num,
    hop,
    frame_size,
    pframe,
    fmin,
    fmax,
    voicing_thr
):
    """
    F0 por autocorrelación normalizada y compensada por ventana.

    Se utiliza una ventana más larga (pframe) centrada
    en cada frame para mejorar la detección de voces graves.

    Devuelve:
        f0
        strength

    con NaN en frames sin pitch detectado.
    """

    half = pframe // 2

    xp = np.pad(
        x,
        (half, pframe + frame_size)
    )

    win = np.hamming(pframe)

    nfft = 2 ** int(
        np.ceil(np.log2(2 * pframe))
    )

    # Autocorrelación de la ventana para compensar
    # el sesgo hacia lags cortos.
    rw = np.fft.irfft(
        np.abs(np.fft.rfft(win, nfft)) ** 2
    )[:pframe]

    rw = rw / rw[0]

    lag_min = max(
        int(fs / fmax),
        1
    )

    lag_max = min(
        int(fs / fmin),
        pframe - 2
    )

    if lag_max <= lag_min:
        raise ValueError(
            "Rango de F0 inválido para la ventana de pitch."
        )

    f0 = np.full(num, np.nan)
    strength = np.zeros(num)

    for m in range(num):

        start = (
            m * hop
            + frame_size // 2
        )

        seg = xp[
            start:start + pframe
        ]

        seg = (
            seg - seg.mean()
        ) * win

        if np.dot(seg, seg) < 1e-10:
            continue

        r = np.fft.irfft(
            np.abs(np.fft.rfft(seg, nfft)) ** 2
        )[:pframe]

        r = r / r[0] / (rw + 1e-9)

        # Primer máximo local cercano al máximo global.
        # Esto reduce errores de octava.
        win_r = r[
            lag_min - 1:
            lag_max + 2
        ]

        mid = win_r[1:-1]

        is_peak = (
            (mid >= win_r[:-2])
            & (mid >= win_r[2:])
        )

        cand = np.where(
            is_peak
            & (mid >= 0.9 * mid.max())
        )[0]

        k = (
            lag_min
            + int(
                cand[0]
                if len(cand)
                else np.argmax(mid)
            )
        )

        strength[m] = r[k]

        if r[k] < voicing_thr:
            continue

        lag = float(k)

        # Interpolación parabólica.
        if 1 <= k < pframe - 1:

            a = r[k - 1]
            b = r[k]
            c = r[k + 1]

            den = a - 2 * b + c

            if den != 0:

                delta = (
                    0.5 * (a - c) / den
                )

                if abs(delta) < 1:
                    lag += delta

        f0[m] = fs / lag

    return f0, strength


def analyze(
    x,
    fs,
    frame_size,
    overlap,
    alpha,
    pframe,
    fmin,
    fmax,
    voicing_thr,
    energy_gate_pct
):
    """Ejecuta todo el procesamiento DSP."""

    hop = max(
        int(frame_size * (1 - overlap)),
        1
    )

    # --------------------------------------------------------------
    # 1. Preénfasis
    # --------------------------------------------------------------
    pre = preemphasis(x, alpha)

    # --------------------------------------------------------------
    # 2. Framing
    # --------------------------------------------------------------
    frames, num = frame_signal(
        pre,
        frame_size,
        hop
    )

    # --------------------------------------------------------------
    # 3. Ventana Hamming
    # --------------------------------------------------------------
    wframes = (
        frames
        * np.hamming(frame_size)
    )

    # --------------------------------------------------------------
    # 4. Energía y potencia
    # --------------------------------------------------------------
    energy = np.sum(
        wframes ** 2,
        axis=1
    )

    power = (
        energy
        / frame_size
    )

    # --------------------------------------------------------------
    # 5. Cruces por cero
    # --------------------------------------------------------------
    zc = zero_crossings(frames)

    zcr = (
        zc
        / frame_size
    )

    # --------------------------------------------------------------
    # 6. Pitch
    # --------------------------------------------------------------
    f0, strength = estimate_pitch(
        x,
        fs,
        num,
        hop,
        frame_size,
        pframe,
        fmin,
        fmax,
        voicing_thr
    )

    # Descartar pitch en frames de energía muy baja.
    gate = (
        energy_gate_pct
        / 100.0
        * energy.max()
    )

    f0[energy < gate] = np.nan

    # --------------------------------------------------------------
    # 7. Tiempo de cada frame
    # --------------------------------------------------------------
    t = (
        np.arange(num) * hop
        + frame_size / 2
    ) / fs

    return {
        "fs": fs,
        "x": x,
        "pre": pre,
        "frames": frames,
        "wframes": wframes,
        "hop": hop,
        "frame_size": frame_size,
        "num": num,
        "t": t,
        "energy": energy,
        "power": power,
        "zc": zc,
        "zcr": zcr,
        "f0": f0,
        "strength": strength,
        "spec": spectrogram_gray(
            wframes,
            frame_size
        ),
        "fmin": fmin,
        "fmax": fmax,
    }


def pitch_stats(f0):
    """Calcula estadísticas del pitch detectado."""

    v = f0[~np.isnan(f0)]

    if len(v) < 2:
        return None

    mean = v.mean()
    std = v.std()

    return {
        "n": len(v),
        "mean": mean,
        "std": std,
        "cv": 100 * std / mean,
        "fmin": v.min(),
        "fmax": v.max(),
        "semitones": (
            12 * np.log2(
                v.max() / v.min()
            )
        ),
    }


# ======================================================================
# GRÁFICAS
# ======================================================================

def draw_plot(
    r,
    key,
    db=False,
    sfmax=5512
):
    """Genera una figura individual para Streamlit."""

    fig, ax = plt.subplots(
        figsize=(11, 4.5)
    )

    t = r["t"]
    fs = r["fs"]

    t_end = (
        len(r["x"])
        / fs
    )

    if key == "wave":

        ax.plot(
            np.arange(len(r["x"])) / fs,
            r["x"],
            linewidth=0.6
        )

        ax.set_ylabel("Amplitud")
        ax.set_title(
            "Forma de onda"
        )

    elif key == "spec":

        im = ax.imshow(
            r["spec"],
            cmap="gray",
            origin="lower",
            aspect="auto",
            extent=[
                0,
                t_end,
                0,
                fs / 2
            ],
            vmin=0,
            vmax=255
        )

        sfmax = min(
            max(sfmax, 100.0),
            fs / 2
        )

        ax.set_ylim(
            0,
            sfmax
        )

        ax.set_ylabel(
            "Frecuencia (Hz)"
        )

        ax.set_title(
            "Espectrograma "
            "(escala de grises)"
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Intensidad (0-255)"
        )

    elif key in ("energy", "power"):

        y = r[key]

        name = (
            "Energía"
            if key == "energy"
            else "Potencia"
        )

        if db:

            y = (
                10
                * np.log10(
                    y + EPS
                )
            )

            ax.set_ylabel(
                f"{name} (dB)"
            )

        else:
            ax.set_ylabel(name)

        ax.plot(
            t,
            y,
            "o-",
            markersize=2.5,
            linewidth=1
        )

        ax.set_title(
            f"{name} por segmento"
        )

    elif key == "zcr":

        ax.plot(
            t,
            r["zc"],
            "o-",
            markersize=2.5,
            linewidth=1
        )

        ax.set_ylabel(
            "Cruces"
        )

        ax.set_title(
            "Cruces por cero "
            "por segmento"
        )

    elif key == "pitch":

        f0 = r["f0"]

        ax.plot(
            t,
            f0,
            "o-",
            markersize=3,
            linewidth=1
        )

        s = pitch_stats(f0)

        if s:

            ax.axhline(
                s["mean"],
                linestyle="--",
                linewidth=0.8
            )

            ax.set_title(
                f"Pitch (F0): media "
                f"{s['mean']:.1f} Hz, "
                f"σ {s['std']:.1f} Hz, "
                f"CV {s['cv']:.1f} %"
            )

        else:

            ax.set_title(
                "Pitch (F0): "
                "no detectado"
            )

        ax.set_ylim(
            r["fmin"],
            r["fmax"]
        )

        ax.set_ylabel(
            "F0 (Hz)"
        )

    if key != "spec":
        ax.grid(
            alpha=0.3
        )

    ax.set_xlim(
        0,
        t_end
    )

    ax.set_xlabel(
        "Tiempo (s)"
    )

    fig.tight_layout()

    return fig


# ======================================================================
# TABLA Y CSV
# ======================================================================

def results_dataframe(r):
    """Construye la tabla por segmento."""

    import pandas as pd

    f0_display = [
        np.nan if np.isnan(v) else v
        for v in r["f0"]
    ]

    return pd.DataFrame({
        "Frame": np.arange(r["num"]),
        "Tiempo (s)": r["t"],
        "Energía": r["energy"],
        "Potencia": r["power"],
        "Cruces": r["zc"].astype(int),
        "ZCR (cruces/muestra)": r["zcr"],
        "F0 (Hz)": f0_display,
    })


def dataframe_to_csv(df):
    """Convierte la tabla a CSV descargable."""

    return df.to_csv(
        index=False
    ).encode("utf-8")


# ======================================================================
# STREAMLIT
# ======================================================================

st.set_page_config(
    page_title="Analizador de audio - PDS",
    page_icon="🎙️",
    layout="wide"
)


st.title(
    "🎙️ Analizador de Audio - PDS"
)

st.caption(
    "WAV → mono/normalizado → preénfasis → "
    "framing → Hamming → características"
)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:

    st.header(
        "Parámetros"
    )

    uploaded_file = st.file_uploader(
        "Selecciona un archivo WAV",
        type=["wav"]
    )

    st.divider()

    frame_size = st.number_input(
        "Tamaño de frame",
        min_value=16,
        max_value=8192,
        value=256,
        step=16
    )

    overlap_pct = st.slider(
        "Traslape (%)",
        min_value=0,
        max_value=99,
        value=50
    )

    alpha = st.number_input(
        "Preénfasis α",
        min_value=0.0,
        max_value=1.0,
        value=0.97,
        step=0.01
    )

    pframe = st.number_input(
        "Ventana de pitch",
        min_value=32,
        max_value=8192,
        value=512,
        step=16
    )

    fmin = st.number_input(
        "F0 mínima (Hz)",
        min_value=20.0,
        max_value=1000.0,
        value=70.0,
        step=5.0
    )

    fmax = st.number_input(
        "F0 máxima (Hz)",
        min_value=50.0,
        max_value=2000.0,
        value=400.0,
        step=5.0
    )

    voicing_thr = st.slider(
        "Umbral de voz",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    energy_gate_pct = st.number_input(
        "Umbral energía (% máx)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.5
    )

    sfmax = st.number_input(
        "Fmáx espectrograma (Hz)",
        min_value=100.0,
        max_value=EXPECTED_FS / 2,
        value=EXPECTED_FS / 2,
        step=100.0
    )

    st.divider()

    st.subheader(
        "Mostrar gráficas"
    )

    show_wave = st.checkbox(
        "Forma de onda",
        value=False
    )

    show_spec = st.checkbox(
        "Espectrograma",
        value=True
    )

    show_energy = st.checkbox(
        "Energía",
        value=True
    )

    show_power = st.checkbox(
        "Potencia",
        value=False
    )

    show_zcr = st.checkbox(
        "Cruces por cero",
        value=True
    )

    show_pitch = st.checkbox(
        "Pitch (F0)",
        value=True
    )

    db = st.checkbox(
        "Energía/potencia en dB",
        value=False
    )

    process_button = st.button(
        "▶ Procesar audio",
        use_container_width=True,
        type="primary"
    )


# ----------------------------------------------------------------------
# Procesamiento
# ----------------------------------------------------------------------

if uploaded_file is not None:

    try:

        # Leer archivo desde memoria.
        audio_bytes = uploaded_file.getvalue()

        fs, audio = load_audio(
            io.BytesIO(audio_bytes)
        )

        duration = (
            len(audio) / fs
        )

        st.success(
            f"Archivo cargado: "
            f"**{uploaded_file.name}**  \n"
            f"Frecuencia de muestreo: **{fs} Hz**  |  "
            f"Muestras: **{len(audio)}**  |  "
            f"Duración: **{duration:.3f} s**"
        )

        if fs != EXPECTED_FS:

            st.warning(
                f"El audio está a **{fs} Hz**, "
                f"pero el proyecto está configurado "
                f"para **{EXPECTED_FS} Hz**."
            )

        # Guardar audio cargado en session_state.
        st.session_state["audio"] = audio
        st.session_state["fs"] = fs

    except Exception as e:

        st.error(
            f"No se pudo leer el archivo WAV: {e}"
        )


if uploaded_file is not None and (
    process_button
    or "results" not in st.session_state
):

    if fmin >= fmax:

        st.error(
            "F0 mínima debe ser menor que F0 máxima."
        )

    elif pframe < 4:

        st.error(
            "La ventana de pitch es demasiado pequeña."
        )

    else:

        try:

            with st.spinner(
                "Procesando audio..."
            ):

                result = analyze(
                    st.session_state["audio"],
                    st.session_state["fs"],
                    frame_size=int(frame_size),
                    overlap=overlap_pct / 100.0,
                    alpha=float(alpha),
                    pframe=int(pframe),
                    fmin=float(fmin),
                    fmax=float(fmax),
                    voicing_thr=float(voicing_thr),
                    energy_gate_pct=float(
                        energy_gate_pct
                    )
                )

                st.session_state["results"] = result

                # Guardamos los parámetros usados
                # para que las gráficas coincidan
                # con el procesamiento.
                st.session_state["params"] = {
                    "sfmax": sfmax,
                    "db": db,
                    "show_wave": show_wave,
                    "show_spec": show_spec,
                    "show_energy": show_energy,
                    "show_power": show_power,
                    "show_zcr": show_zcr,
                    "show_pitch": show_pitch,
                }

        except Exception as e:

            st.error(
                f"Error durante el procesamiento: {e}"
            )


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------

if "results" in st.session_state:

    r = st.session_state["results"]

    st.divider()

    st.header(
        "Resultados"
    )

    # --------------------------------------------------------------
    # Métricas generales
    # --------------------------------------------------------------

    stats = pitch_stats(
        r["f0"]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Frames",
        r["num"]
    )

    c2.metric(
        "Hop",
        f"{r['hop']} muestras"
    )

    if stats is not None:

        c3.metric(
            "Pitch medio",
            f"{stats['mean']:.1f} Hz"
        )

        c4.metric(
            "Pitch detectado",
            f"{stats['n']}/{r['num']}"
        )

    else:

        c3.metric(
            "Pitch medio",
            "No detectado"
        )

        c4.metric(
            "Pitch detectado",
            f"0/{r['num']}"
        )

    # --------------------------------------------------------------
    # Estadísticas detalladas
    # --------------------------------------------------------------

    with st.expander(
        "📊 Estadísticas del pitch",
        expanded=True
    ):

        if stats is None:

            st.info(
                "No se detectó suficiente pitch "
                "para calcular estadísticas."
            )

        else:

            if stats["cv"] < 5:
                verdict = (
                    "prácticamente constante"
                )
            elif stats["cv"] < 15:
                verdict = (
                    "variación moderada"
                )
            else:
                verdict = (
                    "varía considerablemente"
                )

            a, b, c, d, e = st.columns(5)

            a.metric(
                "Media",
                f"{stats['mean']:.1f} Hz"
            )

            b.metric(
                "Desv. estándar",
                f"{stats['std']:.1f} Hz"
            )

            c.metric(
                "Rango",
                f"{stats['fmin']:.1f}–"
                f"{stats['fmax']:.1f} Hz"
            )

            d.metric(
                "Variación",
                f"{stats['semitones']:.1f} st"
            )

            e.metric(
                "CV",
                f"{stats['cv']:.1f} %"
            )

            st.write(
                f"**Interpretación:** {verdict}."
            )

    # --------------------------------------------------------------
    # Gráficas
    # --------------------------------------------------------------

    st.subheader(
        "Visualización"
    )

    selected = []

    if show_wave:
        selected.append("wave")

    if show_spec:
        selected.append("spec")

    if show_energy:
        selected.append("energy")

    if show_power:
        selected.append("power")

    if show_zcr:
        selected.append("zcr")

    if show_pitch:
        selected.append("pitch")

    # Si los checkboxes se cambiaron después del procesamiento,
    # usamos los valores actuales para la visualización.
    if not selected:

        st.info(
            "Selecciona al menos una gráfica "
            "en el panel izquierdo."
        )

    else:

        for key in selected:

            fig = draw_plot(
                r,
                key,
                db=db,
                sfmax=sfmax
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

    # --------------------------------------------------------------
    # Tabla por segmento
    # --------------------------------------------------------------

    st.subheader(
        "Valores por segmento"
    )

    df = results_dataframe(
        r
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=450
    )

    # --------------------------------------------------------------
    # Descargar CSV
    # --------------------------------------------------------------

    csv_data = dataframe_to_csv(
        df
    )

    st.download_button(
        label="⬇️ Descargar resultados CSV",
        data=csv_data,
        file_name="resultados_audio_pds.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.info(
        "👈 Selecciona un archivo WAV en el panel "
        "izquierdo para comenzar."
    )

    st.markdown(
        """
### Pipeline del proyecto

**WAV → Mono/Normalización → Preénfasis → "
"Framing → Ventana Hamming → Características**

Se calculan:

- Energía por segmento
- Potencia por segmento
- Cruces por cero
- ZCR
- Pitch/F0 mediante autocorrelación
- Espectrograma
        """
    )
