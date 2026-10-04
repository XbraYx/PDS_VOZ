"""
Analizador de audio - Proyecto PDS
==================================
Pipeline: WAV -> mono/normalizado -> preénfasis -> framing -> Hamming
Características por segmento: energía, potencia, cruces por cero, pitch (F0)
Visualización: forma de onda, espectrograma (grises), energía, potencia,
cruces por cero y trayectoria del pitch, seleccionables con casillas.

Requisitos: numpy, scipy, matplotlib (tkinter viene con Python)
    pip install numpy scipy matplotlib
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

EXPECTED_FS = 11025
EPS = 1e-12


# ======================================================================
# DSP
# ======================================================================
def load_audio(path):
    """Lee el WAV, lo pasa a float64, mono y normalizado a [-1, 1]."""
    fs, x = wavfile.read(path)
    if x.dtype == np.uint8:                 # WAV de 8 bits: offset de 128
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
    """y[n] = x[n] - alpha * x[n-1]  (y[0] = x[0])"""
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
    idx = np.arange(frame_size)[None, :] + hop * np.arange(num)[:, None]
    return xp[idx], num


def zero_crossings(frames):
    """Número de cambios de signo por frame."""
    return np.sum(frames[:, 1:] * frames[:, :-1] < 0, axis=1)


def spectrogram_gray(wframes, n_fft, dyn_range_db=80.0):
    """Espectrograma de potencia en dB, escalado a 0-255 (uint8).
    Devuelve matriz (frecuencia, tiempo)."""
    spec = np.abs(np.fft.rfft(wframes, n=n_fft, axis=1)) ** 2
    spec_db = 10 * np.log10(spec + 1e-10)
    spec_db = np.clip(spec_db, spec_db.max() - dyn_range_db, None)
    rng = spec_db.max() - spec_db.min()
    img = (spec_db - spec_db.min()) / (rng if rng > 0 else 1.0)
    return (img * 255).astype(np.uint8).T


def estimate_pitch(x, fs, num, hop, frame_size, pframe, fmin, fmax, voicing_thr):
    """
    F0 por autocorrelación (normalizada y compensada por la ventana).
    Se usa una ventana más larga (pframe) centrada en cada frame, porque
    256 muestras a 11025 Hz (23 ms) son pocas para voces graves.
    Devuelve (f0, fuerza) con NaN en frames sin pitch detectado.
    """
    half = pframe // 2
    xp = np.pad(x, (half, pframe + frame_size))
    win = np.hamming(pframe)
    nfft = 2 ** int(np.ceil(np.log2(2 * pframe)))

    # autocorrelación de la ventana, para compensar el sesgo hacia lags cortos
    rw = np.fft.irfft(np.abs(np.fft.rfft(win, nfft)) ** 2)[:pframe]
    rw = rw / rw[0]

    lag_min = max(int(fs / fmax), 1)
    lag_max = min(int(fs / fmin), pframe - 2)
    if lag_max <= lag_min:
        raise ValueError("Rango de F0 inválido para la ventana de pitch.")

    f0 = np.full(num, np.nan)
    strength = np.zeros(num)

    for m in range(num):
        start = m * hop + frame_size // 2          # centro del frame (coord. de xp)
        seg = xp[start:start + pframe]
        seg = (seg - seg.mean()) * win
        if np.dot(seg, seg) < 1e-10:
            continue
        r = np.fft.irfft(np.abs(np.fft.rfft(seg, nfft)) ** 2)[:pframe]
        r = r / r[0] / (rw + 1e-9)

        # primer máximo local cercano al máximo global (evita errores de octava)
        win_r = r[lag_min - 1:lag_max + 2]
        mid = win_r[1:-1]
        is_peak = (mid >= win_r[:-2]) & (mid >= win_r[2:])
        cand = np.where(is_peak & (mid >= 0.9 * mid.max()))[0]
        k = lag_min + int(cand[0] if len(cand) else np.argmax(mid))
        strength[m] = r[k]
        if r[k] < voicing_thr:
            continue

        lag = float(k)
        if 1 <= k < pframe - 1:                    # interpolación parabólica
            a, b, c = r[k - 1], r[k], r[k + 1]
            den = a - 2 * b + c
            if den != 0:
                delta = 0.5 * (a - c) / den
                if abs(delta) < 1:
                    lag += delta
        f0[m] = fs / lag
    return f0, strength


def analyze(x, fs, frame_size, overlap, alpha, pframe, fmin, fmax,
            voicing_thr, energy_gate_pct):
    hop = max(int(frame_size * (1 - overlap)), 1)

    pre = preemphasis(x, alpha)
    frames, num = frame_signal(pre, frame_size, hop)
    wframes = frames * np.hamming(frame_size)

    energy = np.sum(wframes ** 2, axis=1)
    power = energy / frame_size
    zc = zero_crossings(frames)            # Hamming > 0: no cambia los cruces
    zcr = zc / frame_size

    f0, strength = estimate_pitch(x, fs, num, hop, frame_size, pframe,
                                  fmin, fmax, voicing_thr)
    # descartar pitch en frames de energía muy baja (silencio / ruido)
    gate = energy_gate_pct / 100.0 * energy.max()
    f0[energy < gate] = np.nan

    t = (np.arange(num) * hop + frame_size / 2) / fs
    return dict(
        fs=fs, x=x, hop=hop, frame_size=frame_size, num=num, t=t,
        energy=energy, power=power, zc=zc, zcr=zcr, f0=f0, strength=strength,
        spec=spectrogram_gray(wframes, frame_size),
        fmin=fmin, fmax=fmax,
    )


def pitch_stats(f0):
    v = f0[~np.isnan(f0)]
    if len(v) < 2:
        return None
    mean, std = v.mean(), v.std()
    return dict(n=len(v), mean=mean, std=std, cv=100 * std / mean,
                fmin=v.min(), fmax=v.max(),
                semitones=12 * np.log2(v.max() / v.min()))


# ======================================================================
# GUI
# ======================================================================
class App(tk.Tk):
    PLOTS = [
        ("wave",   "Forma de onda"),
        ("spec",   "Espectrograma"),
        ("energy", "Energía"),
        ("power",  "Potencia"),
        ("zcr",    "Cruces por cero"),
        ("pitch",  "Pitch (F0)"),
    ]

    def __init__(self):
        super().__init__()
        self.title("Analizador de audio - PDS")
        self.geometry("1320x900")
        self.fs = None
        self.audio = None
        self.res = None
        self._build_ui()

    # ---------------- interfaz ----------------
    def _build_ui(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(left, text="Abrir WAV…", command=self.open_file).pack(fill=tk.X)
        self.lbl_file = ttk.Label(left, text="(sin archivo)", wraplength=220)
        self.lbl_file.pack(anchor="w", pady=(4, 8))

        # parámetros
        box = ttk.LabelFrame(left, text="Parámetros", padding=6)
        box.pack(fill=tk.X)
        self.params = {}
        defs = [
            ("frame",  "Tamaño de frame",          256),
            ("ovl",    "Traslape (%)",             50),
            ("alpha",  "Preénfasis α",             0.97),
            ("pframe", "Ventana de pitch",         512),
            ("fmin",   "F0 mínima (Hz)",           70),
            ("fmax",   "F0 máxima (Hz)",           400),
            ("vthr",   "Umbral de voz (0-1)",      0.30),
            ("gate",   "Umbral energía (% máx)",   2.0),
            ("sfmax",  "Fmáx espectrograma (Hz)",  5512),
        ]
        for i, (key, text, default) in enumerate(defs):
            ttk.Label(box, text=text).grid(row=i, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=str(default))
            ttk.Entry(box, textvariable=var, width=8).grid(row=i, column=1, padx=4)
            self.params[key] = var

        ttk.Button(left, text="Procesar", command=self.process).pack(fill=tk.X, pady=8)

        # selección de gráficas
        sel = ttk.LabelFrame(left, text="Mostrar", padding=6)
        sel.pack(fill=tk.X)
        self.show = {}
        defaults_on = {"wave": False, "spec": True, "energy": True,
                       "power": False, "zcr": True, "pitch": True}
        for key, text in self.PLOTS:
            var = tk.BooleanVar(value=defaults_on[key])
            row = ttk.Frame(sel)
            row.pack(fill=tk.X)
            ttk.Checkbutton(row, text=text, variable=var,
                            command=self.plot).pack(side=tk.LEFT)
            ttk.Button(row, text="Ampliar", width=8,
                       command=lambda k=key: self.open_popup(k)).pack(side=tk.RIGHT)
            self.show[key] = var
        self.db = tk.BooleanVar(value=False)
        ttk.Checkbutton(sel, text="Energía/potencia en dB", variable=self.db,
                        command=self.plot).pack(anchor="w", pady=(6, 0))
        ttk.Button(sel, text="Abrir seleccionadas por separado",
                   command=self.open_selected).pack(fill=tk.X, pady=(8, 0))

        ttk.Button(left, text="Exportar tabla a CSV…",
                   command=self.export_csv).pack(fill=tk.X, pady=(10, 4))

        self.lbl_stats = ttk.Label(left, text="", justify="left", wraplength=230)
        self.lbl_stats.pack(anchor="w", pady=6)

        # derecha: pestañas
        nb = ttk.Notebook(self)
        nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tab_plot = ttk.Frame(nb)
        nb.add(tab_plot, text="Gráficas")
        self.fig = Figure(figsize=(9, 7), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_plot)
        NavigationToolbar2Tk(self.canvas, tab_plot).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tab_tab = ttk.Frame(nb)
        nb.add(tab_tab, text="Valores por segmento")
        cols = ("frame", "t", "energia", "potencia", "cruces", "zcr", "f0")
        heads = ("Frame", "Tiempo (s)", "Energía", "Potencia",
                 "Cruces", "ZCR (cruces/muestra)", "F0 (Hz)")
        self.tree = ttk.Treeview(tab_tab, columns=cols, show="headings")
        for c, h in zip(cols, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=110, anchor="center")
        sb = ttk.Scrollbar(tab_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------------- acciones ----------------
    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if not path:
            return
        try:
            fs, x = load_audio(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")
            return
        if fs != EXPECTED_FS:
            ok = messagebox.askyesno(
                "Frecuencia de muestreo",
                f"El audio está a {fs} Hz (se esperaban {EXPECTED_FS} Hz).\n"
                "¿Continuar de todos modos?")
            if not ok:
                return
        self.fs, self.audio = fs, x
        self.lbl_file.config(
            text=f"{path.split('/')[-1]}\n{fs} Hz · {len(x)} muestras · {len(x)/fs:.3f} s")
        self.process()

    def _read_params(self):
        p = self.params
        return dict(
            frame_size=int(float(p["frame"].get())),
            overlap=float(p["ovl"].get()) / 100.0,
            alpha=float(p["alpha"].get()),
            pframe=int(float(p["pframe"].get())),
            fmin=float(p["fmin"].get()),
            fmax=float(p["fmax"].get()),
            voicing_thr=float(p["vthr"].get()),
            energy_gate_pct=float(p["gate"].get()),
        )

    def process(self):
        if self.audio is None:
            messagebox.showinfo("Aviso", "Primero abre un archivo WAV.")
            return
        try:
            prm = self._read_params()
            if not 0 <= prm["overlap"] < 1:
                raise ValueError("El traslape debe estar entre 0 y 99 %.")
            self.res = analyze(self.audio, self.fs, **prm)
        except Exception as e:
            messagebox.showerror("Error en parámetros", str(e))
            return
        self.fill_table()
        self.update_stats()
        self.plot()

    def fill_table(self):
        r = self.res
        self.tree.delete(*self.tree.get_children())
        for m in range(r["num"]):
            f0 = "—" if np.isnan(r["f0"][m]) else f"{r['f0'][m]:.1f}"
            self.tree.insert("", "end", values=(
                m, f"{r['t'][m]:.4f}", f"{r['energy'][m]:.6f}",
                f"{r['power'][m]:.8f}", int(r["zc"][m]),
                f"{r['zcr'][m]:.4f}", f0))

    def update_stats(self):
        r = self.res
        s = pitch_stats(r["f0"])
        head = f"Frames: {r['num']}  (hop = {r['hop']} muestras)\n"
        if s is None:
            self.lbl_stats.config(text=head + "\nPitch: no se detectó (¿audio sin voz?)")
            return
        if s["cv"] < 5:
            verdict = "prácticamente constante"
        elif s["cv"] < 15:
            verdict = "variación moderada"
        else:
            verdict = "varía considerablemente"
        self.lbl_stats.config(text=(
            head + "\nPitch (frames con voz):\n"
            f"  Detectado en {s['n']}/{r['num']} frames\n"
            f"  Media: {s['mean']:.1f} Hz\n"
            f"  Desv. estándar: {s['std']:.1f} Hz\n"
            f"  Rango: {s['fmin']:.1f} – {s['fmax']:.1f} Hz\n"
            f"  Variación: {s['semitones']:.1f} semitonos\n"
            f"  CV: {s['cv']:.1f} % → {verdict}"))

    def draw(self, ax, key):
        """Dibuja la gráfica 'key' en 'ax'. Devuelve la imagen si es el espectrograma."""
        r = self.res
        t, fs = r["t"], r["fs"]
        t_end = len(r["x"]) / fs
        im = None

        if key == "wave":
            ax.plot(np.arange(len(r["x"])) / fs, r["x"], lw=0.6)
            ax.set_ylabel("Amplitud")
            ax.set_title("Forma de onda", fontsize=9)

        elif key == "spec":
            try:
                sfmax = float(self.params["sfmax"].get())
            except ValueError:
                sfmax = fs / 2
            sfmax = min(max(sfmax, 100.0), fs / 2)
            im = ax.imshow(r["spec"], cmap="gray", origin="lower", aspect="auto",
                           extent=[0, t_end, 0, fs / 2], vmin=0, vmax=255)
            ax.set_ylim(0, sfmax)
            ax.set_ylabel("Frecuencia (Hz)")
            ax.set_title("Espectrograma (escala de grises)", fontsize=9)

        elif key in ("energy", "power"):
            y = r[key]
            name = "Energía" if key == "energy" else "Potencia"
            if self.db.get():
                y = 10 * np.log10(y + EPS)
                ax.set_ylabel(f"{name} (dB)")
            else:
                ax.set_ylabel(name)
            ax.plot(t, y, "o-", ms=2.5, lw=1)
            ax.set_title(f"{name} por segmento", fontsize=9)

        elif key == "zcr":
            ax.plot(t, r["zc"], "o-", ms=2.5, lw=1, color="tab:green")
            ax.set_ylabel("Cruces")
            ax.set_title("Cruces por cero por segmento", fontsize=9)

        elif key == "pitch":
            f0 = r["f0"]
            ax.plot(t, f0, "o-", ms=3, lw=1, color="tab:red")
            s = pitch_stats(f0)
            if s:
                ax.axhline(s["mean"], ls="--", lw=0.8, color="gray")
                ax.set_title(
                    f"Pitch (F0): media {s['mean']:.1f} Hz, "
                    f"σ {s['std']:.1f} Hz, CV {s['cv']:.1f} %", fontsize=9)
            else:
                ax.set_title("Pitch (F0): no detectado", fontsize=9)
            ax.set_ylim(r["fmin"], r["fmax"])
            ax.set_ylabel("F0 (Hz)")

        if key != "spec":
            ax.grid(alpha=0.3)
        ax.set_xlim(0, t_end)
        return im

    def plot(self):
        self.fig.clear()
        if self.res is None:
            self.canvas.draw_idle()
            return
        sel = [k for k, _ in self.PLOTS if self.show[k].get()]
        if not sel:
            self.canvas.draw_idle()
            return

        # el espectrograma recibe el doble de altura que las demás
        ratios = [2 if k == "spec" else 1 for k in sel]
        axes = np.atleast_1d(self.fig.subplots(
            len(sel), 1, sharex=True, gridspec_kw={"height_ratios": ratios}))
        for ax, key in zip(axes, sel):
            self.draw(ax, key)
        axes[-1].set_xlabel("Tiempo (s)")
        self.canvas.draw_idle()

    def open_popup(self, key):
        """Abre una gráfica individual en su propia ventana (con zoom/guardar)."""
        if self.res is None:
            messagebox.showinfo("Aviso", "Primero abre y procesa un archivo WAV.")
            return
        name = dict(self.PLOTS)[key]
        win = tk.Toplevel(self)
        win.title(name)
        win.geometry("1000x560")
        fig = Figure(figsize=(10, 5.5), layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        NavigationToolbar2Tk(cv, win).update()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ax = fig.subplots()
        im = self.draw(ax, key)
        if im is not None:
            fig.colorbar(im, ax=ax, label="Intensidad (0-255)")
        ax.set_xlabel("Tiempo (s)")
        cv.draw()

    def open_selected(self):
        for key, _ in self.PLOTS:
            if self.show[key].get():
                self.open_popup(key)

    def export_csv(self):
        if self.res is None:
            messagebox.showinfo("Aviso", "No hay resultados para exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        r = self.res
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "tiempo_s", "energia", "potencia",
                        "cruces", "zcr", "f0_hz"])
            for m in range(r["num"]):
                f0 = "" if np.isnan(r["f0"][m]) else f"{r['f0'][m]:.3f}"
                w.writerow([m, f"{r['t'][m]:.6f}", f"{r['energy'][m]:.8f}",
                            f"{r['power'][m]:.10f}", int(r["zc"][m]),
                            f"{r['zcr'][m]:.6f}", f0])
        messagebox.showinfo("Listo", f"Tabla guardada en:\n{path}")


if __name__ == "__main__":
    App().mainloop()
