import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# BANCO DE FILTROS MEL
# Método de muestreo + ventana de Hamming
# ============================================================

# ------------------------------------------------------------
# Parámetros del sistema
# ------------------------------------------------------------

Fs = 11025       # Frecuencia de muestreo [Hz]
N = 128          # Longitud de cada filtro FIR

# ------------------------------------------------------------
# Frecuencias Mel obtenidas previamente
#
# Cada fila:
# [frecuencia central, frecuencia inferior, frecuencia superior]
# ------------------------------------------------------------

filtros = np.array([
    [54.73,    0.00,    113.75],
    [113.75,   54.73,   177.37],
    [177.37,   113.75,  245.98],
    [245.98,   177.37,  319.94],
    [319.94,   245.98,  399.69],
    [399.69,   319.94,  485.68],
    [485.68,   399.69,  578.39],
    [578.39,   485.68,  678.34],
    [678.34,   578.39,  786.12],
    [786.12,   678.34,  902.32],
    [902.32,   786.12,  1027.60],
    [1027.60,  902.32,  1162.68],
    [1162.68,  1027.60, 1308.33],
    [1308.33,  1162.68, 1465.36],
    [1465.36,  1308.33, 1634.67],
    [1634.67,  1465.36, 1817.22],
    [1817.22,  1634.67, 2014.04],
    [2014.04,  1817.22, 2226.25],
    [2226.25,  2014.04, 2455.06],
    [2455.06,  2226.25, 2701.75],
    [2701.75,  2455.06, 2967.74],
    [2967.74,  2701.75, 3254.52],
    [3254.52,  2967.74, 3563.72],
    [3563.72,  3254.52, 3897.11],
    [3897.11,  3563.72, 4256.56],
    [4256.56,  3897.11, 4644.11],
    [4644.11,  4256.56, 5061.97],
    [5061.97,  4644.11, 5512.50]
])

# ------------------------------------------------------------
# Vector temporal
# ------------------------------------------------------------

n = np.arange(N)

# Centro del filtro
M = (N - 1) / 2

# ------------------------------------------------------------
# Ventana de Hamming
# ------------------------------------------------------------

ventana = np.hamming(N)

# ------------------------------------------------------------
# Arreglos para guardar resultados
# ------------------------------------------------------------

coeficientes = []
frecuencias_angular = []

# ============================================================
# DISEÑO DE LOS 28 FILTROS
# ============================================================

for i, (fc, f1, f2) in enumerate(filtros):

    # --------------------------------------------------------
    # Conversión de Hz a frecuencia angular digital
    # --------------------------------------------------------

    w1 = 2 * np.pi * f1 / Fs
    w2 = 2 * np.pi * f2 / Fs

    frecuencias_angular.append([w1, w2])

    # --------------------------------------------------------
    # Variable centrada
    # --------------------------------------------------------

    k = n - M

    # --------------------------------------------------------
    # Respuesta impulsional ideal
    #
    # Filtro pasa-banda:
    #
    # h[n] = [sin(w2*k) - sin(w1*k)] / (pi*k)
    #
    # Cuando k = 0:
    #
    # h[M] = (w2-w1)/pi
    # --------------------------------------------------------

    h_ideal = np.zeros(N)

    for j in range(N):

        if abs(k[j]) < 1e-12:

            h_ideal[j] = (w2 - w1) / np.pi

        else:

            h_ideal[j] = (
                np.sin(w2 * k[j]) -
                np.sin(w1 * k[j])
            ) / (np.pi * k[j])

    # --------------------------------------------------------
    # Aplicar ventana de Hamming
    # --------------------------------------------------------

    h = h_ideal * ventana

    # Guardar coeficientes
    coeficientes.append(h)


# Convertir a matriz
coeficientes = np.array(coeficientes)

frecuencias_angular = np.array(frecuencias_angular)

# ============================================================
# MOSTRAR FRECUENCIAS ANGULARES
# ============================================================

print("\n==============================================")
print(" FRECUENCIAS ANGULARES")
print("==============================================")

print("\nFiltro   w1 [rad/muestra]   w2 [rad/muestra]")

for i in range(28):

    print(
        f"{i+1:2d}       "
        f"{frecuencias_angular[i,0]:8.5f}           "
        f"{frecuencias_angular[i,1]:8.5f}"
    )


# ============================================================
# MOSTRAR COEFICIENTES
# ============================================================

print("\n==============================================")
print(" COEFICIENTES FIR")
print("==============================================")

for i in range(28):

    print(f"\nFiltro {i+1}:")
    print(coeficientes[i])


# ============================================================
# GRÁFICA DE LA VENTANA DE HAMMING
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(n, ventana)

plt.title("Ventana de Hamming")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# RESPUESTAS EN FRECUENCIA DE LOS FILTROS
# ============================================================

# Número de puntos para calcular la respuesta
n_fft = 4096

frecuencia = np.linspace(0, Fs / 2, n_fft // 2 + 1)

plt.figure(figsize=(12, 7))

for i in range(28):

    h = coeficientes[i]

    H = np.fft.rfft(h, n=n_fft)

    magnitud = np.abs(H)

    # Evitar log(0)
    magnitud_db = 20 * np.log10(
        np.maximum(magnitud, 1e-10)
    )

    plt.plot(frecuencia, magnitud_db)


plt.title("Banco de 28 filtros FIR Mel - Ventana de Hamming")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")

plt.xlim(0, Fs / 2)
plt.ylim(-100, 5)

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# RESPUESTAS INDIVIDUALES
# ============================================================

# Mostrar algunos filtros individualmente
filtros_mostrar = [0, 1, 2, 9, 19, 27]

for i in filtros_mostrar:

    H = np.fft.rfft(
        coeficientes[i],
        n=n_fft
    )

    magnitud_db = 20 * np.log10(
        np.maximum(np.abs(H), 1e-10)
    )

    plt.figure(figsize=(10, 4))

    plt.plot(frecuencia, magnitud_db)

    plt.axvline(
        filtros[i, 1],
        linestyle="--",
        label=f"f1 = {filtros[i,1]:.2f} Hz"
    )

    plt.axvline(
        filtros[i, 2],
        linestyle="--",
        label=f"f2 = {filtros[i,2]:.2f} Hz"
    )

    plt.title(
        f"Filtro Mel {i+1} - FIR Hamming N={N}"
    )

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")

    plt.xlim(0, Fs / 2)
    plt.ylim(-100, 5)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# GUARDAR LOS COEFICIENTES
# ============================================================

np.savetxt(
    "coeficientes_banco_mel.txt",
    coeficientes,
    fmt="%.12f"
)

np.savetxt(
    "frecuencias_omega.txt",
    frecuencias_angular,
    fmt="%.8f"
)

print("\n==============================================")
print(" ARCHIVOS GENERADOS")
print("==============================================")

print("coeficientes_banco_mel.txt")
print("frecuencias_omega.txt")