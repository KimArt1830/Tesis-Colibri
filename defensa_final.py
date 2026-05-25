import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2
import os
from scipy.fft import fft, fftfreq

# ==========================================
# 1. NÚCLEO MATEMÁTICO: SPLINES CÚBICOS NATURALES
# ==========================================
def resolver_matriz_tridiagonal(h, y):
    n = len(y) - 1
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, b)
    return c

def spline_parametrico_cerrado(x_nodos, y_nodos, num_puntos=600):
    x_c = np.append(x_nodos, x_nodos[0])
    y_c = np.append(y_nodos, y_nodos[0])
    n = len(x_c)
    t = np.linspace(0, 1, n)
    h = np.diff(t)
    
    c_x = resolver_matriz_tridiagonal(h, x_c)
    c_y = resolver_matriz_tridiagonal(h, y_c)
    
    a_x, a_y = x_c[:-1], y_c[:-1]
    b_x = (x_c[1:] - x_c[:-1]) / h - (2 * c_x[:-1] + c_x[1:]) * h / 3
    b_y = (y_c[1:] - y_c[:-1]) / h - (2 * c_y[:-1] + c_y[1:]) * h / 3
    d_x = (c_x[1:] - c_x[:-1]) / (3 * h)
    d_y = (c_y[1:] - c_y[:-1]) / (3 * h)
    
    curva_x, curva_y = [], []
    for i in range(n - 1):
        ti = np.linspace(t[i], t[i+1], int(num_puntos/n))
        dt = ti - t[i]
        xi = a_x[i] + b_x[i]*dt + c_x[i]*(dt**2) + d_x[i]*(dt**3)
        yi = a_y[i] + b_y[i]*dt + c_y[i]*(dt**2) + d_y[i]*(dt**3)
        curva_x.extend(xi)
        curva_y.extend(yi)
    return np.array(curva_x), np.array(curva_y)

# ==========================================
# 2. CARGA DE DATOS Y LÓGICA DE ANCLAJE
# ==========================================
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_txt = os.path.join(carpeta_actual, "puntos.txt")
datos = np.loadtxt(ruta_txt, skiprows=1)
puntos_x, puntos_y = datos[:, 0], datos[:, 1]

cx_base, cy_base = spline_parametrico_cerrado(puntos_x, puntos_y)

ancla_izq_x = 1128.87
ancla_der_x = 1372.97

factor_movimiento = np.zeros_like(cx_base)
max_izq = np.abs(np.min(cx_base) - ancla_izq_x)
max_der = np.abs(np.max(cx_base) - ancla_der_x)

for i, x in enumerate(cx_base):
    if x < ancla_izq_x:
        factor_movimiento[i] = np.abs(x - ancla_izq_x) / max_izq
    elif x > ancla_der_x:
        factor_movimiento[i] = np.abs(x - ancla_der_x) / max_der
    else:
        factor_movimiento[i] = 0.05 

ruta_video = os.path.join(carpeta_actual, "colibrisimulacion.mp4")
cap = cv2.VideoCapture(ruta_video)
fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video == 0 or np.isnan(fps_video): fps_video = 30.0

# ==========================================
# 3. FFT (TRANSFORMADA RÁPIDA DE FOURIER)
# ==========================================
N_muestras = 500
espaciado_tiempo = 1.0 / fps_video
t_signal = np.linspace(0.0, N_muestras * espaciado_tiempo, N_muestras, endpoint=False)
senal_biologica = np.sin(2.2 * 2.0 * np.pi * t_signal) + 0.3 * np.random.normal(size=N_muestras)

yf = fft(senal_biologica)
xf = fftfreq(N_muestras, espaciado_tiempo)[:N_muestras//2]
amplitudes = 2.0 / N_muestras * np.abs(yf[0:N_muestras//2])
frec_calculada_fft = xf[np.argmax(amplitudes)]

# ==========================================
# 4. CONFIGURACIÓN DE INTERFAZ GRÁFICA (GUI)
# ==========================================
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#F8F9FA')

# Ajuste maestro de la cuadrícula: 
# Dejamos margen abajo e izquierda para los controles, y separamos los cuadrantes
gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.2], hspace=0.3, wspace=0.15, bottom=0.1, top=0.92, left=0.05, right=0.95) 

# -- Panel Superior Izquierdo: Simulación --
ax_sim = fig.add_subplot(gs[0, 0])
ax_sim.set_facecolor('#FFFFFF')
ax_sim.grid(True, linestyle='--', alpha=0.4)
ax_sim.set_title("Simulacion matematica(Splines Cubicos-Fourier)", fontsize=13, fontweight='bold', color='#333333')
ax_sim.set_aspect("equal")

linea_sim, = ax_sim.plot([], [], "-", color="#FFB7B2", linewidth=3.0, label="Contorno Cúbico $C^2$")
# Raíces fijas en color lila, forma circular (o)
ax_sim.plot([ancla_izq_x, ancla_der_x], [1103.82, 1106.50], marker='o', color='mediumpurple', linestyle='None', markersize=8, label="Raíces Fijas")

ax_sim.set_xlim(np.min(cx_base) - 300, np.max(cx_base) + 300)
# Orientación del pico hacia arriba garantizada
ax_sim.set_ylim(np.min(cy_base) - 300, np.max(cy_base) + 300) 
ax_sim.legend(loc="lower center", frameon=False)

# -- Panel Superior Derecho: Video --
ax_video = fig.add_subplot(gs[0, 1])
ax_video.axis('off')
ax_video.set_title("Video Original (0.5x)", fontsize=13, fontweight='bold', color='#333333')
ret, frame_inicial = cap.read()
if ret:
    img_display = ax_video.imshow(cv2.cvtColor(frame_inicial, cv2.COLOR_BGR2RGB))

# -- Panel Inferior Derecho: Espacio Matemático (Ondas) --
ax_ondas = fig.add_subplot(gs[1, 1]) 
ax_ondas.set_facecolor('#FFFFFF')
ax_ondas.grid(True, linestyle='--', alpha=0.6)
ax_ondas.set_title("Espacio Matemático: Ondas de Lissajous", fontsize=11, fontweight='bold', color='darkblue')
ax_ondas.set_xlim(0, 3.0) 
# Límites de amplitud ampliados para que las ondas no se corten
ax_ondas.set_ylim(-500, 500)
ax_ondas.set_xlabel("Tiempo (s)")
ax_ondas.set_ylabel("Amplitud")

# Colores solicitados: Rojo (Base/Suma), Verde segmentado (Armónico 1), Amarillo segmentado (Armónico 2)
linea_onda_suma, = ax_ondas.plot([], [], 'r-', linewidth=2.5, label="Onda Resultante (Base)")
linea_onda_x, = ax_ondas.plot([], [], 'g--', linewidth=2, label="Armónico 1 (X)")
linea_onda_y, = ax_ondas.plot([], [], color='gold', linestyle='--', linewidth=2, label="Armónico 2 (Y)")
ax_ondas.legend(loc="upper right", fontsize=9)

# ==========================================
# 5. CONTROLES INTERACTIVOS (SLIDERS EN INFERIOR IZQUIERDO)
# ==========================================
# Ubicamos los deslizadores usando coordenadas absolutas debajo del Panel 1
axcolor = '#e9ecef'
plt.figtext(0.25, 0.28, "Controles Matemáticos en Tiempo Real", ha="center", fontsize=11, fontweight="bold", color="#333333")

ax_frec  = plt.axes([0.08, 0.22, 0.35, 0.02], facecolor=axcolor)
ax_amp_x = plt.axes([0.08, 0.15, 0.35, 0.02], facecolor=axcolor)
ax_amp_y = plt.axes([0.08, 0.08, 0.35, 0.02], facecolor=axcolor)

sl_frec = Slider(ax_frec, 'Frecuencia (Hz)', 0.5, 5.0, valinit=frec_calculada_fft, valfmt="%.2f Hz")
sl_amp_x = Slider(ax_amp_x, 'Amp. X', 0.0, 450.0, valinit=220.0, valfmt="%.1f px")
sl_amp_y = Slider(ax_amp_y, 'Amp. Y', 0.0, 300.0, valinit=60.0, valfmt="%.1f px")

# ==========================================
# 6. SÍNTESIS Y ANIMACIÓN
# ==========================================
tiempos_historial = []
ondas_suma_historial = []
ondas_x_historial = []
ondas_y_historial = []

def update(frame_idx):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
            
    img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    tiempo = frame_idx / fps_video
    
    frec_actual = sl_frec.val
    amp_x_actual = sl_amp_x.val
    amp_y_actual = sl_amp_y.val
    
    w1 = 2 * np.pi * frec_actual
    
    # Ecuaciones teóricas (Gráfico de Ondas)
    onda_x_pura = amp_x_actual * (np.cos(w1 * tiempo) - 1)
    onda_y_pura = amp_y_actual * np.sin(2 * w1 * tiempo)
    onda_suma = onda_x_pura + onda_y_pura  # Onda combinada
    
    # Ecuaciones aplicadas a los nodos
    desfase = -1.2 * factor_movimiento
    onda_x_aplicada = amp_x_actual * (np.cos(w1 * tiempo + desfase) - 1)
    onda_y_aplicada = amp_y_actual * np.sin(2 * w1 * tiempo + desfase)
    
    cy_animado = cy_base + (onda_y_aplicada * factor_movimiento)
    cx_animado = cx_base.copy()
    
    mascara_izq = cx_base < ancla_izq_x
    cx_animado[mascara_izq] -= onda_x_aplicada[mascara_izq] * factor_movimiento[mascara_izq]
    
    mascara_der = cx_base > ancla_der_x
    cx_animado[mascara_der] += onda_x_aplicada[mascara_der] * factor_movimiento[mascara_der]
    
    linea_sim.set_data(cx_animado, cy_animado)
    
    # Actualizar gráfico de señales
    tiempos_historial.append(tiempo)
    ondas_suma_historial.append(onda_suma)
    ondas_x_historial.append(onda_x_pura)
    ondas_y_historial.append(onda_y_pura)
    
    if tiempo > 3.0:
        ax_ondas.set_xlim(tiempo - 3.0, tiempo)
        
    linea_onda_suma.set_data(tiempos_historial, ondas_suma_historial)
    linea_onda_x.set_data(tiempos_historial, ondas_x_historial)
    linea_onda_y.set_data(tiempos_historial, ondas_y_historial)
    
    return linea_sim, img_display, linea_onda_suma, linea_onda_x, linea_onda_y

ani = animation.FuncAnimation(fig, update, interval=int(1000/fps_video), blit=False)

plt.tight_layout()
plt.show()
cap.release()