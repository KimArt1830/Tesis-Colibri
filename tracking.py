# Requiere: pip install opencv-python
import cv2

puntos = []

def extraer_coordenadas(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        print(f"Punto guardado: x={x}, y={y}")
        # Dibuja un pequeño círculo rojo donde hiciste clic
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow('Tracking del Colibri', img)

img = cv2.imread('colibri.png')
cv2.imshow('Tracking del Colibri', img)
cv2.setMouseCallback('Tracking del Colibri', extraer_coordenadas)

print("Haz clic en los bordes de las alas. Presiona cualquier tecla para salir.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Tus coordenadas finales:", puntos)