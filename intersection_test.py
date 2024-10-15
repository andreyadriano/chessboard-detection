import cv2
import numpy as np
import argparse

def wait_key():
    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

def extrapolate_line(x1, y1, x2, y2, img_shape):
    height, width = img_shape[:2]

    # Calcular coeficiente angular e linear
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope = float('inf')
        intercept = None

    # Encontrar interseções com as bordas
    if slope != float('inf'):
        y0 = int(slope * 0 + intercept)  # Interseção esquerda
        y_max = int(slope * width + intercept)  # Interseção direita
        return 0, y0, width, y_max
    else:
        return x1, 0, x2, height  # Linha vertical

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    A1 = float(y2 - y1)
    B1 = float(x1 - x2)
    C1 = float(A1 * x1 + B1 * y1)

    A2 = float(y4 - y3)
    B2 = float(x3 - x4)
    C2 = float(A2 * x3 + B2 * y3)

    determinant = A1 * B2 - A2 * B1

    if abs(determinant) < 1e-10:
        return None  # Linhas quase paralelas

    try:
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return int(x), int(y)
    except OverflowError:
        return None

# Configurar o parser de argumentos
parser = argparse.ArgumentParser(description="Detecção de linhas em uma imagem de tabuleiro de xadrez.")
parser.add_argument("imagem_caminho", type=str, help="Caminho para a imagem do tabuleiro de xadrez.")
args = parser.parse_args()

# Carregar a imagem
img = cv2.imread(args.imagem_caminho)
if img is None:
    print(f"Erro: Não foi possível carregar a imagem '{args.imagem_caminho}'")
    exit()

cv2.namedWindow("output", cv2.WINDOW_NORMAL)

# Converter para escala de cinza e aplicar filtro de mediana
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
median_blurred = cv2.medianBlur(gray, 7)

# Detectar bordas
edges = cv2.Canny(median_blurred, 30, 120, apertureSize=3, L2gradient=True)

# Aplicar dilatação
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)

# Detectar linhas
lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=20)

# Encontrar quadriláteros
if lines is not None:
    quadrilateros = []
    # Extrapolar linhas e armazenar
    extrapolated_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1_ext, y1_ext, x2_ext, y2_ext = extrapolate_line(x1, y1, x2, y2, img.shape)
        extrapolated_lines.append((x1_ext, y1_ext, x2_ext, y2_ext))

    # Checar interseções entre linhas extrapoladas
    for i in range(len(extrapolated_lines)):
        for j in range(i + 1, len(extrapolated_lines)):
            intersection = line_intersection(extrapolated_lines[i], extrapolated_lines[j])
            if intersection is not None:
                quadrilateros.append(intersection)

    # Desenhar quadriláteros na imagem original
    for point in quadrilateros:
        cv2.circle(img, point, 5, (0, 0, 255), -1)  # Desenhar ponto vermelho

# Mostrar a imagem original com quadriláteros
cv2.imshow("output", img)
wait_key()
cv2.destroyAllWindows()
