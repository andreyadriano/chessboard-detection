import cv2
import numpy as np
import argparse

def wait_key():
    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass

def extrapolate_line(x1, y1, x2, y2, img_shape):
    height, width = img_shape[:2]
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

def calculate_angle(line1, line2):
    dx1, dy1 = line1[2] - line1[0], line1[3] - line1[1]
    dx2, dy2 = line2[2] - line2[0], line2[3] - line2[1]

    dot_product = dx1 * dx2 + dy1 * dy2
    mag1 = np.sqrt(dx1**2 + dy1**2)
    mag2 = np.sqrt(dx2**2 + dy2**2)

    if mag1 == 0 or mag2 == 0:
        return None

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * (180.0 / np.pi)

    return angle

def line_intersection(line1, line2):
    angle = calculate_angle(line1, line2)
    if angle is not None and (angle < 85 or angle > 95):
        return None

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
        return None

    try:
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return int(x), int(y)
    except OverflowError:
        return None

# Função para calcular o tamanho da imagem baseada nos 4 pontos
def calculate_image_size(rect):
    # Calcular a largura e a altura com base nas distâncias entre os pontos
    width_top = np.linalg.norm(rect[0] - rect[1])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_left = np.linalg.norm(rect[0] - rect[3])
    height_right = np.linalg.norm(rect[1] - rect[2])

    # Largura e altura médias
    width = int((width_top + width_bottom) / 2)
    height = int((height_left + height_right) / 2)

    return width, height

# Função para aplicar a transformada de perspectiva
def apply_perspective_transform(img, corners):
    # Calcular o tamanho da nova imagem
    width, height = calculate_image_size(corners)

    # Definir os pontos de destino para a transformação
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

    # Obter a matriz de transformação de perspectiva
    M = cv2.getPerspectiveTransform(corners, dst_points)

    # Aplicar a transformação de perspectiva
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped

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

# Encontrar interseções
if lines is not None:
    intersections = []
    extrapolated_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1_ext, y1_ext, x2_ext, y2_ext = extrapolate_line(x1, y1, x2, y2, img.shape)
        extrapolated_lines.append((x1_ext, y1_ext, x2_ext, y2_ext))

    for i in range(len(extrapolated_lines)):
        for j in range(i + 1, len(extrapolated_lines)):
            intersection = line_intersection(extrapolated_lines[i], extrapolated_lines[j])
            if intersection is not None:
                intersections.append(intersection)

    intersections = np.array(intersections)

    if len(intersections) >= 4:
        hull = cv2.convexHull(intersections)

        if len(hull) >= 4:
            hull = np.squeeze(hull)
            rect = np.zeros((4, 2), dtype='float32')

            s = hull.sum(axis=1)
            rect[0] = hull[np.argmin(s)]  # top-left
            rect[2] = hull[np.argmax(s)]  # bottom-right

            diff = np.diff(hull, axis=1)
            rect[1] = hull[np.argmin(diff)]  # top-right
            rect[3] = hull[np.argmax(diff)]  # bottom-left

            warped = apply_perspective_transform(img, rect)

            cv2.imshow("output", warped)
            wait_key()

cv2.destroyAllWindows()