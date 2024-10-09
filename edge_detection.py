import cv2
import numpy as np
import argparse

def wait_key():
    # Esperar por uma tecla ou fechar a janela após 30 segundos
    if cv2.waitKey(30000) & 0xFF == ord('q'):
        pass

# Configurar o parser de argumentos
parser = argparse.ArgumentParser(description="Detecção de linhas em uma imagem de tabuleiro de xadrez.")
parser.add_argument("imagem_caminho", type=str, help="Caminho para a imagem do tabuleiro de xadrez.")
args = parser.parse_args()

# Carregar a imagem
img = cv2.imread(args.imagem_caminho)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print(f"Erro: Não foi possível carregar a imagem '{args.imagem_caminho}'")
else:
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
    cv2.imshow("Gray", gray)
    wait_key()
    cv2.destroyAllWindows()

    # Aplicar filtro Gaussian para suavizar a imagem e reduzir ruído
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
    cv2.imshow("Blurred", blurred)
    wait_key()
    cv2.destroyAllWindows()

    # Detectar as bordas usando Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.imshow("Edges", edges)
    wait_key()
    cv2.destroyAllWindows()

    # Aplicar dilatação para fechar pequenas lacunas nas bordas
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    cv2.namedWindow("Dilated", cv2.WINDOW_NORMAL)
    cv2.imshow("Dilated", dilated)
    wait_key()
    cv2.destroyAllWindows

    # Aplicar Transformada de Hough para detectar as linhas
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Verificar se linhas foram detectadas
    if lines is not None:
        # Desenhar as linhas detectadas na imagem original
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Exibir a imagem original com as linhas detectadas
    cv2.namedWindow("Linhas Detectadas", cv2.WINDOW_NORMAL)
    cv2.imshow("Linhas Detectadas", img)
    wait_key()
    cv2.destroyAllWindows()
