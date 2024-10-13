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
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de mediana para reduzir ruído
    # median_blurred = cv2.medianBlur(gray, 7)

    # Aplicar filtro Gaussian para suavizar ainda mais a imagem
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Aplicar operação morfológica (closing) para remover pequenos buracos e sombras
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # Detectar as bordas usando Canny
    edges = cv2.Canny(morphed, 50, 150, apertureSize=3, L2gradient=True)

    # Aplicar dilatação para fechar pequenas lacunas nas bordas
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)  # Aumentar a iteração para fechar mais lacunas

    # Aplicar Transformada de Hough para detectar as linhas
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

    # Verificar se linhas foram detectadas
    if lines is not None:
        # Desenhar as linhas detectadas na imagem original
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar as imagens intermediárias
    cv2.imshow("output", gray)
    wait_key()

    cv2.imshow("output", blurred)
    wait_key()

    cv2.imshow("output", morphed)
    wait_key()

    cv2.imshow("output", edges)
    wait_key()

    cv2.imshow("output", dilated)
    wait_key()

    # Exibir a imagem original com as linhas detectadas
    cv2.imshow("output", img)
    wait_key()
    cv2.destroyAllWindows()