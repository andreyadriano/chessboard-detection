#!/bin/bash

# Função de ajuda para exibir o uso correto
usage() {
  echo "Uso: $0 -i <input_directory> -o <output_directory>"
  exit 1
}

# Verifica se o número de argumentos é correto
if [ "$#" -ne 4 ]; then
  usage
fi

# Variáveis para armazenar os diretórios de entrada e saída
while getopts "i:o:" opt; do
  case "$opt" in
    i) input_dir="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    *) usage ;;
  esac
done

# Verifica se os diretórios existem
if [ ! -d "$input_dir" ]; then
  echo "Erro: Diretório de entrada '$input_dir' não encontrado."
  exit 1
fi

if [ ! -d "$output_dir" ]; then
  echo "Erro: Diretório de saída '$output_dir' não encontrado."
  exit 1
fi

# Itera sobre todos os arquivos .jpg no diretório de entrada
for img_file in "$input_dir"/*.jpg; do
  # Obtém o nome do arquivo sem o diretório (somente o nome)
  img_name=$(basename "$img_file")

  # Cria o caminho completo do arquivo de saída no diretório de saída
  output_file="$output_dir/$img_name"

  # Executa o script Python para o arquivo atual
  echo "Processando $img_file..."
  python3 detect_chessboard.py "$img_file" "$output_file"

  # Verifica se a execução foi bem-sucedida
  if [ $? -ne 0 ]; then
    echo "Erro ao processar $img_file"
  else
    echo "Arquivo de saída salvo em $output_file"
  fi
done

echo "Processamento concluído."
