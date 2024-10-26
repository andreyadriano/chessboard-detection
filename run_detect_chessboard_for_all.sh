#!/bin/bash

# Help function to display correct usage
usage() {
  echo "Usage: $0 -i <input_directory> -o <output_directory>"
  exit 1
}

# Checks if the number of arguments is correct
if [ "$#" -ne 4 ]; then
  usage
fi

# Variables to store input and output directories
while getopts "i:o:" opt; do
  case "$opt" in
    i) input_dir="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    *) usage ;;
  esac
done

# Checks if the directories exist
if [ ! -d "$input_dir" ]; then
  echo "Error: Input directory '$input_dir' not found."
  exit 1
fi

if [ ! -d "$output_dir" ]; then
  echo "Error: Output directory '$output_dir' not found."
  exit 1
fi

# Iterate over all .jpg files in the input directory
for img_file in "$input_dir"/*.jpg; do
  # Get the file name without the directory (only the name)
  img_name=$(basename "$img_file")

  # Create the full output file path in the output directory
  output_file="$output_dir/$img_name"

  # Run the Python script for the current file
  echo "Processing $img_file..."
  python3 detect_chessboard.py "$img_file" "$output_file"

  # Check if the execution was successful
  if [ $? -ne 0 ]; then
    echo "Error processing $img_file"
  else
    echo "Output file saved to $output_file"
  fi
done

echo "Processing complete."
