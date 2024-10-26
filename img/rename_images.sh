#!/bin/bash

# Check if the starting number was passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <starting_number>"
  exit 1
fi

# Store the provided starting number
counter=$1

# Loop through the files in the current directory that follow the format YYYYMMDD_hhmmss.*
# Sorted by file name (which corresponds to chronological order)
for file in $(ls -1 | grep -E '^[0-9]{8}_[0-9]{6}' | sort); do
  # Extract the file extension
  extension="${file##*.}"

  # Create the new name with the incremented number
  new_name=$(printf "%04d.%s" $counter $extension)

  # Rename the file
  mv "$file" "$new_name"

  # Increment the counter
  counter=$((counter + 1))
done
