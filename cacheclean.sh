#!/bin/bash

# Find all __pycache__ directories recursively starting from the current directory
find . -name "__pycache__" -type d -print0 | while IFS= read -r -d $'\0' dir; do
  # Remove the directory.  -r is important for recursive delete
  rm -rf "$dir"
  echo "Removed: $dir"  # Optional:  Show what was deleted
done

echo "Finished removing __pycache__ directories."