#!/bin/bash

DATA_DIR="data"

for file in "$DATA_DIR"/*.jsonl; do
    filename=$(basename "$file" .jsonl)

    # Skip parsed.jsonl 
    if [ "$filename" == "parsed" ]; then 
        echo "Skipping $file" 
        continue 
    fi

    echo "Building index for $file -> $INDEX_DIR/$filename"

    /usr/bin/time -v python RAG/main.py --mode build --data "$file" --rebuild --index "$filename"

done

echo "All indexes built."
