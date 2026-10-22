#!/usr/bin/env bash
# Download the pretrained embeddings used as baseline meaning models.
# SWOW and the Codenames wordlist must be fetched manually; see data/README.md.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR/embeddings"
cd "$DATA_DIR/embeddings"

if [ ! -f glove.6B.300d.txt ]; then
  echo "Downloading GloVe 6B (~820MB zip)..."
  curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
  unzip -o glove.6B.zip glove.6B.300d.txt
  rm glove.6B.zip
fi

if [ ! -f numberbatch-en-19.08.txt.gz ]; then
  echo "Downloading ConceptNet Numberbatch (English, ~300MB)..."
  curl -L -o numberbatch-en-19.08.txt.gz \
    https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
fi

echo "Done. Files in $DATA_DIR/embeddings"
