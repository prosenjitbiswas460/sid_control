#!/usr/bin/env bash
# Fetch datasets directly on a server (avoids Python proxy issues).
#   ./scripts/download_data.sh ml1m
#   ./scripts/download_data.sh beauty sports
set -euo pipefail

SNAP="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

fetch() {  # fetch <url> <dest>
  if [[ -s "$2" ]]; then echo "have $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "fetching $1"
  wget -c -O "$2" "$1"
}

amazon() {  # amazon <Tag>
  fetch "$SNAP/reviews_$1_5.json.gz" "data/amazon/reviews_$1_5.json.gz"
  fetch "$SNAP/meta_$1.json.gz"      "data/amazon/meta_$1.json.gz"
}

for target in "$@"; do
  case "$target" in
    ml1m|ml-1m)
      fetch "https://files.grouplens.org/datasets/movielens/ml-1m.zip" \
            "data/ml-1m/ml-1m.zip"
      (cd data/ml-1m && unzip -o -q ml-1m.zip)
      ;;
    beauty) amazon "Beauty" ;;
    sports) amazon "Sports_and_Outdoors" ;;
    toys)   amazon "Toys_and_Games" ;;
    *) echo "unknown target: $target" >&2; exit 1 ;;
  esac
done

echo "done"
du -sh data/* 2>/dev/null || true
