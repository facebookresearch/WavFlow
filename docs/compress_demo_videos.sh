#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -e

SRC="/data/repos/fbsource/fbcode/scripts/feiyanzhou/demo_video"
DST="/data/repos/fbsource/fbcode/scripts/feiyanzhou/wavflow_project/docs/videos/demo"

mkdir -p "$DST"

shopt -s nullglob
files=("$SRC"/*.mp4)
total=${#files[@]}
i=0

for f in "${files[@]}"; do
  i=$((i + 1))
  name=$(basename "$f")
  out="$DST/$name"

  if [ -f "$out" ]; then
    printf "[%2d/%d] skip (exists): %s\n" "$i" "$total" "$name"
    continue
  fi

  printf "[%2d/%d] encoding: %s ..." "$i" "$total" "$name"

  # scale to fit inside 854x480 keeping aspect ratio, then pad to exactly 854x480
  ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -vf "scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1" \
    -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p \
    -c:a copy \
    -movflags +faststart \
    "$out"

  size=$(du -h "$out" | cut -f1)
  echo " done ($size)"
done

echo ""
echo "Source total: $(du -sh "$SRC" | cut -f1)"
echo "Output total: $(du -sh "$DST" | cut -f1)"
