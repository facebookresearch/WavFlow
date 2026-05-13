#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -e

SRC="/data/repos/fbsource/fbcode/scripts/feiyanzhou/moviegen_wavflow_mmaudio_moviegen_contrast"
DST="/data/repos/fbsource/fbcode/scripts/feiyanzhou/wavflow_project/docs/videos/moviegen"

mkdir -p "$DST"

# Map original prefix -> case number (numerically sorted, ignoring random "267" position)
declare -a CASES=(
    "0_eattingapple"
    "1_penguinwalking"
    "2_boxing"
    "3_horsetrotting"
    "4_bear"
    "5_firework"
    "8_dog_drinking"
    "9_car_approaching"
    "10_rub_plasticbag"
    "267_catplaying"
)

encode() {
    local in="$1" out="$2"
    if [ -f "$out" ]; then
        printf "    skip (exists): %s\n" "$(basename "$out")"
        return
    fi
    printf "    encoding: %s ..." "$(basename "$out")"
    ffmpeg -hide_banner -loglevel error -y -i "$in" \
        -vf "scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1" \
        -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p \
        -c:a copy \
        -movflags +faststart \
        "$out"
    echo " done ($(du -h "$out" | cut -f1))"
}

idx=0
for prefix in "${CASES[@]}"; do
    idx=$((idx + 1))
    cs=$(printf "case%02d" "$idx")
    echo "[$idx/10] $prefix"

    encode "$SRC/${prefix}_0_wavflow.mp4" "$DST/${cs}_wavflow.mp4"
    encode "$SRC/${prefix}_mmaudio.mp4"   "$DST/${cs}_mmaudio.mp4"
    encode "$SRC/${prefix}_moviegen.mp4"  "$DST/${cs}_moviegen.mp4"
done

echo ""
echo "Source total: $(du -sh "$SRC" | cut -f1)"
echo "Output total: $(du -sh "$DST" | cut -f1)"
