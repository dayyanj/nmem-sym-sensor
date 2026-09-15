#!/bin/bash
# Transcode any AV1 videos to h264 in-place
# Run this after downloading new videos
DIR=${1:-.}
count=0
for f in "$DIR"/*.mp4; do
    [ -f "$f" ] || continue
    codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f" 2>/dev/null)
    if [ "$codec" = "av1" ]; then
        echo "Transcoding: $(basename "$f")"
        ffmpeg -i "$f" -c:v libx264 -preset fast -crf 23 -c:a copy -y "${f%.mp4}_h264.mp4" 2>/dev/null
        mv "${f%.mp4}_h264.mp4" "$f"
        count=$((count+1))
    fi
done
echo "Transcoded $count AV1 videos in $DIR"
