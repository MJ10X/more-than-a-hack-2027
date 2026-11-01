#!/usr/bin/env bash
# Fetch Metrica Sports' public sample data (3 anonymised matches, tracking + event data) into
# data/external/metrica-sample/. Used only as a validation set from OTHER games -- see RULES.md,
# "Data of this kind about other games is fine". Source: https://github.com/metrica-sports/sample-data
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${MTAH_DATA_DIR:-$ROOT/data}/external/metrica-sample"
BASE="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

FILES=(
  Sample_Game_1/Sample_Game_1_RawEventsData.csv
  Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv
  Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv
  Sample_Game_2/Sample_Game_2_RawEventsData.csv
  Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv
  Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv
  Sample_Game_3/Sample_Game_3_events.json
  Sample_Game_3/Sample_Game_3_metadata.xml
  Sample_Game_3/Sample_Game_3_tracking.txt
)

for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  if [ ! -s "$DEST/$f" ]; then
    echo "fetching $f"
    curl -fsSL "$BASE/$f" -o "$DEST/$f"
  fi
done
echo "done: $DEST"
