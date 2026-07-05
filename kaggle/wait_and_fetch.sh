#!/usr/bin/env bash
# Poll one or more Kaggle kernels until all reach a terminal state, download their
# outputs, copy probability arrays into oof/, and print CV summaries.
# Usage: wait_and_fetch.sh <slug1> [<slug2> ...]
export KAGGLE_API_TOKEN="KGAT_3e5b52454fe8f2c89b7dae5ccfa4413b"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
ROOT="/g/ml/New folder/clinspan"
OOF="$ROOT/oof"
mkdir -p "$OOF"
slugs=("$@")

declare -A done_map
while true; do
  all_done=1
  for slug in "${slugs[@]}"; do
    [ "${done_map[$slug]}" = "1" ] && continue
    st=$(kaggle kernels status "nakuls1nha/$slug" 2>/dev/null | tr -d '\r')
    echo "$(date +%H:%M:%S) $slug -> $st"
    if echo "$st" | grep -qiE "complete|error|cancel"; then
      done_map[$slug]=1
      out="$ROOT/kaggle_out/$slug"
      mkdir -p "$out"; rm -f "$out"/*
      kaggle kernels output "nakuls1nha/$slug" -p "$out" >/dev/null 2>&1
      cp "$out"/oof_*.npy "$out"/test_*.npy "$OOF"/ 2>/dev/null
      echo "  downloaded $slug outputs:"; ls "$out" 2>/dev/null | tr '\n' ' '; echo
      for cj in "$out"/cv_*.json; do
        [ -f "$cj" ] && python -c "import json;d=json.load(open(r'$cj'));print('  CV macro=%.4f plain=%.4f'%(d['macro'],d['plain']))" 2>/dev/null
      done
    else
      all_done=0
    fi
  done
  [ "$all_done" = "1" ] && break
  sleep 120
done
echo "ALL_DONE"