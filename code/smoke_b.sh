#!/usr/bin/env bash
# Wave 1-B GPU gate + 1-epoch smoke (Crop36 and COCO, three losses, batch 256 and 64).
set -u
cd ~/MM_Wave1/code
mkdir -p ~/MM_Wave1/logs/B_smoke
python gate_b.py --device cuda 2>&1 | tee ~/MM_Wave1/logs/B_smoke/gate.log
grep -q "GATE B: PASS" ~/MM_Wave1/logs/B_smoke/gate.log || { echo "GATE B FAILED (see gate.log); smoke still runs for timing"; }
for d in crop36 coco; do for l in vsepp clip siglip; do
  python mm_b.py --data $d --loss $l --epochs 1 --seed 777 --out ~/MM_Wave1/runs/B_smoke --force \
    > ~/MM_Wave1/logs/B_smoke/${d}_${l}.log 2>&1
  tail -1 ~/MM_Wave1/logs/B_smoke/${d}_${l}.log
done; done
for l in clip siglip; do
  python mm_b.py --data crop36 --loss $l --batch 64 --epochs 1 --seed 777 --out ~/MM_Wave1/runs/B_smoke --force \
    > ~/MM_Wave1/logs/B_smoke/crop36_${l}_b64.log 2>&1
  tail -1 ~/MM_Wave1/logs/B_smoke/crop36_${l}_b64.log
done
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "SMOKE B FINISHED"
