python3 -W ignore main.py \
  --epoch 100 \
  --lr 1e-4 \
  --data-train-root grasp-anything++/seen \
  --data-val-root grasp-anything++/unseen \
  --batch-size 16 \
  --save-path outdir \