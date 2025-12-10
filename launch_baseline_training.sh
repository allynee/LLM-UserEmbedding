#!/bin/bash

# Configuration
MODEL="lightgcn"
DATASET="amazon"
GPU_ID="0"

# --- Session 1: Backbone ---
SESSION_NAME="train_backbone"
CMD="python encoder/train_encoder.py --model ${MODEL} --dataset ${DATASET} --cuda ${GPU_ID}"
echo "Starting session: $SESSION_NAME..."
# Create detached session, run command, and keep window open (with 'read') if it crashes/finishes
tmux new-session -d -s "$SESSION_NAME" "$CMD; echo 'Process finished. Press Enter to close.'; read"

# --- Session 2: RLMRec-Con (Contrastive) ---
SESSION_NAME="train_rlmrec_con"
CMD="python encoder/train_encoder.py --model ${MODEL}_plus --dataset ${DATASET} --cuda ${GPU_ID}"
echo "Starting session: $SESSION_NAME..."
tmux new-session -d -s "$SESSION_NAME" "$CMD; echo 'Process finished. Press Enter to close.'; read"

# --- Session 3: RLMRec-Gen (Generative) ---
SESSION_NAME="train_rlmrec_gen"
CMD="python encoder/train_encoder.py --model ${MODEL}_gene --dataset ${DATASET} --cuda ${GPU_ID}"
echo "Starting session: $SESSION_NAME..."
tmux new-session -d -s "$SESSION_NAME" "$CMD; echo 'Process finished. Press Enter to close.'; read"

echo "---------------------------------------------------"
echo "All 3 sessions started!"
echo "Use 'tmux ls' to see them."
echo "Use 'tmux attach -t <session_name>' to view output."