#!/bin/bash

ROOT_DIR="."

# Delete all checkpoint_* folders
find "$ROOT_DIR" -type d -name "checkpoints_*" -exec rm -rf {} +

# Delete all training.log files
find "$ROOT_DIR" -type f -name "training.log" -delete

echo "Cleanup complete!"
