#!/bin/bash
#
# run.sh is the entry point of the submission.
# nvidia-docker run -v ${INPUT_DIR}:/input_images -v ${OUTPUT_DIR}:/output_images
#       -w /competition ${DOCKER_IMAGE_NAME} sh ./run.sh /input_images /output_images
# where:
#   INPUT_DIR - directory with input png images
#   OUTPUT_DIR - directory with output png images
#

INPUT_DIR=$1
OUTPUT_DIR=$2

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_path=/competition/models/inception_v1/inception_v1.ckpt

  INPUT_DIR=$1
OUTPUT_FILE=$2

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path=./models/inception_v1/inception_v1.ckpt