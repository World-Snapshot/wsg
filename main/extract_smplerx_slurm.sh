#!/usr/bin/env bash
set -x

PARTITION=Zoetrope

INPUT_VIDEO=$1
FORMAT=$2
FPS=$3
CKPT=$4
OUTPUT_FOLDER=$5
SMPLERX_PYTHON=${SMPLERX_PYTHON:-/research/cbim/vast/sf895/miniforge3/envs/smplerx/bin/python}

GPUS=1
JOB_NAME=inference_${INPUT_VIDEO}

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4 # ${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

#IMG_PATH=../demo/images/${INPUT_VIDEO}
#SAVE_DIR=../demo/results/${INPUT_VIDEO}
CROP_FRAME_PATH=${OUTPUT_FOLDER}/crop_frame
SAVE_DIR_SMPL=${OUTPUT_FOLDER}/results_smplerx
#RECORD_DIR=${OUTPUT_FOLDER}/record
RECORD_DIR_SMPL="./${OUTPUT_FOLDER#../}/record_smplerx"


# 获取输出视频同名目录的路径部分 (../xxxx)
OUTPUT_FOLDER_PATH=$(dirname "${OUTPUT_FOLDER}")
# 获取输出视频同名目录的文件名部分，文件名就是视频名 (video_name)
OUTPUT_VIDEO_NAME=$(basename "${OUTPUT_FOLDER}")


# video to images
mkdir -p $CROP_FRAME_PATH
mkdir -p $SAVE_DIR_SMPL


# ffmpeg -i ${INPUT_VIDEO}.${FORMAT} -f image2 -vf fps=${FPS}/1 -qscale 0 ${OUTPUT_FOLDER}/img/%06d.jpg 

end_count=$(find "$CROP_FRAME_PATH" -type f | wc -l)
echo $end_count

# inference
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="${SCRIPT_DIR}/transformer_utils:${SCRIPT_DIR}/..:${PYTHONPATH}" \
"${SMPLERX_PYTHON}" extract_smplerx_inference.py \
    --num_gpus ${GPUS_PER_NODE} \
    --output_folder ${SAVE_DIR_SMPL} \
    --pretrained_model ${CKPT} \
    --agora_benchmark agora_model \
    --img_path ${CROP_FRAME_PATH} \
    --start 1 \
    --end  $end_count \
    --multi_person \
    --det_score_thr "${SMPLERX_DET_SCORE_THR:-0.7}" \
    --containment_thr "${SMPLERX_CONTAINMENT_THR:-0.85}" \
    --iou_thr "${SMPLERX_IOU_THR:-0.5}" \
    --bbox_thr "${SMPLERX_BBOX_THR:-50}"
    #--exp_name ${RECORD_DIR_SMPL} \
    #--exp_name output/demo_${JOB_NAME} \



# 注释掉渲染和视频生成步骤以加速处理。如果你需要可视化请取消注释。
# python get_smplerx_render.py \
#     --data_path ${OUTPUT_FOLDER_PATH} --seq ${OUTPUT_VIDEO_NAME} \
#     --image_path ${CROP_FRAME_PATH} \
#     --render_biggest_person False \
#     --load_mode smplerx
# 
# # orig images to video, 使用 H.264 编码器
# ffmpeg -y -f image2 -r ${FPS} -i ${OUTPUT_FOLDER}/smplerx_overlay_img/%08d.jpg -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p ${OUTPUT_FOLDER}/visualization_smplerx_overlap.mp4
