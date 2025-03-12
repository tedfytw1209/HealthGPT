#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out

QUESTION_FILE=${1:-"./playground/data/eval/iu_xray/simple_questions.jsonl"}
IMAGE_DIR=${2:-"/orange/bianjiang/VLM_dataset/ReportGeneration/IU_X-Ray/Kaggle/images/images_normalized"}
ANSWER_FILE=${3:-"./eval_output/finetune/lc_ct/simple_answers.jsonl"}
METADATA=${4:-"None"}
BS=${5:-"1"}

# Run a tutorial python script within the container. Modify the path to your container and your script.
bash llava/demo/vqa_infer_phi4.sh $QUESTION_FILE $IMAGE_DIR $ANSWER_FILE $METADATA $BS
