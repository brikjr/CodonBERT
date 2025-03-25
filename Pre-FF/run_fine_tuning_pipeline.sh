#!/bin/bash

# Run the entire fine-tuning pipeline for CodonBERT on RNA elements

# Define variables and defaults
RNA_ELEMENTS_CSV="rna_elements.csv"
OUTPUT_DIR="output"
MODEL_DIR="../model"
ELEMENT_TYPE="" # Leave empty to use all types or specify "RBP", "UTR", "miRNA"
EXPRESSION_CSV="rna_elements_expression.csv"
WINDOW_SIZE=300
STRIDE=100
BATCH_SIZE=16
LEARNING_RATE=5e-5
EPOCHS=10
USE_LORA=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data)
      RNA_ELEMENTS_CSV="$2"
      shift
      shift
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --model)
      MODEL_DIR="$2"
      shift
      shift
      ;;
    --element_type)
      ELEMENT_TYPE="$2"
      shift
      shift
      ;;
    --window_size)
      WINDOW_SIZE="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --use_lora)
      USE_LORA=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directories
mkdir -p "${OUTPUT_DIR}/fine_tuned"
mkdir -p "${OUTPUT_DIR}/analysis"

# Set element type argument if specified
if [[ -n "${ELEMENT_TYPE}" ]]; then
  ELEMENT_TYPE_ARG="--element_type ${ELEMENT_TYPE}"
else
  ELEMENT_TYPE_ARG=""
fi

# Set LoRA argument if specified
if [[ "${USE_LORA}" = true ]]; then
  LORA_ARG="--use_lora"
else
  LORA_ARG=""
fi

echo "============================================================"
echo "                CodonBERT Fine-Tuning Pipeline               "
echo "============================================================"
echo "Starting time: $(date)"
echo "Data file: ${RNA_ELEMENTS_CSV}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model directory: ${MODEL_DIR}"
if [[ -n "${ELEMENT_TYPE}" ]]; then
  echo "RNA element type: ${ELEMENT_TYPE}"
else
  echo "RNA element type: All types"
fi
echo "Window size: ${WINDOW_SIZE}"
echo "Stride: ${STRIDE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${EPOCHS}"
echo "Using LoRA: ${USE_LORA}"
echo "============================================================"

# Step 1: Create simulated expression dataset
echo "STEP 1: Creating simulated expression dataset..."
python create_expression_dataset.py \
  --input "${RNA_ELEMENTS_CSV}" \
  --output "${EXPRESSION_CSV}" \
  ${ELEMENT_TYPE_ARG}

if [ $? -ne 0 ]; then
  echo "Error creating expression dataset"
  exit 1
fi

# Step 2: Fine-tune the model
echo "STEP 2: Fine-tuning CodonBERT model..."
python fine_tune_rna_elements.py \
  --data "${EXPRESSION_CSV}" \
  --model "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}/fine_tuned" \
  --window_size ${WINDOW_SIZE} \
  --stride ${STRIDE} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --epochs ${EPOCHS} \
  ${LORA_ARG} \
  ${ELEMENT_TYPE_ARG} \
  --target_metric "expression"

if [ $? -ne 0 ]; then
  echo "Error fine-tuning the model"
  exit 1
fi

# Step 3: Analyze the model performance
echo "STEP 3: Analyzing fine-tuned model performance..."
python analyze_fine_tuned_model.py \
  --model_dir "${OUTPUT_DIR}/fine_tuned/final_model" \
  --test_data "${EXPRESSION_CSV}" \
  --output_dir "${OUTPUT_DIR}/analysis" \
  --window_size ${WINDOW_SIZE} \
  --stride ${STRIDE} \
  --batch_size ${BATCH_SIZE} \
  ${ELEMENT_TYPE_ARG} \
  --target_metric "expression"

if [ $? -ne 0 ]; then
  echo "Error analyzing model performance"
  exit 1
fi

echo "============================================================"
echo "Pipeline completed successfully!"
echo "Results are available in: ${OUTPUT_DIR}"
echo "Fine-tuned model: ${OUTPUT_DIR}/fine_tuned/final_model"
echo "Analysis results: ${OUTPUT_DIR}/analysis"
echo "Ending time: $(date)"
echo "============================================================" 