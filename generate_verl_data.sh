MODEL=${1:-"Qwen/Qwen3-8B"}                              # Huggingface model (used for embedding)
DATASET=${2:-"lasgroup/verifiable-corpus"}               # Huggingface dataset (training dataset)
BENCHMARK=${3:-"math-ai/aime25"}                         # Huggingface dataset (target benchmark)
DATA_LOCATION=${4="/users/$USER/data"}                   # Where to store the dataset files
PRESELECTION=${5:-"false"}                               # Alternative mean,SIFT, ... (nearest_neighbor)
PRESELECTION_SIZE=${6:-500000}                           # Number of most similar target questions for dataset
OVERWRITE=${7:-"false"}                                  # Whether to overwrite previously created datasets
DATASET_CATEGORY=${8:-"false"}                           # Subset specification for training benchmark
BENCHMARK_CATEGORY=${9:-"false"}                         # Subset specification for target benchmark
EXEC=${10:-"false"}                                      # Whether to execute dataset creation or just print commands
TRAINING_SET_LOCATION=${11:-""}                          # Set if training set should be copied from existing location

if [[ "$BENCHMARK_CATEGORY" == "false" ]]; then
  BENCHMARK_NAME="${BENCHMARK//\//_}"
else
  BENCHMARK_NAME="${BENCHMARK//\//_}_${BENCHMARK_CATEGORY}"
fi
if [[ "$DATASET_CATEGORY" == "false" ]]; then
  DS_NAME="${DATASET//\//_}_${BENCHMARK_NAME}_${PRESELECTION_SIZE}"
else
  DS_NAME="${DATASET//\//_}_${BENCHMARK_NAME}_${PRESELECTION_SIZE}_${DATASET_CATEGORY}"
fi

DATASET_QUESTION_KEY="description"
BENCHMARK_QUESTION_KEY="description"

CMD=""

### Load Benchmark dataset

BENCHMARK_DIR="$DATA_LOCATION/benchmark_datasets"
BENCHMARK_PATH="$BENCHMARK_DIR/${BENCHMARK_NAME}.json"
mkdir -p $BENCHMARK_DIR
if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_PATH" ]]; then
    CMD+="python ${PYTHONPATH}data/load_dataset.py \
     --dataset_name $BENCHMARK \
     --output_path $BENCHMARK_PATH \
     --start_idx 0 \
     --category $BENCHMARK_CATEGORY;"
else
  echo "Benchmark dataset already exists at $BENCHMARK_PATH."
fi

### Create embeddings

EMBEDDING_DIR="${DATA_LOCATION}/embeddings_${MODEL//\//_}"
mkdir -p "$EMBEDDING_DIR"
DATASET_EMBEDDING_PATH="$EMBEDDING_DIR/${DATASET//\//_}.npy"
BENCHMARK_EMBEDDING_PATH="$EMBEDDING_DIR/${BENCHMARK_NAME}.npy"

if [[ ! -f "$DATASET_EMBEDDING_PATH" ]]; then
    CMD+="python ${PYTHONPATH}data/compute_embeddings.py \
     --dataset_name $DATASET \
     --category $DATASET_CATEGORY \
     --output_path $DATASET_EMBEDDING_PATH \
     --question_key $DATASET_QUESTION_KEY \
     --start_data_index 0 \
     --model_id $MODEL \
     --batch_size 8;"
else
  echo "Embeddings already exists at $DATASET_EMBEDDING_PATH."
  echo "python ${PYTHONPATH}data/compute_embeddings.py \
     --dataset_name $DATASET \
     --category $DATASET_CATEGORY \
     --output_path $DATASET_EMBEDDING_PATH \
     --question_key $DATASET_QUESTION_KEY \
     --start_data_index 0 \
     --model_id $MODEL \
     --batch_size 8;"
fi

if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_EMBEDDING_PATH" ]]; then
    CMD+="python ${PYTHONPATH}data/compute_embeddings.py \
     --dataset_name $BENCHMARK \
     --output_path $BENCHMARK_EMBEDDING_PATH \
     --question_key $BENCHMARK_QUESTION_KEY \
     --model_id $MODEL \
     --start_data_index 0 \
     --batch_size 8;"
else
  echo "Embeddings already exists at $BENCHMARK_EMBEDDING_PATH."
fi

### Reload dataset with embeddings
TTRL_DATASET_DIR="$DATA_LOCATION/verl_data/$DS_NAME"
TTRL_DATASET_TRAIN_PATH="$TTRL_DATASET_DIR/train.json"
TTRL_DATASET_TEST_PATH="$TTRL_DATASET_DIR/test.json"
mkdir -p $TTRL_DATASET_DIR
mkdir -p "${PYTHONPATH}TTRL/verl/data"
ln -sf "$DATA_LOCATION/verl_data" "${PYTHONPATH}TTRL/verl/data"

if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_PATH" ]]; then
    CMD+="python ${PYTHONPATH}data/load_dataset.py \
     --dataset_name $BENCHMARK \
     --output_path $TTRL_DATASET_TEST_PATH \
     --start_idx 0 \
     --category $BENCHMARK_CATEGORY \
     --embeddings_file $BENCHMARK_EMBEDDING_PATH;"
else
  echo "Benchmark dataset already exists at $BENCHMARK_PATH."
fi


if [[ "$TRAINING_SET_LOCATION" == "" ]]; then
  ### Create similarities
  SIMILARITIES_DIR="${DATA_LOCATION}/similarities_${MODEL//\//_}"
  mkdir -p "$SIMILARITIES_DIR"
  SIMILARITIES_PATH="$SIMILARITIES_DIR/${DATASET//\//_}_${BENCHMARK_NAME}.npy"

  if [[ "$OVERWRITE" == "true" || ! -f "$SIMILARITIES_PATH" ]]; then
      CMD+="python ${PYTHONPATH}data/compute_neighborhoods.py \
      --embeddings $BENCHMARK_EMBEDDING_PATH \
      --embeddings_2 $DATASET_EMBEDDING_PATH \
      --method $PRESELECTION \
      --output $SIMILARITIES_PATH;"
  else
    echo "Similarities already exists at $SIMILARITIES_PATH."
  fi

  ### Sort dataset
  DATASET_DIR="${DATA_LOCATION}/datasets_${MODEL//\//_}"
  mkdir -p "$DATASET_DIR"

  if [[ "$OVERWRITE" == "true" || ! -f "$DATASET_PATH" ]]; then
      CMD+="python ${PYTHONPATH}data/sort_dataset.py \
      --dataset_name $DATASET \
      --output_path $TTRL_DATASET_TRAIN_PATH \
      --similarities_file_name $SIMILARITIES_PATH \
      --start_idx 0 \
      --num_el $PRESELECTION_SIZE \
      --category $DATASET_CATEGORY \
      --embeddings_file $DATASET_EMBEDDING_PATH;"
  else
    echo "Sorted dataset already exists at $DATASET_PATH."
  fi
  
  ### Preprocess train and test data
  CMD+="python ${PYTHONPATH}data/preprocess.py \
     --data_source $TTRL_DATASET_DIR
     --test_only False;"
else
  mkdir -p $TTRL_DATASET_DIR
  
  ### Copy Training Set
  cp "$TRAINING_SET_LOCATION/train.json" "$TTRL_DATASET_DIR/train.json"
  cp "$TRAINING_SET_LOCATION/train.parquet" "$TTRL_DATASET_DIR/train.parquet"
  
  ### Preprocess train and test data
  CMD+="python ${PYTHONPATH}data/preprocess.py \
     --data_source $TTRL_DATASET_DIR
     --test_only True;"
fi

### Run command if execute is true

if [[ "$EXEC" == "true" ]]; then
  echo -e $CMD
  eval "$CMD"
else
  echo -e $CMD
fi