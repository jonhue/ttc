MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET=${2:-"lasgroup/ttt_reasoning_dataset"}      # Huggingface dataset (open-r1/DAPO-Math-17k-Processed)
BENCHMARK=${3:-"math-ai/aime25"}                    # Huggingface dataset
SELECTION=${4:-"nearest_neighbor"}                  # Alternative mean,SIFT, ...
SELECTION_SIZE=${5:-1000}                           # Number of most similar target questions for dataset
OVERWRITE=${6:-false}
DS_FILTER=${7:-false}
BENCHMARK_CATEGORY=${8:-"false"}
EXEC=${9:-"false"}
TRAINING_SET_LOCATION=${10:-""}
declare -A split_map
declare -A question_key_map

# ./setup_training.sh Qwen/Qwen3-8B lasgroup/ttt_reasoning_dataset math-ai/aime25 nearest_neighbor 1000 true

if [[ "$BENCHMARK_CATEGORY" == "false" ]]; then
  BENCHMARK_NAME="${BENCHMARK//\//_}"
else
  BENCHMARK_NAME="${BENCHMARK//\//_}_${BENCHMARK_CATEGORY}"
fi
if [[ "$DS_FILTER" == "false" ]]; then
  DS_NAME="${DATASET//\//_}_${BENCHMARK_NAME}_${SELECTION_SIZE}"
else
  DS_NAME="${DATASET//\//_}_${BENCHMARK_NAME}_${SELECTION_SIZE}_${DS_FILTER}"
fi

split_map=(
  ["lasgroup/ttt_reasoning_dataset"]="train"
  ["math-ai/math500-5"]="test"
  ["open-r1/DAPO-Math-17k-Processed"]="train"
  ["open-thoughts/OpenThoughts3-1.2M"]="train"
  ["math-ai/math500"]="test"
  ["math-ai/aime24"]="test"
  ["math-ai/aime25"]="test"
  ["math-ai/amc23"]="test"
  ["Qwen/CodeElo"]="test"
  ["open-r1/codeforces"]="test"
  ["daman1209arora/jeebench"]="test"
  ["MathArena/hmmt_feb_2025"]="train"
  ["cais/hle"]="test"
  ["Idavidrein/gpqa-D"]="train"
  ["TIGER-Lab/MMLU-Pro"]="test"
  ["livecodebench/code_generation_lite-v5"]="test"
  ["livecodebench/code_generation_lite-v6"]="test"
  ["evalplus/humanevalplus"]="test"
  ["evalplus/mbppplus"]="test"
  ["openai/gsm8k"]="test"
)

DATASET_SPLIT=${split_map[$DATASET]}
DATASET_QUESTION_KEY="description"
BENCHMARK_SPLIT=${split_map[$BENCHMARK]}
BENCHMARK_QUESTION_KEY="description"

CMD=""

### Load Benchmark dataset

BENCHMARK_DIR="/users/$USER/ldiazbone-shared/data/benchmark_datasets"
BENCHMARK_PATH="$BENCHMARK_DIR/${BENCHMARK_NAME}.json"
mkdir -p $BENCHMARK_DIR
if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_PATH" ]]; then
    CMD+="python /users/$USER/ttt-reasoning/data/load_dataset.py \
     --dataset_name $BENCHMARK \
     --dataset_split $BENCHMARK_SPLIT \
     --output_path $BENCHMARK_PATH \
     --start_idx 0 \
     --category $BENCHMARK_CATEGORY;"
else
  echo "Benchmark dataset already exists at $BENCHMARK_PATH."
fi

### Create embeddings

EMBEDDING_DIR="/users/$USER/ldiazbone-shared/data/embeddings_${MODEL//\//_}"
mkdir -p "$EMBEDDING_DIR"
DATASET_EMBEDDING_PATH="$EMBEDDING_DIR/${DATASET//\//_}.npy"
BENCHMARK_EMBEDDING_PATH="$EMBEDDING_DIR/${BENCHMARK_NAME}.npy"

if [[ ! -f "$DATASET_EMBEDDING_PATH" ]]; then
    CMD+="python /users/$USER/ttt-reasoning/embeddings/scripts/compute_embeddings.py \
     --dataset_name $DATASET \
     --dataset_split $DATASET_SPLIT \
     --output_path $DATASET_EMBEDDING_PATH \
     --question_key $DATASET_QUESTION_KEY \
     --start_data_index 0 \
     --model_id $MODEL \
     --batch_size 8;"
else
  echo "Embeddings already exists at $DATASET_EMBEDDING_PATH."
  echo "python /users/$USER/ttt-reasoning/embeddings/scripts/compute_embeddings.py \
     --dataset_name $DATASET \
     --dataset_split $DATASET_SPLIT \
     --output_path $DATASET_EMBEDDING_PATH \
     --question_key $DATASET_QUESTION_KEY \
     --start_data_index 0 \
     --model_id $MODEL \
     --batch_size 8;"
fi

if [[ "$BENCHMARK_CATEGORY" == "false" ]]; then
  BENCHMARK_SPLIT=$BENCHMARK_SPLIT
  LOCAL_PATH="false"
else
  echo "Local"
  BENCHMARK_SPLIT="train"
  LOCAL_PATH=$BENCHMARK_PATH
fi

if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_EMBEDDING_PATH" ]]; then
    CMD+="python /users/$USER/ttt-reasoning/embeddings/scripts/compute_embeddings.py \
     --dataset_name $BENCHMARK \
     --dataset_split $BENCHMARK_SPLIT \
     --output_path $BENCHMARK_EMBEDDING_PATH \
     --question_key $BENCHMARK_QUESTION_KEY \
     --model_id $MODEL \
     --start_data_index 0 \
     --batch_size 8 \
     --local_path $LOCAL_PATH;"
else
  echo "Embeddings already exists at $BENCHMARK_EMBEDDING_PATH."
fi


### Reload dataset with embeddings
TTRL_DATASET_DIR="/users/$USER/ldiazbone-shared/data/verl_data/data/$DS_NAME" #${DATASET//\//_}_${BENCHMARK//\//_} #"/users/$USER/ttt-reasoning/TTRL/verl/data/${DATASET//\//_}_${BENCHMARK//\//_}"
TTRL_DATASET_TRAIN_PATH="$TTRL_DATASET_DIR/train.json"
TTRL_DATASET_TEST_PATH="$TTRL_DATASET_DIR/test.json"
mkdir -p $TTRL_DATASET_DIR

if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_PATH" ]]; then
    CMD+="python /users/$USER/ttt-reasoning/data/load_dataset.py \
     --dataset_name $BENCHMARK \
     --dataset_split $BENCHMARK_SPLIT \
     --output_path $TTRL_DATASET_TEST_PATH \
     --start_idx 0 \
     --category $BENCHMARK_CATEGORY \
     --embeddings_file $BENCHMARK_EMBEDDING_PATH;"
else
  echo "Benchmark dataset already exists at $BENCHMARK_PATH."
fi

### Create similarities

SIMILARITIES_DIR="/users/$USER/ldiazbone-shared/data/similarities_${MODEL//\//_}"
mkdir -p "$SIMILARITIES_DIR"
SIMILARITIES_PATH="$SIMILARITIES_DIR/${DATASET//\//_}_${BENCHMARK_NAME}.npy" #{DATASET//\//_}_${BENCHMARK//\//_}

if [[ "$OVERWRITE" == "true" || ! -f "$BENCHMARK_PATH" ]]; then
    CMD+="python /users/$USER/ttt-reasoning/data/load_dataset.py \
     --dataset_name $BENCHMARK \
     --dataset_split $BENCHMARK_SPLIT \
     --output_path $TTRL_DATASET_TEST_PATH \
     --start_idx 0 \
     --category $BENCHMARK_CATEGORY \
     --embeddings_file $BENCHMARK_EMBEDDING_PATH;"
else
  echo "Benchmark dataset already exists at $BENCHMARK_PATH."
fi

### Create similarities

if [[ "$TRAINING_SET_LOCATION" == "" ]]; then
  SIMILARITIES_DIR="/users/$USER/ldiazbone-shared/data/similarities_${MODEL//\//_}"
  mkdir -p "$SIMILARITIES_DIR"
  SIMILARITIES_PATH="$SIMILARITIES_DIR/${DATASET//\//_}_${BENCHMARK_NAME}.npy" #{DATASET//\//_}_${BENCHMARK//\//_}

  if [[ "$OVERWRITE" == "true" || ! -f "$SIMILARITIES_PATH" ]]; then
      CMD+="python /users/$USER/ttt-reasoning/embeddings/scripts/compute_neighborhoods.py \
      --embeddings $BENCHMARK_EMBEDDING_PATH \
      --embeddings_2 $DATASET_EMBEDDING_PATH \
      --output $SIMILARITIES_PATH;"
  else
    echo "Similarities already exists at $SIMILARITIES_PATH."
  fi

  ### Sort dataset

  DATASET_DIR="/users/$USER/ldiazbone-shared/data/datasets_${MODEL//\//_}"
  mkdir -p "$DATASET_DIR"
  #DATASET_PATH="$DATASET_DIR/$DS_NAME.json" #${DATASET//\//_}_${BENCHMARK//\//_}

  if [[ "$OVERWRITE" == "true" || ! -f "$DATASET_PATH" ]]; then
      CMD+="python /users/$USER/ttt-reasoning/data/sort_dataset.py \
      --dataset_name $DATASET \
      --output_path $TTRL_DATASET_TRAIN_PATH \
      --similarities_file_name $SIMILARITIES_PATH \
      --start_idx 0 \
      --num_el $SELECTION_SIZE \
      --data_source $DS_FILTER \
      --embeddings_file $DATASET_EMBEDDING_PATH;"
  else
    echo "Sorted dataset already exists at $DATASET_PATH."
  fi
else
  TTRL_DATASET_DIR="/users/$USER/ldiazbone-shared/data/verl_data/data/$DS_NAME" #${DATASET//\//_}_${BENCHMARK//\//_} #"/users/$USER/ttt-reasoning/TTRL/verl/data/${DATASET//\//_}_${BENCHMARK//\//_}"
  TTRL_DATASET_TRAIN_PATH="$TTRL_DATASET_DIR/train.json"
  mkdir -p $TTRL_DATASET_DIR
  ### COPY CURRENT FULL TRAIN DATASET
  #FULL_TRAIN_DATASET_PATH="/users/$USER/ldiazbone-shared/data/verl_data/data/lasgroup_ttt_reasoning_dataset_openai_gsm8k_500000/train.json"
  cp $TRAINING_SET_LOCATION "$TTRL_DATASET_DIR/train.json"
fi



# ## Run command if execute is true

if [[ "$EXEC" == "true" ]]; then
  echo -e $CMD
  eval "$CMD"
  CMD=""
else
  echo "Just print"
fi



### Create verl train dataset



# if [[ "$OVERWRITE" == "true" || ! -f "$TTRL_DATASET_TRAIN_PATH" ]]; then
#     cp $DATASET_PATH "$TTRL_DATASET_DIR/train_temp.json"
#     {
#     echo "["
#     sed \
#         -e '$! s/}$/},/' \
#         "$TTRL_DATASET_DIR/train_temp.json"
#     echo "]"
#     } > "$TTRL_DATASET_TRAIN_PATH"
#     rm "$TTRL_DATASET_DIR/train_temp.json"
#     #-e "s/\"prompt\"/\"$DATASET_QUESTION_KEY\"/g"\
#     #-e 's/"solution"/"answer"/g' \
# else
#   echo "Verl train dataset already exists at $TTRL_DATASET_TRAIN_PATH."
# fi

### Create verl test dataset

# TTRL_DATASET_TEST_PATH="$TTRL_DATASET_DIR/test.json"
# if [[ "$OVERWRITE" == "true" || ! -f "$TTRL_DATASET_TEST_PATH" ]]; then
#     cp $BENCHMARK_PATH "$TTRL_DATASET_DIR/test_temp.json"
#     {
#     echo "["
#     sed \
#         -e '$! s/}$/},/' \
#         "$TTRL_DATASET_DIR/test_temp.json"
#     echo "]"
#     } > "$TTRL_DATASET_TEST_PATH"
#     rm "$TTRL_DATASET_DIR/test_temp.json"
#     #-e "s/\"prompt\"/\"$BENCHMARK_QUESTION_KEY\"/g"\
#     #-e 's/"solution"/"answer"/g' \
# else
#   echo "Verl test dataset already exists at $TTRL_DATASET_TEST_PATH."
# fi

### Preprocess train and test data

CMD+="python /users/$USER/ttt-reasoning/data/preprocess.py \
     --data_source $TTRL_DATASET_DIR;"

### Print or run all commands

echo -e $CMD

if [[ "$EXEC" == "true" ]]; then
  eval "$CMD"
  CMD=""
else
  echo ""
fi
