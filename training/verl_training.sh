#!/bin/bash
# export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
# export RAY_DEBUG=1

BACKBONE=${1:-"Qwen/Qwen3-8B"}
TASK=${2:-"lasgroup_verifiable-corpus_math-ai_aime25_500000"}
DYNAMIC_DATASET=${3:-"True"}
CURRICULUM=${4:-"False"}
EXPERIMENT=${5:-"test"}
ADVANTAGE=${6:-"grpo"}  # passk_combined, grpo
ACQUISITION_FUNCTION=${7:-"vtl"} # only used if DYNAMIC_DATASET is True
MAJ_ON_TEST=${8:-"False"}  # only used if DYNAMIC_DATASET is True
FILTER_ACHIEVABILITY=${9:-"False"}  # only used if CURRICULUM is True (Should be the same as CURRICULUM)
UPDATE_DELAY=${10:-10}  # only used if CURRICULUM is True
DATASET_SIZE=${11:-1000}
TOTAL_DATASET_SIZE=${12:-1000}  # total number of datapoints for training until completing an episode, falls back to DATASET_SIZE if CURRICULUM is False
FILTER_OVERLONG_PROMPTS=${13:-"True"}
RESUME_MODE=${14:-"auto"}
EPISODE=${15:-2}  # automatically set to 1 if CURRICULUM is True
LOG_NAME=${16:-""}
SIFT_LAMBDA=${17:-"0.1"}
CLIP_RATIO_HIGH=${18:-"0.28"}
K=${19:-"8"}
N=${20:-"4"}
RESUME_PATH=${21:-"none"}
DATA_RESUME=${22-"True"}
THINKING=${23-"False"}
FILTER_KIND=${24:-""}


PASS_K=8

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

SEED=42

NNODES=1
N_GPUS_PER_NODE=4

VAL_FREQ=10
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=$((1024 * $K))

VERIFIER_NAME="TIGER-Lab/general-verifier"
VERIFIER_LEN=2048

DATA_TRAIN_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=16
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2
SAVE_FREQ=10 # One checkpoint per validation
MEMORY_UTILIZATION=0.6
DATA_LOCAL_DIR="/users/$USER/ttt-reasoning/TTRL/verl/data"
BACKBONE_PATH="${BACKBONE}"
MAX_ACTOR_CKPT_TO_KEEP=1

SPARSE_REWARDS=False
MAX_TEST_CASES=20
MAJ_THRESHOLD=0

CLIP_RATIO_ARGS="actor_rollout_ref.actor.clip_ratio_low=0.2 \
actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH"


if [ "$RESUME_PATH" == "none" ]; then
  VAL_ONLY_ARGS=""
else
  echo "Validation-only with path: $RESUME_PATH"
  if [ "$RESUME_PATH" == "initial" ]; then
    RESUME_MODE="disable"
  else
    RESUME_MODE="resume_path"
    VAL_ONLY_ARGS="trainer.resume_from_path=$RESUME_PATH \
    trainer.val_before_train=True"
  fi
  if [ "$DATA_RESUME" == "True" ]; then  # unless DATA_RESUME is False, we set the following to effectively stop training after the initial validation
    SAVE_FREQ="-1"
    DYNAMIC_DATASET="False"
    FILTER_OVERLONG_PROMPTS="False"
    EPISODE="1"  # doesn't sync to w&b otherwise
    VAL_FREQ="10000"
  fi
fi

RESUME_ARGS="trainer.resume_mode=$RESUME_MODE"


if [ "$CURRICULUM" == "True" ]; then
  EPISODE="1"
  SUBSET_SIZE=$((DATA_TRAIN_BATCH_SIZE * UPDATE_DELAY))
  DATA_SELECTION_ARGS="data.dynamic.enable=$DYNAMIC_DATASET \
  data.dynamic.maj_on_test=$MAJ_ON_TEST \
  data.dynamic.acquisition_function=$ACQUISITION_FUNCTION \
  data.dynamic.sift_lambda=$SIFT_LAMBDA \
  data.dynamic.filter_achievability.enable=$FILTER_ACHIEVABILITY \
  data.dynamic.filter_achievability.min_ach_band=0.25 \
  data.dynamic.filter_achievability.max_ach_band=0.6 \
  data.dynamic.filter_achievability.min_questions_in_band=1000 \
  data.dynamic.filter_achievability.linear_estimation_offset_clip=0.5 \
  data.dynamic.filter_kind=$FILTER_KIND \
  data.dynamic.subset_size=$SUBSET_SIZE \
  data.dynamic.total_data_size=$TOTAL_DATASET_SIZE \
  data.dynamic.resume=$DATA_RESUME"
else
  DATA_SELECTION_ARGS="data.dynamic.enable=$DYNAMIC_DATASET \
  data.dynamic.maj_on_test=$MAJ_ON_TEST \
  data.dynamic.filter_achievability.enable=False \
  data.dynamic.acquisition_function=$ACQUISITION_FUNCTION \
  data.dynamic.sift_lambda=$SIFT_LAMBDA \
  data.dynamic.filter_kind=$FILTER_KIND \
  data.dynamic.subset_size=$DATASET_SIZE \
  data.dynamic.total_data_size=$DATASET_SIZE \
  data.dynamic.resume=$DATA_RESUME"
fi

WANDB_PROJECT="TTCs"
EXPERIMENT_IDENTIFIER="params_${CURRICULUM}_${ACQUISITION_FUNCTION}_${MAJ_ON_TEST}_${SUBSET_SIZE}_${TOTAL_DATASET_SIZE}_${EPISODE}_${SEED}" # Should contain all hyperparameters relevant for distinguishing experiments
OUTPUT_DIR="/capstor/scratch/cscs/$USER/ttc_runs/${EXPERIMENT}/${TASK}/${BACKBONE//\//_}/${EXPERIMENT_IDENTIFIER}"
if [ "$LOG_NAME" == "" ]; then
  LOG_NAME="${EXPERIMENT}-${TASK}-${BACKBONE//\//_}-${ADVANTAGE}-${EXPERIMENT_IDENTIFIER}"
else
  echo $LOG_NAME
fi


# ------------------------------------------------------------
ARGUMENTS="reward_model.reward_manager=naive \
reward_model.strategy=verifier \
reward_model.reward_kwargs.sparse_rewards=$SPARSE_REWARDS \
reward_model.reward_kwargs.max_test_cases=$MAX_TEST_CASES \
reward_model.reward_kwargs.maj_threshold=$MAJ_THRESHOLD \
reward_model.enable=True \
reward_model.model.path=$VERIFIER_NAME \
reward_model.model.input_tokenizer=$BACKBONE_PATH \
reward_model.model.max_new_tokens=$VERIFIER_LEN \
reward_model.micro_batch_size=0 \
reward_model.num_examine=2 \
data.train_files=[\"$DATA_LOCAL_DIR/$TASK/train.parquet\"] \
data.val_files=[\"$DATA_LOCAL_DIR/$TASK/test.parquet\"] \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS \
data.truncation='right' \
data.shuffle=True \
$DATA_SELECTION_ARGS \
data.thinking=$THINKING \
actor_rollout_ref.model.path=$BACKBONE_PATH \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.grad_clip=1.0 \
$CLIP_RATIO_ARGS \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.optim.warmup_style='constant' \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=$MEMORY_UTILIZATION \
actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.n=$N \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
actor_rollout_ref.rollout.val_kwargs.enable_verifier_for_validation=False \
actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
algorithm.kl_ctrl.kl_coef=0.00 \
algorithm.adv_estimator=$ADVANTAGE \
algorithm.pass_k=$PASS_K \
$VAL_ONLY_ARGS \
trainer.logger=['console','wandb'] \
trainer.project_name=$WANDB_PROJECT \
trainer.group_name=$EXPERIMENT \
trainer.experiment_name=$LOG_NAME \
$RESUME_ARGS \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$NNODES \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$VAL_FREQ \
trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
trainer.default_local_dir=$OUTPUT_DIR \
trainer.total_epochs=$EPISODE \
seed=$SEED"
echo $ARGUMENTS
python -m verl.trainer.main_ppo $ARGUMENTS

echo "Output directory: $OUTPUT_DIR"
