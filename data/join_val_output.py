import pandas as pd

path = "/users/ldiazbone/ldiazbone-shared/data/verl_data/"
folder_list = ["data/lasgroup_ttt_reasoning_dataset_lasgroup_ttt_reasoning_dataset_dapo_math_10/",
               "data/lasgroup_ttt_reasoning_dataset_lasgroup_ttt_reasoning_dataset_dapo_math_10_1/",
               "data/lasgroup_ttt_reasoning_dataset_lasgroup_ttt_reasoning_dataset_dapo_math_10_2/",
               "data/lasgroup_ttt_reasoning_dataset_lasgroup_ttt_reasoning_dataset_dapo_math_10_3/",
               "data/lasgroup_ttt_reasoning_dataset_lasgroup_ttt_reasoning_dataset_math_10/",
               "data/lasgroup_ttt_reasoning_dataset_lasgroup_ttt_reasoning_dataset_gsm8k_10/"]
qwen_file_name = "Qwen_Qwen3-8B_val_output.csv"
apertus_file_name = "_iopsstor_scratch_cscs_smoalla_projects_swiss-alignment_artifacts_shared_outputs_train_sft_final-run_Apertus8B-tokens15T-it2627139-apertus-sft-mixture-7-ln-v2-ademamix_checkpoints_2087d36b7ab2cef8_checkpoint-8926_val_output.csv"

qwen_dfs = []
apertus_dfs = []
for folder in folder_list:
    output_file_qwen = path + folder + qwen_file_name
    output_file_apertus = path + folder + apertus_file_name
    qwen_dfs.append(pd.read_csv(output_file_qwen))
    apertus_dfs.append(pd.read_csv(output_file_apertus))

qwen_df = pd.concat(qwen_dfs)
apertus_df = pd.concat(apertus_dfs)

qwen_df = qwen_df.groupby("indices").mean().reset_index()
apertus_df = apertus_df.groupby("indices").mean().reset_index()

print("Write outputs.")
qwen_df[["indices", "scores"]].to_csv(path + "achievability_estimates/math_Qwen_3_8B.csv")
apertus_df[["indices", "scores"]].to_csv(path + "achievability_estimates/math_apertus_8B.csv")
