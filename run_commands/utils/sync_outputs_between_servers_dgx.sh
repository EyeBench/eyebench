
# Expected to run from the eyebench_private root directory that has the 'outputs' folder.
# used to sync model checkpoints to laccl-srv1
USER=$(whoami)
tasks=(
    'OneStop_RC' 
    'SBSAT_RC'
    'PoTeC_RC'
    'PoTeC_DE'
    'IITBHGC_CV'
    'CopCo_TYP'
    'SBSAT_STD'
    'CopCo_RCS'
    'MECOL2_LEX'
)
models=(
    'DummyClassifierMLArgs'
    'LogisticRegressionMLArgs'
    'Roberta'
    'LogisticMeziereArgs'
    'SupportVectorMachineMLArgs'
    'XGBoostMLArgs'
    'RandomForestMLArgs'
    'AhnRNN'
    'AhnCNN'
    'BEyeLSTMArgs'
    'PLMASArgs'
    'PLMASfArgs'
    'RoberteyeWord'
    'RoberteyeFixation'
    'MAG'
    'PostFusion'
)


# Run cleanup on all servers
ssh "$USER@dgx-master.technion.ac.il" "cd work/eyebench_private; /home/shubi/athena/miniforge3/envs/decoding/bin/python src/run/multi_run/cleanup_models.py --real_run --keep_one_lowest"

# Sync outputs for specific task-model combinations
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        # Pattern to match: +data=TASK,+model=MODEL,+trainer=...
        PATTERN="+data=${task},+model=${model}"
        echo "Syncing outputs for $task - $model"
        
        rsync \
                --exclude='*DEBUG*' \
                --exclude='*debug*' \
                --include="*${PATTERN}*/" \
                --include="*${PATTERN}*/fold_index=*/" \
                --include="*${PATTERN}*/fold_index=*/.hydra/" \
                --include="*${PATTERN}*/fold_index=*/.hydra/**" \
                --include="*${PATTERN}*/fold_index=*/*.ckpt" \
                --include="*${PATTERN}*/fold_index=*/trial_level_test_results.csv" \
                --exclude='*' \
                --prune-empty-dirs -avzP --chmod=777 \
                $USER@dgx-master.technion.ac.il:/rg/berzak_prj/$USER/eyebench_private/outputs/ outputs/ &
    done
    wait
done
