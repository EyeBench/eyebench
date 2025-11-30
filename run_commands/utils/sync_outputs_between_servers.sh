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

servers=(
    'nlp11.iem.technion.ac.il'
    'nlp12.iem.technion.ac.il'
    'nlp13.iem.technion.ac.il'
    'nlp16.iem.technion.ac.il'
    'nlp18.iem.technion.ac.il'
    'nlp19.iem.technion.ac.il'
    'nlp20.iem.technion.ac.il'
    'nlp-srv2.iem.technion.ac.il'
    'laccl01.dds.technion.ac.il'
    'laccl-srv1.dds.technion.ac.il'
)

# Run cleanup on all servers
for server in "${servers[@]}"; do
    echo "Running cleanup_models.py on $server..."
    ssh $USER@$server "cd /data/home/$USER/eyebench_private && python src/run/multi_run/cleanup_models.py --real_run --keep_one_lowest"
done

# Sync outputs for specific task-model combinations
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        # Pattern to match: +data=TASK,+model=MODEL,+trainer=...
        PATTERN="+data=${task},+model=${model}"
        echo "Syncing outputs for $task - $model"
        
        for server in "${servers[@]}"; do
            echo "  From $server..."
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
                  --prune-empty-dirs --mkpath -avzP --chmod=777 \
                  $USER@$server:/data/home/$USER/eyebench_private/outputs/ outputs/
        done
    done
done

