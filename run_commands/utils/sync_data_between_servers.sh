# Expected to run from the eyebench_private root directory that has the 'data' folder.
USER=$(whoami)
tasks=(
    'MECOL2_LEX'
    'OneStop_RC' 
    'SBSAT_RC'
    'PoTeC_RC'
    'PoTeC_DE'
    'IITBHGC_CV'
    'CopCo_TYP'
    'SBSAT_STD'
    'CopCo_RCS'
)
models=(
    # 'XGBoostMLArgs'
    'DummyClassifierMLArgs'
    'LogisticRegressionMLArgs'
    'LogisticMeziereArgs'
    'SupportVectorMachineMLArgs'
    'RandomForestMLArgs'
    'SupportVectorRegressorMLArgs'
    'RandomForestRegressorMLArgs'
    'LinearRegressionArgs'
    'LinearMeziereArgs'
    'DummyRegressorMLArgs'
    'AhnRNN'
    'AhnCNN'
    'BEyeLSTMArgs'
    'Roberta'
    'PLMASArgs'
    'PLMASfArgs'
    'RoberteyeWord'
    'RoberteyeFixation'
    'MAG'
    'PostFusion'
)
for task in  "${tasks[@]}"; do
    for model in "${models[@]}"; do
        DATA_PATH="data/cache/features/${task}_${model}"
        echo "Syncing $DATA_PATH to all servers in parallel..."

        # Start all rsync commands in parallel (background)
        # rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@laccl-srv1.dds.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp11.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp12.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp13.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        # rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp14.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        # rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp15.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp16.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp18.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp19.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp20.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@nlp-srv2.iem.technion.ac.il:/data/home/"$USER"/eyebench_private/"$DATA_PATH" &
        rsync --mkpath -avzP --chmod 777 "$DATA_PATH"/ "$USER"@laccl01.dds.technion.ac.il:/data/home/shared/eyebench_private/"$DATA_PATH" &
        
        # Wait for all background rsync jobs to complete before moving to next task/model
        wait
        echo "Completed syncing $DATA_PATH to all servers"
    done
done
