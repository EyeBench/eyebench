paths=(
"outputs/+data=MECOL2_LEX,+model=AhnCNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnCNN_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=AhnRNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnRNN_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=BEyeLSTMArgs,+trainer=TrainerDL,trainer.wandb_job_type=BEyeLSTMArgs_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=PLMASArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASArgs_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=PLMASfArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASfArgs_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_MECOL2_LEX"
"outputs/+data=MECOL2_LEX,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_MECOL2_LEX"
)

# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for path in "${paths[@]}"; do
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py "eval_path=\"${path}\""
done
