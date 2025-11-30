paths=(
"outputs/+data=CopCo_RCS,+model=AhnCNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnCNN_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=AhnRNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnRNN_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=BEyeLSTMArgs,+trainer=TrainerDL,trainer.wandb_job_type=BEyeLSTMArgs_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=PLMASArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASArgs_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=PLMASfArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASfArgs_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_CopCo_RCS"
"outputs/+data=CopCo_RCS,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_CopCo_RCS"
)

# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for path in "${paths[@]}"; do
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py "eval_path=\"${path}\""
done
