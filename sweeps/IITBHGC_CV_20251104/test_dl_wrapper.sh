paths=(
"outputs/+data=IITBHGC_CV,+model=AhnCNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnCNN_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=AhnRNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnRNN_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=BEyeLSTMArgs,+trainer=TrainerDL,trainer.wandb_job_type=BEyeLSTMArgs_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=PLMASArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASArgs_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=PLMASfArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASfArgs_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_IITBHGC_CV"
"outputs/+data=IITBHGC_CV,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_IITBHGC_CV"
)

# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for path in "${paths[@]}"; do
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py "eval_path=\"${path}\""
done
