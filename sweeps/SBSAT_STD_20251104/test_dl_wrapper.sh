paths=(
"outputs/+data=SBSAT_STD,+model=AhnCNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnCNN_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=AhnRNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnRNN_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=BEyeLSTMArgs,+trainer=TrainerDL,trainer.wandb_job_type=BEyeLSTMArgs_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=PLMASArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASArgs_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=PLMASfArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASfArgs_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_SBSAT_STD"
"outputs/+data=SBSAT_STD,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_SBSAT_STD"
)

# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for path in "${paths[@]}"; do
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py "eval_path=\"${path}\""
done
