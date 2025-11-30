paths=(
"outputs/+data=SBSAT_RC,+model=AhnCNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnCNN_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=AhnRNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnRNN_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=BEyeLSTMArgs,+trainer=TrainerDL,trainer.wandb_job_type=BEyeLSTMArgs_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=PLMASArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASArgs_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=PLMASfArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASfArgs_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_SBSAT_RC"
"outputs/+data=SBSAT_RC,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_SBSAT_RC"
)

# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for path in "${paths[@]}"; do
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py "eval_path=\"${path}\""
done
