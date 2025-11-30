paths=(
"outputs/+data=PoTeC_DE,+model=AhnCNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnCNN_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=AhnRNN,+trainer=TrainerDL,trainer.wandb_job_type=AhnRNN_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=BEyeLSTMArgs,+trainer=TrainerDL,trainer.wandb_job_type=BEyeLSTMArgs_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=MAG,+trainer=TrainerDL,trainer.wandb_job_type=MAG_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=PLMASArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASArgs_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=PLMASfArgs,+trainer=TrainerDL,trainer.wandb_job_type=PLMASfArgs_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=PostFusion,+trainer=TrainerDL,trainer.wandb_job_type=PostFusion_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=Roberta,+trainer=TrainerDL,trainer.wandb_job_type=Roberta_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=RoberteyeFixation,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeFixation_PoTeC_DE"
"outputs/+data=PoTeC_DE,+model=RoberteyeWord,+trainer=TrainerDL,trainer.wandb_job_type=RoberteyeWord_PoTeC_DE"
)

# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for path in "${paths[@]}"; do
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py "eval_path=\"${path}\""
done
