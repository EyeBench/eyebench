# DGX Commands

## interactive terminal

For running commands without changes:
```bash
srun --pty --partition=work,mig --nodes=1 --ntasks=1 --gpus=0 --mem=40G --cpus-per-task=3 --container-mounts="/rg/berzak_prj/shubi:/home/shubi" --container-image="/rg/berzak_prj/shubi/prj/rev05_pytorchlightning+pytorch_lightning.sqsh" --container-workdir="/home/shubi/eyebench_private" bash -i
```

With changes:
```bash
srun --pty --partition=work,mig --nodes=1 --ntasks=1 --gpus=1 --mem=40G --cpus-per-task=3 --container-mounts="/rg/berzak_prj/shubi:/home/shubi" --container-image="/rg/berzak_prj/shubi/prj/rev03_pytorchlightning+pytorch_lightning.sqsh" --container-workdir="/home/shubi/eyebench_private" --container-save="/rg/berzak_prj/shubi/prj/rev05_pytorchlightning+pytorch_lightning.sqsh" bash -i
```

Then `git pull`, `mamba env update -f environment.yml`
finally exit the interactive session to save your changes.
Update rev (in container-save and in actual scripts) if making changes!

