# Expected to run from the eyebench_private root directory that has the 'data' folder.
USER=$(whoami)
DATA_PATH="data"
rsync -avzP --chmod 777 "$DATA_PATH"/ "$USER"@dgx-master.technion.ac.il:/rg/berzak_prj/"$USER"/eyebench_private/"$DATA_PATH"