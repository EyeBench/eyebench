#!/bin/bash

# Example usage:
# bash run_commands/utils/gpu_servers_status.sh "nvidia-smi"
# bash run_commands/utils/gpu_servers_status.sh "~/miniforge3/bin/gpustat"

# Check if command is provided
if [ -z "$1" ]; then
    echo "No command provided. Usage: ./gpu_servers_status.sh <command>"
    exit 1
fi

command_to_run=$1
user=$(whoami)

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

for server in "${servers[@]}"; do
    echo "$server:"
    (ssh "$user@$server" "$command_to_run" &)
    sleep 1
done
wait
