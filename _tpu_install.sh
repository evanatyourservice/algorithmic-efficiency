#!/bin/bash
set -ex
LOGFILE="/setup_log.txt"

{
# python 11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git

# clone algoperf
git clone https://github.com/evanatyourservice/algorithmic-efficiency.git
cd algorithmic-efficiency

# env
mkdir -p /algorithmic-efficiency/venv_py311
python3.11 -m venv /algorithmic-efficiency/venv_py311
source /algorithmic-efficiency/venv_py311/bin/activate

# upgrade pip and install setuptools, wheel
pip install --upgrade pip setuptools wheel

# install algoperf
pip3 install -e '.[pytorch_cpu]'
pip3 install -e '.[jax_tpu]'
pip3 install -e '.[full]'
pip3 install wandb

# mount attached disk to /mnt/disks/persist, create data dir, and change permissions
sudo mkdir -p /mnt/disks/persist && \
sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist && \
sudo mkdir -p /mnt/disks/persist/algoperf_data && \
sudo chown -R $(whoami):$(whoami) /mnt/disks/persist/algoperf_data && \
sudo chmod -R 775 /mnt/disks/persist/algoperf_data

echo "Setup completed successfully!"

} | tee -a "$LOGFILE"
