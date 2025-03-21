#!/bin/bash
set -ex

# all users are given some safe sudo access on TPU VMs, so let's just let this run as root

LOGFILE="/setup_log.txt"

{
# python 11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git

# clone algoperf
git clone https://github.com/evanatyourservice/algorithmic-efficiency.git

# env
mkdir -p /algorithmic-efficiency/venv_py311
python3.11 -m venv /algorithmic-efficiency/venv_py311
source /algorithmic-efficiency/venv_py311/bin/activate

# upgrade pip and install setuptools, wheel
pip install --upgrade pip setuptools wheel

# install algoperf
cd algorithmic-efficiency
pip3 install -e '.[pytorch_cpu]'
pip3 install -e '.[jax_tpu]'
pip3 install -e '.[full]'
pip3 install wandb

# mount attached disk (read-only)
sudo mkdir -p /mnt/disks/persist
sudo umount /mnt/disks/persist || true
sudo fsck -n /dev/sdb
sudo mount -t ext4 -o ro,noload /dev/sdb /mnt/disks/persist
if [ -d /mnt/disks/persist/algoperf_data ]; then
  echo "Directory /mnt/disks/persist/algoperf_data exists. Contents:"
  ls -l /mnt/disks/persist/algoperf_data
else
  echo "Directory /mnt/disks/persist/algoperf_data does not exist."
fi

echo "Setup completed successfully!"

} | tee -a "$LOGFILE"
