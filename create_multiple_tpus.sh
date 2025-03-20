#!/bin/bash

# runs create_tpu.sh multiple times naming them node-3, node-4, etc. starting from start_num

# Usage: ./create_multiple_tpus.sh <num_tpus> <data_disk_name> <start_num>
# Example: ./create_multiple_tpus.sh 5 dev-evan-1 2

if [ $# -ne 3 ]; then
  echo "Usage: $0 <num_tpus> <data_disk_name> <start_num>"
  exit 1
fi

num_tpus=$1
data_disk_name=$2
start_num=$3

for i in $(seq 1 $num_tpus); do
  node_num=$((start_num + i - 1))
  node_name="node-$node_num"
  echo "Creating TPU $node_name with data disk $data_disk_name..."
  ./create_tpu.sh -n "$node_name" -d "$data_disk_name"
done

echo "Finished creating TPUs."
