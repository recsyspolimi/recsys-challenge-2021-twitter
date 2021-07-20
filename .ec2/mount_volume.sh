#!/bin/bash
# THIS SCRIPT ASSUMES THE
# DISK HAS A FILESYSTEM

mount_dir="/home/ubuntu/data"
# disk_name="/dev/xvdf"

if [[ -z "${DISK}" ]]; then
  echo "no DISK env variable found"
else
  disk_name="${DISK}"
  # show info about the disk
  sudo file -s ${disk_name}

  # directory where the disk will be mounted
  sudo mkdir ${mount_dir}

  # copy original fstab for backup
  sudo cp /etc/fstab /etc/fstab.orig

  # get disk UUID
  uuid=$(lsblk -no UUID ${disk_name})  # should work like this, without 'sudo'
  echo "UUID for ${disk_name} is ${uuid}"

  # append UUID to fstab to automatically mount at boot
  sudo echo "UUID=${uuid}  ${mount_dir}  xfs  defaults,nofail  0  2" | sudo tee -a /etc/fstab

  # mount all
  sudo mount -a

  echo "Done."
fi


