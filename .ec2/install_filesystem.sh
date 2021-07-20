#!/bin/bash

disk_name="/dev/xvdf"

sudo mkfs -t xfs ${disk_name}

echo "Done."