#!/bin/bash
# number of spatial stream

# ssh logins
us="admin" # change it
ps="letmein" # change it

# rx and tx numbers
tx="1"
rxs="2 3 4 5"

channel=$1
BW=$2

echo "Setting up the transmitter"
sshpass -p ${ps} ssh ${us}@192.168.50.${tx} /jffs/send_periodically/./config.sh ${channel} ${BW}

echo "Setting up the receiver"

for rx in $rxs ; do
  sshpass -p ${ps} ssh ${us}@192.168.50.${rx} /jffs/send_periodically/config_rx.sh ${channel} ${BW}
done


