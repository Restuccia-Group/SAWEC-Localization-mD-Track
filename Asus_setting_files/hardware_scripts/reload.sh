#!/bin/bash

# ssh logins
us="admin"
pw="letmein"

tx="1"
rxs="2 3 4 5"

echo "Reloading the transmitter"
sshpass -p ${pw} ssh ${us}@192.168.50.${tx} /jffs/send_periodically/./reload.sh


echo "Reloading the receivers"
for rx in $rxs ; do
  sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /jffs/send_periodically/./reload.sh
done

