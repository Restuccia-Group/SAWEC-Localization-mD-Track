#!/bin/bash

# ssh logins
us="admin"
pw="letmein"

# rx and tx numbers
tx="1"
rxs="2 3 4 5"

folder_name=$1

pkts=$2

name=$3

if [ "$pkts" = "" ]; then
  echo "Missing number of packets per measurement to send"
  exit 1
fi

# go to the current directory
cd "$(dirname "$0")"

# sleep one second just in case
sleep 1

# create the folder to storage the CSI data
mkdir ../traces/$folder_name

# remove the phase jumps
echo "remove phase jumps"

for rx in $rxs ; do
  sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /jffs/send_periodically/./disable_phase_jumps.sh
done

# collect the CSI data
echo "Collecting"

while true ; do

printf "\n"$name"\n" 

# create the folder to store the csi trace
mkdir ../traces/$folder_name/${name}/

for rx in $rxs ; do
  # sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /jffs/send_periodically/./collectcsi.sh trace.pcap & 
  /usr/lib/x86_64-linux-gnu/wireshark/extcap/sshdump --extcap-interface=sshdump --fifo=../traces/${folder_name}/${name}/${rx}.pcap --capture --remote-host=192.168.50.${rx} --remote-username=${us} --remote-password=${pw} --remote-capture-command='/jffs/send_periodically/tcpdump -i eth6 dst port 5500 -w -' &
done

# sleep just in case
sleep 0.5

# send the # of packets with BW and number of spatial streams 
echo "Sending"
sshpass -p ${pw} ssh ${us}@192.168.50.${tx} /jffs/send_periodically/./send.sh ${pkts} 
sleep 0.5 

echo kill the collection of the routers
echo "Killing"
for rx in $rxs ; do
  sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /usr/bin/killall tcpdump  
done

name=$((name + 1))

done


