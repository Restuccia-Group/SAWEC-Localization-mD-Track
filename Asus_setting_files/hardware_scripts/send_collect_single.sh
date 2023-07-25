#!/bin/bash


# trap "for rx in $rxs ; do sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /usr/bin/killall tcpdump done" EXIT

# ssh logins
us="admin"
pw="letmein"

# rx and tx numbers
tx="1"
rxs="2 3 4 5"

folder_name=$1

pkts=$2

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

# create the folder to store the csi trace

#for rx in $rxs ; do
#  mkdir ../traces/$folder_name/${rx}/
#done

for rx in $rxs ; do
  # sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /jffs/send_periodically/./collectcsi.sh trace.pcap & 
  /usr/lib/x86_64-linux-gnu/wireshark/extcap/sshdump --extcap-interface=sshdump --fifo=../traces/${folder_name}/${rx}.pcap --capture --remote-host=192.168.50.${rx} --remote-username=${us} --remote-password=${pw} --remote-capture-command='/jffs/send_periodically/tcpdump -w - -i eth6 dst port 5500' &
  # tcpdump -i enx14ebb6850787 -r - -w ../traces/${folder_name}/${rx}/trace.pcap dst port 22 | sshpass -p ${pw} ssh ${us}@192.168.50.${rx} /jffs/tcpdump -w - -C 1 -W 10 -i eth6 dst port 5500 &
done

# sleep just in case
sleep 1

# send the # of packets with BW and number of spatial streams 
echo "Sending"
sshpass -p ${pw} ssh ${us}@192.168.50.${tx} /jffs/send_periodically/./send.sh ${pkts} 
sleep 0.5 

