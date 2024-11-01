#!/bin/sh


cd "$(dirname "$0")"

channel=$1
BW=$2

echo $PATH
PATH=/bin:/usr/bin:/sbin:/usr/sbin:/home/imdea:/mmc/sbin:/mmc/bin:/mmc/usr/sbin:/mmc/usr/bin:/opt/sbin:/opt/bin:/opt/usr/sbin:/opt/usr/bin
echo $PATH

# equivalent to configcsiblablabla.sh
/usr/sbin/wl -i eth6 up
/usr/sbin/wl -i eth6 radio on
/usr/sbin/wl -i eth6 country UG
/usr/sbin/wl -i eth6 chanspec ${channel}/${BW}
/usr/sbin/wl -i eth6 monitor 1
/sbin/ifconfig eth6 up

# generate packet
NODEID=$(ifconfig br0 | grep inet | awk '{ print $2 }' | awk -F"." '{ print $4 }')
./setpackets $NODEID
dd if=/dev/zero bs=4 count=1 of=/tmp/4zeroes
cat /tmp/4zeroes packetnode1x1.dat > packetnode1x1BP.dat

CONFIG=$(./makecsiparams -e 1 -m ff:ff:00:12:34:56 -c 0xe29b -C 0xf -N 0x1 -b 0x88)
LEN=34
./nexutil -I eth6 -s500 -b -l${LEN} -v${CONFIG} 

# equivalent to ./setdumpparameters 2 0
wl -i eth6 shmem 0x172a 2
wl -i eth6 shmem 0x172c 0

# set txcore
wl -i eth6 txcore -s 1 -c 1 -o 1 -k 1

# equivalent to setrxcore.sh
./nexutil -Ieth6 -s528 -i -v15 -l 2

# now reset channel
# wl -i eth6 chanspec 157/$BW

# equivalent to restarthfc.sh
wl -i eth6 shmem 0x1776 0x9000

# pace frames every 4ms
wl -i eth6 shmem 0x177e 0xf000

