#!/bin/sh

cd "$(dirname "$0")"

packets=$1

if [ "$packets" = "" ]; then
  echo "Missing the packets"
  exit 1
fi


./rawperf -i eth6 -n ${packets} -f packetnode1x1BP.dat -t 6000 -q 0 

# -i    name of interface
# -f    filename with frame data
# -n    number of frames to send
# -t    delay in us between frames
# -q    number of frames before sending full matrix
# -h    prints this message

