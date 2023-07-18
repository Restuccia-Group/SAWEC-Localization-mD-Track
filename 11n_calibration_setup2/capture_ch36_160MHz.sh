#!/bin/bash -e

for phy_id in $@
do
    sudo array_prepare_for_picoscenes $phy_id "5180 HT20"
done
sleep 5

PicoScenes "-d debug; 
	-i $1 --preset RX_CBW_20 --rxcm 1 --output $1 --mode logger;
	#-i $2 --preset RX_CBW_20 --output $2 --mode logger;
        #-i $3 --preset RX_CBW_20 --output $3 --mode logger"
        #-i $4 --preset RX_CBW_160 --source-address-filter 00:16:ea:12:34:56 --output $4 --mode logger;
        #-i $5 --preset RX_CBW_160 --source-address-filter 00:16:ea:12:34:56 --output $5 --mode logger;
        #-i $6 --preset RX_CBW_160 --source-address-filter 00:16:ea:12:34:56 --output $6 --mode logger;
        #-i $7 --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output $7 --mode logger;
        #-i $8 --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output $8 --mode logger;
        #-i $9 --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output $9 --mode logger;
        #-i ${10} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${10} --mode logger;
        #-i ${11} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${11} --mode logger;
        #-i ${12} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${12} --mode logger;
        #-i ${13} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${13} --mode logger;
        #-i ${14} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${14} --mode logger;
        #-i $4 --preset TX_CBW_160_HESU --txcm 1 --repeat 100000 --delay 5e3 --mode injector"
        #-i $2 --preset TX_CBW_160_HESU --txcm 1 --repeat 100000 --delay 10e3 --mode injector"
        #-i ${16} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${15} --mode logger;
        #-i ${17} --preset RX_CBW_160 --rxcm 1 --source-address-filter 00:16:ea:12:34:56 --output ${16} --mode logger;
        #-i ${17} --preset RX_CBW_160 --source-address-filter 00:16:ea:12:34:56 --output ${17} --mode logger;
        #-i ${18} --preset RX_CBW_160 --source-address-filter 00:16:ea:12:34:56 --output ${18} --mode logger;
