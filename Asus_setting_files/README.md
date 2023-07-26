# DEMO SHARP industry day

## Setup

N ASUS routers AC86U (IEEE 802.11ac).

1 PC


## Set up the router (this should have already been done: you do not need to re-execute these steps every time)

First, you have to configure the router by the ASUS webpage. 

* Give an IP in the range of 192.168.50.X where X is the IP assigned
by each router.
* Enable SSH
* Put the router in AP mode 
* Update the firmware. The firmware is: ```Asus_setting_files/FW_RT-AC86U_300438215098.w```. To do so, follow the instruction in the [link](https://www.asus.com/support/FAQ/1008000/#a2), follow Method 2: Update Manually

Once these steps are done:
* Copy the ```Asus_setting_files/send_periodically``` folder to the every router inside the ```/jffs/``` folder


## Extracting CSI

To extract CSI, you have to use the files in hadware_scripts. These scripts automitize the extraction based on the scripts inside ```Asus_setting_files/send_periodically```.

To do that run the following commands:
1) Move inside the correct folder
```
cd Asus_setting_files/hardware_scripts/
```
2) Load the dhd.ko module to extract CSi
```
bash reload.sh 
```
3) Configure the TX and RX router. 
```
bash config.sh <channel> <BW[MHz]>
```

These two scripts must be executed one time. Once you do a power cycle of the router you have to run them another time.

To send packets and extract CSI, run this command:
```
bash send_collect_single.sh <folder_name> <packets_per_capture> 
```
e.g., ```bash send_collect_single.sh 072423 1000```
where name is the name of the folder where you want to save the traces, packets means the number of packets to send. It sends a packet every 6ms.

NOTE1: Every bash file is configured with the following login and pass (if needed change the values):
```
# ssh logins
us="admin" 
ps="letmein"
```

NOTE2: These scripts assume that the TX is 192.168.50.1 and 1 RX as 192.168.50.5 (if needed change the values):
```
# rx and tx numbers
tx="1"
rxs="2 3 4 5"
```


## Analyze in real time the CSI with Python 

Activate conda environment (NOTE: this is in Francesca's PC, to be replicated in others if needed):
```
conda activate tfenv
```

Go inside the correct folder
```
cd csiread-master/
```

Run the follwoing mat file changing the name of the folder where the CSI traces are being saved: 
```
python3 -W ignore data_load_single.py folder_name file_name_base n_streams n_cores
```

e.g., ```python3 -W ignore data_load_single.py ../Asus_setting_files/traces/072423 2 1 4```
