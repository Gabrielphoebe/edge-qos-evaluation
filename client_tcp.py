import os
import time

lengths = [4, 8, 16, 32, 64] ## packet size
parallels = [1, 2, 4, 8, 16] ## number of connection or bandwidth
windows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]##window size6

for length in lengths:
    for parallel in parallels:
        for window in windows:
            
            tcpdump_file = f"tcp_l{length}_p{parallel}_w{window}.txt"
            
            tcpdump_cmd = f"tcpdump -i wlan0 host 192.168.1.5 -s 0 -tttt -n -q -c 500 | awk '{{print $1, $2, $NF}}' > {tcpdump_file} 2>/dev/null &"
            os.system(tcpdump_cmd)
            
            iperf_cmd = f"iperf -c 192.168.1.6 -l {length} -P {parallel} -w {window}"
            os.system(iperf_cmd)
            
            os.system("pkill -f tcpdump")
