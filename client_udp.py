import os

lengths = [4, 8, 16, 32, 64]
parallels = [1, 2, 4, 8, 16]

for length in lengths:
    for parallel in parallels:
        tcpdump_file = f"udp_l{length}_p{parallel}.txt"
        
        tcpdump_cmd = f"tcpdump -i wlan0 host 192.168.1.5 -s 0 -tttt -n -q -c 500 | awk '{{print $1, $2, $NF}}' > {tcpdump_file} 2>/dev/null &"
        os.system(tcpdump_cmd)
        
        iperf_cmd = f"iperf -c 192.168.1.6 -l {length} -P {parallel} -u"
        os.system(iperf_cmd)
        
        os.system("pkill -f tcpdump") 
