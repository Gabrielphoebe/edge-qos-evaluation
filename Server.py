import os
import time


server_cmd = "iperf -s"

os.system(server_cmd)
time.sleep(2)
os.system("pkill -f iperf")
