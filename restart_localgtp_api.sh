#!/bin/bash

echo "Process found:"
pgrep -u ec2-user -af "python3 run_localGPT_API.py" 

echo "Do you want to restart it? (y/n)"
read -r answer

if [[ "$answer" == "y" ]]; then
  # Do something
  echo "Killing the process"
  pkill -u ec2-user -f "python3 run_localGPT_API.py"
  echo "Restarting the process"
  cd ~/TEMP/GIT/localGPT
  nohup python3 run_localGPT_API.py &
else
  echo "Nothing was done."
  exit 1
fi
