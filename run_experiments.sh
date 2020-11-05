~/miniconda3/bin/python3 GDA.py --gpus 1 &
sleep 5

~/miniconda3/bin/python3 SGDA.py --gpus 1 &
sleep 5

~/miniconda3/bin/python3 DoubleSmoothedGDA.py --gpus 1 &