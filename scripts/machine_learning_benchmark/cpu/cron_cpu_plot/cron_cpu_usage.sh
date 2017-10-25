#!/bin/bash
while :
do
    # gpu: date memory.MiB
    echo `date +%s` `nvidia-smi | sed -n '9p' | awk '{print $9}'` `tail -1 /home/xzhang1/run/master/PbS_QD/bare_qd_testing/Q0_Test_convergence/Pb55S38/static0/run.log 2>/dev/null` >> gpu.log
    # cpu: date memory(mb) stage(marked by vasp output, i.e. n de: or dmm 1:)
    # echo `date +%s` `free -m | sed -n '2p' | awk '{print $3}'` `tail -1 /home/xzhang1/run/master/PbS_QD/bare_qd_testing/Q0_Test_convergence/Pb55S38/static0/run.log 2>/dev/null` >> gpu.log
    sleep 5
done
