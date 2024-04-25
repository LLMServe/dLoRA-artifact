#!/bin/bash

log_path=$1
output_path=$2

queueing_time=$(cat $log_path | grep wait_time | cut -d ' ' -f 12 | sort -n)
cat $log_path | grep wait_time | cut -d ' ' -f 14 | sort -n | awk 'BEGIN{count=0} {arr[count]=$1; count+=1} END{print "P10:" arr[int(0.1*count)]; print "P20:" arr[int(0.2*count)]; print "P30:" arr[int(0.3*count)]; print "P40:" arr[int(0.4*count)]; print "P50:" arr[int(0.5*count)]; print "P60:" arr[int(0.6*count)]; print "P70:" arr[int(0.7*count)]; print "P80:" arr[int(0.8*count)]; print "P90:" arr[int(0.9*count)]; print "P95:" arr[int(0.95*count)]; print "P99:" arr[int(0.99*count)]}'

cat $log_path | grep wait_time | cut -d ' ' -f 14 | awk '{ sum += $1; n++ } END { if (n > 0) print "avg waiting: " sum/n; }'

cat $log_path | grep 'Solve time' | cut -d ' ' -f 3  | cut -d ',' -f 1 | awk '{ sum += $1; n++ } END { if (n > 0) print "solve time: " sum; }'

cat $log_path | grep 'adjust time' | awk '{max = $NF > max ? $NF : max} END {print "max adjust time " max}'

cat $log_path | grep wait_time | cut -d ' ' -f 10 | sort -n | awk 'BEGIN{count=0} {arr[count]=$1; count+=1} END{print "P90:" arr[int(0.9*count)]; print "P95:" arr[int(0.95*count)]; print "P99:" arr[int(0.99*count)]}'