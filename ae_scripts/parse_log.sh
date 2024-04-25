#!/bin/bash

log_path=$1
type=$2

if [ "$type" == "fig2b" ]; then
    cat $log_path | grep wait_time | awk '{
      wait_sum[$(NF-10)] += $(NF-2);  # 按第一列的值进行累加
      exec_sum[$(NF-10)] += $(NF-4);
      count[$(NF-10)]++;   # 统计每组的数量
    }
    END {
      for (key in wait_sum) {
        wait_avg = wait_sum[key] / count[key];  # 计算平均值
        exec_avg = exec_sum[key] / count[key];
        print key, wait_avg, exec_avg;               # 输出每组的平均值
      }
    }'
else

    cat $log_path | grep wait_time | awk '{ sum += $NF; n++ } END { if (n > 0) print "avg queueing time:" sum/n; }'

    cat $log_path | grep 'adjust time' | awk 'BEGIN{max=0} {max = $NF > max ? $NF : max} END {print "max adjust time:" max}'

    cat $log_path | grep 'Solve time' | cut -d ' ' -f 3  | cut -d ',' -f 1 | awk '{ sum += $1; n++ } END { if (n > 0) print "solve time:" sum; }'

fi

