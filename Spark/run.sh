export HADOOP_CONF_DIR=/home/hadoop/hadoop-2.7.3/etc/hadoop
/home/hadoop/spark-2.0.1-bin-hadoop2.7/bin/spark-submit \
            --master yarn \
            --num-executors 8 \
            --executor-memory 1g \
            slope_one.py
