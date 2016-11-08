#-*- coding:utf-8 -*-

# 在终端命令 cat README.md | python shell_python.py 10 > sample_output.txt

import sys,random

# 获得参数 (10)
print sys.argv[1]

# 获得从终端的输入 stdin
for line in sys.stdin:
    print line.strip()  # 输出到终端 stdout




