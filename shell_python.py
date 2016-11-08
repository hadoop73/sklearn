#-*- coding:utf-8 -*-

"""
 在终端命令 cat README.md | python shell_python.py 10 > sample_output.txt
 #!/usr/bin/env python                        同样可以在文件首行添加表示执行脚本的解释器,但是文件需要有执行权限
 http://blog.csdn.net/wh_19910525/article/details/8040494                                                          
 http://blog.csdn.net/ritsu_/article/details/12617525                                                      
"""

import sys,random

# 获得参数 (10)
print sys.argv[1]

# 获得从终端的输入 stdin
for line in sys.stdin:
    print line.strip()  # 输出到终端 stdout




