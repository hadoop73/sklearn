




[Python中的logging模块                   ](http://python.jobbole.com/86887/?utm_source=blog.jobbole.com&utm_medium=relatedPosts                                                                                      )


```python
# 引入模块
import logging
import sys


# 获得 logger 实例,参数为空则获得 root logger

logger = logging.getLogger('AppName')

# 指定 logger 输出格式
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

# 文件日志
file_handler = logging.FileHandler("test.log")
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式                                                                                                                        

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter

# 为 logger 添加日志处理器
logger.addHandler(console_handler)


# 指定日志最低输出级别,默认为 WARN 级别
logger.setLevel(logging.INFO)

# 输出 log
logger.info('this is debug info')

# 2016-10-08 21:59:19,493 INFO    : this is information                                                         
```


