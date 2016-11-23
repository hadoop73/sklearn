# coding:utf-8


from pyspark import SparkContext




if __name__ == "__main__":

    sc = SparkContext(appName="MovieFeature")
    user_data = sc.textFile("../../ml-100k/u.user")
    print user_data.first()








