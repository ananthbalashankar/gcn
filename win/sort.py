from __future__ import print_function

import sys

from pyspark import SparkContext


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit sort <inputfile> <outputfile>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonSort")

    lines = sc.textFile(sys.argv[1], 1)
    words = lines.flatMap(lambda x: x.split(' '))
    counts = words.map(lambda x: (str(x), 1))

    sortedData = counts.sortByKey()
    sortedData.saveAsTextFile(sys.argv[2])

    sc.stop()
