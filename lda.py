import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import functions as f
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
  

bucket = '<your bucket name>'
prefix = '<your prefix>'
dir_path = 's3://{}/{}'.format(bucket, prefix)
output_dir = dir_path + '/output/lda-reduction'

target_column = '<taret feature for LDA>'

# params for LDA
k = 10
maxIter = 100
seed = 1
optimizer = 'em'

def to_array(col):
    """ Split a vector to columns in DataFrame
    """
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)
  
## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

glueContext = GlueContext(SparkContext.getOrCreate())

schema = StructType([
    StructField('category_id', IntegerType(), False),
    StructField('sum_value', IntegerType(), False),
    StructField('userid', IntegerType(), False),
])

dyf = glueContext.create_dynamic_frame_from_options(
    's3', 
    {'paths': [dir_path]}, 
    format='csv',
    format_options={'withHeader':True})

# DynamicFrame から DataFrame への変換
df = dyf.toDF()
df = df.withColumn(target_column, f.col(target_column).cast(IntegerType()))
df = df.withColumn("sum_value", f.col("sum_value").cast(IntegerType()))
df = df.withColumn("userid", f.col("userid").cast(IntegerType()))

print("size of dataframe")
print((df.count(), len(df.columns)))

# category_id で横持ちへpivot
df2 = df.groupby('userid').pivot(target_column).sum('sum_value').fillna(0)
print("pivot")
print((df2.count(), len(df2.columns)))

# ベクトル変量への変換
assembler = VectorAssembler(inputCols=df2.columns[1:], outputCol="features")
feature_vectors = assembler.transform(df2)
df3 = feature_vectors[['userid', 'features']]
print('feature vectors')
print((df3.count(), len(df3.columns)))

# LDA で次元圧縮
# Trains a LDA model.
model = LDA(k=k, maxIter=maxIter, seed=seed, optimizer=optimizer).fit(df3)
score = model.transform(df3)
df4 = score['userid','topicDistribution']
df5 = df4.withColumn(
    'lda_feature',
    to_array(col('topicDistribution'))).select(['userid'] + [col('lda_feature')[i] for i in range(k)])
df5.show()
print('LDA features')
print((df5.count(), len(df5.columns)))

# 保存
df5.write.mode('overwite').csv(output_dir, header = 'true')