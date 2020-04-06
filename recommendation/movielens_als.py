# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # PySpark MovieLens Recommendation by ALS

# %%
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame as SDF
from pyspark.sql import Row, SparkSession

# %% [markdown]
# ## Utility関数定義

# %%
def parse_ratings(path: str) -> SDF:
    lines = spark.read.text(path).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    rdd = (
        parts
        .map(
            lambda p: Row(
                userId=int(p[0]),
                movieId=int(p[1]),
                rating=float(p[2]),
                timestamp=int(p[3])))
    )
    return spark.createDataFrame(rdd)


def parse_movies(path: str) -> SDF:
    lines = spark.read.text(path).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    rdd = (
        parts
        .map(
            lambda p: Row(
                movieId=int(p[0]),
                title=str(p[1]),
                genre=str(p[2])))
    )
    return spark.createDataFrame(rdd)


def pd_parse_ratings(path: str) -> SDF:
    return spark.createDataFrame(
        pd.read_csv(
            path,
            engine='python',
            sep='::',
            header=None,
            names=['userId', 'movieId', 'rating', 'timestamp']))


def pd_parse_movies(path: str) -> SDF:
    return spark.createDataFrame(
        pd.read_csv(
            path,
            engine='python',
            sep='::',
            header=None,
            names=['movieId', 'title', 'genre']))


# %%
spark = (
    SparkSession
    .builder
    .master('k8s://https://kubernetes.default.svc.cluster.local:443')
    .appName('spark_on_k8s')
    .config('spark.kubernetes.container.image', 'kanchishimono/pyspark-worker:latest')
    .config('spark.kubernetes.pyspark.pythonVersion', 3)
    .config('spark.executor.instances', 2)
    .config('spark.kubernetes.namespace', 'notebook')
    .config('spark.port.maxRetries', 3)
    .config('spark.history.ui.port', True)
    .config('spark.ui.enabled', True)
    .config('spark.ui.port', 4040)
    .config('spark.driver.host', 'yhdqer9.notebook.svc.cluster.local')
    .config('spark.driver.port', 29413)
    .config('spark.executor.memory', '3G')
    .config('spark.executor.cores', 1)
    .config('spark.default.parallelism', 10)
    .config('spark.sql.shuffle.partitions', 10)
    .config('spark.eventLog.compress', True)
    .config('spark.eventLog.enabled', True)
    .config('spark.eventLog.dir', 'file:///tmp/spark-events')
    .getOrCreate()
)


# %%
spark

# %% [markdown]
# ## MovieLensデータセット読み込み

# %%
ratings_df = pd_parse_ratings('/home/work/ml-1m/ratings.dat').repartition(10)
movies_df = pd_parse_movies('/home/work/ml-1m/movies.dat').repartition(10)


# %%
ratings_df.toPandas()


# %%
movies_df.toPandas()


# %%
train_df, test_df = ratings_df.randomSplit([0.6, 0.4], seed=12345)
train_df.persist()

# %% [markdown]
# ## レコメンドモデル (ALS) 定義
# ### パラメーター
#
# |パラメータ名|説明|
# |:--|:--|
# |userCol|ユーザーIDが記録されているカラム名|
# |itemCol|アイテムIDが記録されているカラム名|
# |ratingCol|ユーザーのアイテムに対する評価値が記録されているカラム名。レビュー値のように明示的なものと、アクセス回数など暗黙的なものどちらかを使用する。|
# |coldStartStrategy|訓練データに含まれない未知のユーザーやアイテムの取り扱い方法。'nan'は未知のIDに対する推論値をnanで返す。'drop'は未知のIDが含まれる行を落とす。|
# |numUserBlocks|ユーザーの潜在因子行列のDataFrameパーティション数|
# |numItemBlocks|アイテムの潜在因子行列のDataFrameパーティション数|
# |implicitPrefs|ratingColに暗黙的な値を使用するか。Trueの場合、暗黙的評価値用の計算式が内部でしようされる。|
# |nonnegative|ratingColに含まれる値が非負値か。|
# |maxIter|収束計算の繰り返し回数|
# |rank|ユーザー、アイテムの潜在因子行列の次元数|
# |alpha|implicitPrefs=Trueの時のみ有効。暗黙的評価値の信頼度の高さを示す値。|
# |regParam|正則化パラメータ|
#

# %%
als = ALS(
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    coldStartStrategy='drop',
    numUserBlocks=2,
    numItemBlocks=2,
    implicitPrefs=False,
    nonnegative=True)

# %% [markdown]
# ## レコメンドモデル計算
# ### グリッドサーチパラメータ定義

# %%
param_grid = (
    ParamGridBuilder()
    .addGrid(als.maxIter, [5, 10, 15])
    .addGrid(als.rank, [10, 15, 20])
    # alpha enabled only when implicitPrefs is True
    # .addGrid(als.alpha, [1.0, 10.0, 100.0])
    .addGrid(als.regParam, [0.01, 0.05, 0.1, 0.5, 1.0])
    .build()
)

# %% [markdown]
# ### 評価指標選択 (RMSE)

# %%
evaluator = RegressionEvaluator(
    metricName='rmse',
    labelCol='rating',
    predictionCol='prediction')

# %% [markdown]
# ### グリッドサーチ実行

# %%
tsv = TrainValidationSplit(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8,
    collectSubModels=True)


# %%
models = tsv.fit(train_df)


# %%
best_model = models.bestModel

# %% [markdown]
# ## モデル評価

# %%
prediction = best_model.transform(test_df)
rmse = evaluator.evaluate(prediction)
print('RMSE: {:.3f}'.format(rmse))


# %%
metric_param_sets = [(m, p) for m, p in zip(models.validationMetrics, param_grid)]

# %% [markdown]
# ## レコメンド結果確認
# ### 確認対象ユーザー選択

# %%
users = (
    ratings_df
    .select('userId')
    .distinct()
    .filter(F.col('userId') == 55)
)


# %%
users.toPandas()

# %% [markdown]
# ### 確認対象ユーザー評価履歴確認

# %%
(
    ratings_df
    .join(users, ['userId'], 'inner')
    .join(movies_df, ['movieId'], 'inner')
    .orderBy('userId', F.desc('rating'))
    .toPandas()
)

# %% [markdown]
# ### レコメンド結果確認

# %%
(
    best_model
    .recommendForUserSubset(users, 10)
    .withColumn('temp', F.explode('recommendations'))
    .select(
        'userId',
        F.col('temp').getItem('movieId').alias('movieId'),
        F.col('temp').getItem('rating').alias('rating')
    )
    .join(movies_df, ['movieId'], 'inner')
    .orderBy(F.desc('rating'))
    .show(truncate=False)
)


# %%



# %%
spark.stop()


# %%
