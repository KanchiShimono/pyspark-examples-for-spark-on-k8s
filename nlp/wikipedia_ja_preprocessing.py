# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # PySpark preprocessing for Wikipedia Japanese airticles

# %%
import math
import multiprocessing
import re
import unicodedata
from functools import reduce
from typing import Any, Callable, List, Optional

import MeCab
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import DataFrame as SDF
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

# %%
NUM_CPU = multiprocessing.cpu_count() - 1
WIKI_DATA_URL = 'https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2'
WIKI_JSON_PATH = 's3://bucket/path/to/json/'
WIKI_PARQUET_PATH = 's3://bucket/path/to/parquet/'
MECAB_DICT = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
OUTPUT_PATH = 's3://bucket/path/to/final/output/'

# %% [markdown]
# ## 関数定義

# %%
def compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    for i, func in enumerate(funcs):
        if not callable(func):
            raise TypeError(
                'Position {} argument {} is not callable'.format(i, func)
            )

    return reduce(
        lambda next_func, prev_func: lambda *args, **kwargs: next_func(
            prev_func(*args, **kwargs)
        ),
        funcs,
    )


def tokenize(
    dict_path: Optional[str] = None,
    filter_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> Callable[[str], List[str]]:
    """Tokenize Japanese strings by Mecab

    Args:
        dict_path (Optional[str], optional): Path to mecab dictionary.
            neologd is the one of most famous dictionary for Mecab.
            Defaults to None.
        filter_fn (Optional[Callable[[str], Optional[str]]], optional):
            Filter function applied to string parsed by Mecab.
            Defaults to None.

    Returns:
        Callable[[str], List[str]]: Closure function for parsing Mecab. This is
            useful to specify Mecab dictionary path for PySpark UDF.
    """

    if dict_path is None:
        mecab_arg = '-Ochasen'
    else:
        mecab_arg = '-Ochasen -r /dev/null -d {}'.format(dict_path)

    def _tokenize(sentence: str) -> List[str]:
        tagger = MeCab.Tagger(mecab_arg)
        parsed = tagger.parse(sentence).split('\n')

        tokens = []
        for s in parsed:
            token = filter_fn(s) if filter_fn else s.split('\t')[0]
            if not token:
                continue

            tokens.append(token.strip().replace(' ', '_'))

        return tokens

    return _tokenize


def mecab_filter_fn(analysed_str: str) -> Optional[str]:
    """Filter function applied to string generated by Mecab tagger parse method

    Args:
        analysed_str (str): String generated by Mecab tagger parse method.
            e.g.)
                - 'こんにちは\tコンニチハ\tこんにちは\t感動詞\t\t'
                - 'ぼく\tボク\tぼく\t名詞-代名詞-一般\t\t'
                - 'EOS'
                - ''

    Returns:
        Optional[str]: A word passed filter like 'ぼく'.
    """
    splited = analysed_str.split('\t')

    if len(splited) < 4:
        return None

    pos = splited[3].split('-')
    if (pos[0] == '名詞' and pos[1] != '数') or (pos[0] == '形容詞'):
        return splited[0].strip().replace(' ', '_')

    return None


def normalize(sentence: str) -> str:
    return unicodedata.normalize('NFKC', sentence.rstrip()).lower()


def remove_html_tag(sentence: str) -> str:
    pattern = r'<.*?>'
    return re.sub(pattern, '', sentence)


def remove_url(sentence: str) -> str:
    pattern = r'https?://[0-9a-zA-Z_/:%#\$&\?\(\)~\.=\+\-]+'
    return re.sub(pattern, '', sentence)


def remove_symbol(sentence: str) -> str:
    pattern = r'[!-/:-@\[-`{-~\]：-＠]+'
    return re.sub(pattern, ' ', sentence)


def replace_number(sentence: str, repl: str = '000') -> str:
    pattern = r'(-?[\d]*\.?[\d]+)'
    return re.sub(pattern, repl, sentence)


def remove_price(sentence: str) -> str:
    pattern = r'([\d]+円)|(￥|¥[\d]+)'
    return re.sub(pattern, '', sentence)


def remove_deliv(sentence: str) -> str:
    pattern = r'[\d]+(週間|ヵ月|カ月|か月|ヶ月|日)'
    return re.sub(pattern, '', sentence)


def remove_point(sentence: str) -> str:
    pattern = r'[\d]+倍'
    return re.sub(pattern, '', sentence)


def remove_line_feed(sentence: str) -> str:
    pattern = r'\n|\r|\rn'
    return re.sub(pattern, '', sentence)


@F.udf(returnType=T.StringType())
def cleansing_udf(sentence):
    return compose(
        # replace_number,
        remove_price,
        remove_deliv,
        remove_point,
        remove_symbol,
        normalize,
        remove_line_feed,
        remove_url,
        remove_html_tag,
    )(sentence)


def calc_idf(num_docs: int) -> Callable[[int], float]:
    """Caliculate IDF"""

    def _calc_idf(doc_freq: int) -> float:
        return math.log((num_docs + 1.0) / (doc_freq + 1.0))

    return _calc_idf


def calc_tfidf(
    df: SDF,
    tokens_col: str = 'tokens',
    doc_id_col: str = 'doc_id',
    word_col: str = 'word',
) -> SDF:
    """Caliculate TF-IDF

    Args:
        df (SDF): Spark DataFrame contains following columns.
            - tokens: List of string that be tokenized like
                ['I', 'am', 'from', 'Japan'].
            - doc_id: Document identifier for calculationg IDF.
        tokens_col (str, optional): Column name of tokens.
            Tokens mean list of word.
            Defaults to 'tokens'.
        doc_id_col (str, optional): Column name of doc_id.
            Defaults to 'doc_id'.
        word_col (str, optional): Column name for explode tokens_col.
            Defaults to 'word'.

    Returns:
        SDF: Spark DataFrame contains TF, IDF, TF-IDF, L2 normalized TF-IDF.
    """

    unfolded_word = df.select(
        F.col(doc_id_col),
        F.col(tokens_col),
        F.explode(tokens_col).alias(word_col),
    )

    num_words_in_doc = unfolded_word.groupBy(doc_id_col).agg(
        F.count(tokens_col).alias('num_words_in_doc')
    )

    term_freq = (
        unfolded_word.groupBy(doc_id_col, word_col)
        .agg(F.count(tokens_col).alias('word_count'))
        .join(num_words_in_doc, [doc_id_col], 'inner')
        .withColumn('tf', F.col('word_count') / F.col('num_words_in_doc'))
    )

    doc_freq = unfolded_word.groupBy(word_col).agg(
        F.countDistinct(doc_id_col).alias('df')
    )

    num_docs = unfolded_word.select(doc_id_col).distinct().count()

    calc_idf_udf = F.udf(calc_idf(num_docs), T.DoubleType())

    idf = doc_freq.withColumn('idf', calc_idf_udf(F.col('df')))

    spec_l2_norm = Window.partitionBy(doc_id_col)
    tfidf = (
        term_freq.join(idf, [word_col], 'inner')
        .withColumn('tfidf', F.col('tf') * F.col('idf'))
        .withColumn(
            'l2_norm',
            F.sqrt(F.sum(F.col('tfidf') * F.col('tfidf')).over(spec_l2_norm)),
        )
        .withColumn('normalized_tfidf', F.col('tfidf') / F.col('l2_norm'))
    )

    return tfidf


def _stopwords() -> List[str]:
    alphabet = [chr(i) for i in range(97, 97 + 26)]
    hiragana = [chr(i) for i in range(12353, 12438)]
    katakana = [chr(i) for i in range(12449, 12538)]
    return alphabet + hiragana + katakana


def _get_tfidf_top_n(df: SDF, n: int = 10) -> SDF:
    tfidf_df = calc_tfidf(
        df, tokens_col='removed', doc_id_col='title', word_col='word',
    )

    spec = Window.partitionBy('title').orderBy(F.desc('normalized_tfidf'))

    return (
        tfidf_df.withColumn('rank', F.dense_rank().over(spec))
        .filter(F.col('rank') <= n)
        .groupby('title')
        .agg(F.collect_list('word').alias('tfidf_ranked_text'))
    )

# %% [markdown]
# ## Wikipediaデータ準備

# %%
! git clone https://github.com/attardi/wikiextractor
! wget $WIKI_DATA_URL

# %%
! python ./wikiextractor/WikiExtractor.py -b 32M --processes $NUM_CPU -o data/json --json jawiki-latest-pages-articles.xml.bz2
! aws s3 cp ./data/json/ $WIKI_JSON_PATH --recursive

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
    .config('spark.driver.host', 'bxk70o2.notebook.svc.cluster.local')
    .config('spark.driver.port', 29413)
    .config('spark.driver.memory', '3G')
    .config('spark.driver.cores', 1)
    .config('spark.executor.memory', '4G')
    .config('spark.executor.cores', 1)
    .config('spark.default.parallelism', 10)
    .config('spark.sql.shuffle.partitions', 10)
    .config('spark.eventLog.compress', True)
    .config('spark.eventLog.enabled', True)
    .config('spark.eventLog.dir', 'file:///tmp/spark-events')
    .getOrCreate()
)

# %%
schema = T.StructType(
    T.StructField('id', T.LongType()),
    T.StructField('url', T.StringType()),
    T.StructField('title', T.StringType()),
    T.StructField('text', T.StringType()),
)

raw_df = (
    spark
    .read
    .option('mode', 'FAILFAST')
    .schema(schema)
    .json(WIKI_JSON_PATH)
    .select('id', 'url', 'title', 'text')
    .repartition(int(spark.conf.get('spark.sql.shuffle.partitions')))
)

# %%
raw_df.write.parquet(WIKI_PARQUET_PATH)

# %% [markdown]
# ## PySparkを使ったWikipediaデータ処理

# %%
# Read dataframe of wikipedia articles
wikipedia_df = spark.read.parquet(WIKI_PARQUET_PATH)

# %%
# Cleansing text
cleansed_df = wikipedia_df.withColumn(
    'cleansed', cleansing_udf(F.col('text'))
)

# %%
# Tokenize (split sentence to list of word)
tokenize_udf = F.udf(
    tokenize(MECAB_DICT, mecab_filter_fn), T.ArrayType(T.StringType())
)
tokenized_df = cleansed_df.withColumn(
    'tokenized', tokenize_udf(F.col('cleansed'))
).drop('cleansed')
tokenized_df.persist()

# %%
stopwords = _stopwords()
remover = StopWordsRemover(
    inputCol='tokenized', outputCol='removed', stopWords=stopwords
)
removed_stopwords_df = remover.transform(tokenized_df).drop('tokenized')
removed_stopwords_df.persist()

# %%
# Calculate TF-IDF. This will run shuffle operation.
tfidf = _get_tfidf_top_n(removed_stopwords_df, n=20)
tfidf.write.parquet(OUTPUT_PATH)

# %%
# wrap up
tokenized_df.unpersist()
removed_stopwords_df.unpersist()

spark.stop()
