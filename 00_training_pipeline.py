from luigi.contrib.spark import PySparkTask
from luigi.parameter import IntParameter
from luigi import LocalTarget, Task


class Download(Task):
    import praw
    
    # Die Version des Models
    version = IntParameter(default=1)
    
    # Es werden maximal 500 Posts pro Klasse geladen
    limit = IntParameter(default=500)
    
    # Definition der Subreddits mit der der DecisionTree trainiert wird
    subreddits = ["datascience", "gameofthrones"]
    
    # PRAW benötigt einen Account bei Reddit 
    # inklusive einer registrierten Anwendung mit Client-ID und Secret
    reddit = praw.Reddit(user_agent="test",
                         client_id="wpaIV3-b3AYOJQ", client_secret="-M_LPtLCpkqlJTCyg--Rg9ePAwg")
    
    # Das LocalTarget fuer die rohen Daten
    # Die Daten werden unter
    # "model/<version>/raw.csv gespeichert
    def output(self):
        return LocalTarget("model/%d/raw.csv" % self.version)
    
    # Die Posts werden heruntergeladen,
    # in einen Dataframe konvertiert
    # und als CSV in das Target geschrieben
    def run(self):
        dataset = reduce(lambda p, n: p.append(n), self.fetch_reddit_data())
        with self.output().open("w") as out:
            dataset.to_csv(out, encoding='utf-8', index=False, sep=';')

    def fetch_reddit_data(self):
        from pandas import DataFrame
        for sub in self.subreddits:
            posts = list(self.reddit.subreddit(sub).hot(limit=self.limit))
            relevant = DataFrame([p.__dict__ for p in posts])[['title', 'selftext', "subreddit"]]
            yield relevant


class Clean(Task):
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    version = IntParameter(default=1)
    limit = IntParameter(default=500)
    
    # Der verwendete Tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    # Die Liste von Stop-Woertern
    # die herausgefiltert werden
    stopwords = nltk.corpus.stopwords.words('english')
    
     # Der Stemmer fuer Englische Woerter
    stemmer = nltk.SnowballStemmer("english")
    
    # Als Abhaengigkeit wird der
    # Task *Download* zurueckgegeben
    def requires(self):
        return Download(self.version, self.limit)
    
    # Das LocalTarget fuer die sauberen Daten
    # Die Daten werden unter
    # "model/<version>/cleaned.csv gespeichert
    def output(self):
        return LocalTarget("model/%d/cleaned.csv" % self.version)

    # Die Rohdaten werden zerstueckelt
    # durch die Stopwort-Liste gefiltert
    # und auf ihre Wortstaemme zurueckgefuehrt
    def run(self):
        import pandas
        dataset = pandas.read_csv(self.input().path, encoding='utf-8', sep=';').fillna('')
        dataset["cleaned_words"] = dataset.apply(self.clean_words, axis=1)
        with self.output().open("w") as out:
            dataset[["cleaned_words", "subreddit"]].to_csv(out,  encoding='utf-8', index=False, sep=';')

    def clean_words(self, post):
        tokenized = self.tokenizer.tokenize(post["title"] + " " + post["selftext"])
        lowercase = [word.lower() for word in tokenized]
        filtered = [word for word in lowercase if word not in self.stopwords]
        stemmed = [self.stemmer.stem(word) for word in filtered]
        return " ".join(stemmed)


class TrainModel(PySparkTask):
    version = IntParameter(default=1)
    limit = IntParameter(default=500)

    # PySpark Parameter
    driver_memory = '1g'
    executor_memory = '2g'
    executor_cores = '2'
    num_executors = '4'
    master = 'local'
    
    # Als Abhaengigkeit wird der
    # Task *Clean* zurueckgegeben
    def requires(self):
        return Clean(self.version, self.limit)
    
    # Das LocalTarget fuer das Model
    # Die Daten werden unter
    # "model/<version>/model gespeichert
    def output(self):
        return LocalTarget("model/%d/model" % self.version)

    def main(self, sc, *args):
        from pyspark.sql.session import SparkSession
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import HashingTF, Tokenizer
        from pyspark.ml.classification import DecisionTreeClassifier
        
        # Initialisiere den SQLContext
        sql = SparkSession.builder\
            .enableHiveSupport() \
            .config("hive.exec.dynamic.partition", "true") \
            .config("hive.exec.dynamic.partition.mode", "nonstrict") \
            .config("hive.exec.max.dynamic.partitions", "4096") \
            .getOrCreate()
        
        # Lade die bereinigten Daten
        df = sql.read.format("com.databricks.spark.csv") \
            .option("header", "true") \
            .option("delimiter", ";") \
            .load(self.input().path)
        
        # Den Klassifikator trainieren
        labeled = df.withColumn("label", df.subreddit.like("datascience").cast("double"))
        train_set, test_set = labeled.randomSplit([0.8, 0.2])
        tokenizer = Tokenizer().setInputCol("cleaned_words").setOutputCol("tokenized")
        hashing = HashingTF().setNumFeatures(1000).setInputCol("tokenized").setOutputCol("features")
        decision_tree = DecisionTreeClassifier()
        pipeline = Pipeline(stages=[tokenizer, hashing, decision_tree])
        model = pipeline.fit(train_set)
        model.save(self.output().path)
