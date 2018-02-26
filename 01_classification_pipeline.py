from luigi.contrib.spark import PySparkTask
from luigi.parameter import IntParameter, DateParameter
from luigi import LocalTarget, Task, WrapperTask


class Fetch(Task):
    from datetime import date, timedelta

    # Ein Datum wird als Parameter uebergeben
    date = DateParameter(default=date.today())

    # PRAW arbeitet mit Zeitintervallen
    # Um einen Tag zu importieren wird
    # von Tag N bis Tag N+1 importiert
    delta = timedelta(days=1)

    # Das LocalTarget fuer die rohen Daten
    # Die Daten werden unter
    # "daily/<datum>/roh.csv gespeichert
    def output(self):
        prefix = self.date.strftime("%m-%d-%Y")
        return LocalTarget("daily/%s/roh.csv" % prefix)

    # Die Posts fuer einen Tag
    # werden heruntergeladen,
    # in einen Dataframe konvertiert
    # und als CSV in das Target geschrieben
    def run(self):
        start = self.date
        end = start + self.delta
        posts = self.fetch(start, end)
        frame = self.konvertiere(posts)
        self.speichern(frame, self.output())

    def fetch(self, start, end):
        import time
        import praw
        subreddits = ["datascience", "gameofthrones"]
        reddit = praw.Reddit(user_agent="test",
                             client_id="wpaIV3-b3AYOJQ", 
                             client_secret="-M_LPtLCpkqlJTCyg--Rg9ePAwg")
        subreddits = '+'.join(subreddits)
        subreddit = reddit.subreddit(subreddits)
        start = time.mktime(self.date.timetuple())
        end = self.date + self.delta
        end = time.mktime(end.timetuple())
        filtered = list(subreddit.submissions(start=start, end=end))
        return filtered
    
    def konvertiere(self, posts):
        import pandas
        dataframe = pandas.DataFrame([f.__dict__ for f in posts])[['id', 'title', 'selftext', 'subreddit']]
        return dataframe

    def speichern(self, dataframe, target):
        with target.open("w") as out:
            dataframe.to_csv(out, encoding='utf-8', index=False, sep=';')


class Clean(Task):
    from datetime import date
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # Ein Datum wird als Parameter uebergeben
    date = DateParameter(default=date.today())

    # Die Liste von Stop-Woertern
    # die herausgefiltert werden
    stoppwoerter = nltk.corpus.stopwords.words('english')

    # Der verwendete Tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # Der Stemmer fuer Englische Woerter
    stemmer = nltk.SnowballStemmer("english")

    # Als Abhaengigkeit wird der
    # Task *Fetch* zurueckgegeben
    def requires(self):
        return Fetch(self.date)

    # Das LocalTarget fuer die sauberen Daten
    # Die Daten werden unter
    # "daily/<datum>/cleaned.csv gespeichert
    def output(self):
        prefix = self.date.strftime("%m-%d-%Y")
        return LocalTarget("daily/%s/cleaned.csv" % prefix)

    # Die Rohdaten werden zerstueckelt
    # durch die Stopwort-Liste gefiltert
    # und auf ihre Wortstaemme zurueckgefuehrt
    def run(self):
        csv = self.lade()
        tokenized = self.tokenize(csv)
        gefiltert = self.entferne(tokenized)
        wortstamm = self.stemme(gefiltert)
        csv["cleaned_words"] = wortstamm
        self.speichern(csv, self.output())

    def lade(self):
        import pandas
        dataset = pandas.read_csv(self.input().path, encoding='utf-8', sep=';').fillna('')
        return dataset

    def tokenize(self, csv):
        def tok(post):
            tokenized = self.tokenizer.tokenize(post["title"] + " " + post["selftext"])
            return tokenized
        tokenized = csv.apply(tok, axis=1)
        return tokenized

    def entferne(self, tokenized):
        lowercase = tokenized.apply(lambda post: [wort.lower() for wort in post])
        filtered = lowercase.apply(lambda post: [wort for wort in post if wort not in self.stoppwoerter])
        return filtered

    def stemme(self, gefiltert):
        wortstamm = gefiltert.apply(lambda post: [self.stemmer.stem(wort) for wort in post])
        wortstamm = wortstamm.apply(lambda post: " ".join(post))
        return wortstamm
    
    def speichern(self, dataframe, target):
        with target.open("w") as out:
            dataframe[["id", "cleaned_words", "subreddit"]].to_csv(out, encoding='utf-8', index=False, sep=';')


class ModelExists(WrapperTask):
    version = IntParameter(default=1)

    def output(self):
        return LocalTarget("model/%d/model" % self.version)


from luigi.contrib.spark import PySparkTask
from luigi.parameter import IntParameter, DateParameter
from luigi import LocalTarget
class Classify(PySparkTask):
    from datetime import date

    date = DateParameter(default=date.today())
    version = IntParameter(default=1)

    # PySpark Parameter
    driver_memory = '1g'
    executor_memory = '2g'
    executor_cores = '2'
    num_executors = '4'
    master = 'local'

    # Als Abhaengigkeit werden
    # Task *Clean* und *ModelExists*
    # zurueckgegeben
    def requires(self):
        return [ModelExists(self.version), Clean(self.date)]

    # Das LocalTarget fuer die Klassifikation
    # Die Daten werden unter
    # "daily/<datum>/ergebnis.csv gespeichert
    def output(self):
        prefix = self.date.strftime("%m-%d-%Y")
        return LocalTarget("daily/%s/ergebnis.csv" % prefix)

    def main(self, sc, *args):
        from pyspark.sql.session import SparkSession
        from pyspark.ml import PipelineModel
        from pyspark.sql.functions import when

        # Initialisiere den SQLContext
        sql = SparkSession.builder\
            .enableHiveSupport() \
            .config("hive.exec.dynamic.partition", "true") \
            .config("hive.exec.dynamic.partition.mode", "nonstrict") \
            .config("hive.exec.max.dynamic.partitions", "4096") \
            .getOrCreate()

        # Lade die bereinigten Daten
        df = sql.read.format("com.databricks.spark.csv") \
                     .option("delimiter", ";") \
                     .option("header", "true") \
                     .load(self.input()[1].path)

        # Lade das Model das zuvor mit SparkML trainiert wurde
        model = PipelineModel.load(self.input()[0].path)

        # Klassifiziere die Datensaetze eines Tages mit dem Model
        ergebnis = model.transform(df)[["id",
                                        "subreddit",
                                        "probability",
                                        "prediction"]]

        # Eine kleine Aufbereitung der Daten denn
        # die Klasse "1" hat den Namen "datascience"
        ergebnis = ergebnis.withColumn("prediction_label",
                                        when(ergebnis.prediction==1,
                                            "datascience") \
                                        .otherwise("gameofthrones"))

        # Der Einfachheit halber wird der Dataframe
        # in einen Pandas Dataframe konvertiert.
        # Dies sollte bei grossen Datenmengen vermieden.
        with self.output().open("w") as out:
            ergebnis.toPandas().to_csv(out,
                                       encoding='utf-8',
                                       index=False,
                                       sep=';')
