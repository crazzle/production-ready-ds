# Big Data mit Luigi und Python
Der Beispielcode zum Artikel "Vom Data Science Projekt zu produktionsreifem Code mit Python und Luigi"

## Requirements installieren
Die Pipelines nutzen verschiedene Python Pakete, von Luigi über Pandas zu NLTK und PySpark. Die Pakete können mit

```bash
pip install -r requirements.txt --user
```

installiert werden.

## Training
*00_training_pipeline.py* enthält den Code für die Trainingspipeline.

*Download* und *Clean* nutzen Standard Python Libraries (Pandas, PRAW, NLTK). Der *Training* Task ist als PySpark-Job implementiert.

Gestartet wird die Pipeline mit

```bash
PYTHONPATH='.' luigi --module 00_training_pipeline TrainModel --version 1 \
                                                              --local-scheduler
```

## Klassifikation
*01_classification_pipeline.py* enhtält den Code für die tägliche Klassifikationspipeline.

*Fetch* und *Clean* nutzen Standard Python Libraries (Pandas, PRAW, NLTK). Der *Classify* Task ist als PySpark-Job implementiert.

Gestartet wird die Pipeline mit

```bash
PYTHONPATH='.' luigi --module 01_classification_pipeline RangeDailyBase --of Classify \
                                                                        --stop=$(date +"%Y-%m-%d") \
                                                                        --days-back 4 \
                                                                        --Classify-version 1 \
                                                                        --reverse \
                                                                        --local-scheduler
```

