#!/usr/bin/env bash

PYTHONPATH='.'

luigi --module 00_training_pipeline TrainModel --version 1 \
                                               --local-scheduler

PYTHONPATH='.' luigi --module 01_classification_pipeline RangeDailyBase --of Classify \
                                                         --stop=$(date +"%Y-%m-%d") \
                                                         --days-back 4 \
                                                         --Classify-version 1 \
                                                         --reverse \
                                                         --local-scheduler