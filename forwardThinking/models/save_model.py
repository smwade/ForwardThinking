from pymongo import MongoClient
import json


def store_results(model, dataset='unknown'):
    """ Stores the results of a trained model as an experiment in a mongo database """
    client = MongoClient()
    db = client.forward_thinking
    db.experiments.insert_one(model.summary(dataset))

