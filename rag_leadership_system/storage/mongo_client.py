# storage/mongo_client.py
from pymongo import MongoClient
from config import settings


def get_mongo_client():
    return MongoClient(settings.MONGODB_URI)


def get_database():
    client = get_mongo_client()
    return client[settings.MONGODB_DB_NAME]