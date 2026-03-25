from pymongo import MongoClient

client = MongoClient("mongodb+srv://djcarbajal_db_user:Mellow@cluster0.qupfqm3.mongodb.net/?appName=Cluster0")

db = client["TrainSenseAI"]

users = db["users"]
predictions = db["predictions"]
daily_inputs = db["daily_inputs"]