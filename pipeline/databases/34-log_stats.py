#!/usr/bin/env python3
"""
Provides some stats about Nginx logs stored in MongoDB
"""

from pymongo import MongoClient

if __name__ == "__main__":
    """
    Gets stats about Nginx logs stored in MongoDB
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client.logs
    collection = db.nginx
    total_logs = collection.count_documents({})
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = [collection.count_documents({"method": method}) for method in methods]
    special_logs_count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{total_logs} logs where {total_logs} is the number of documents in this collection")
    print("Methods:")
    for method, count in zip(methods, method_counts):
        print(f"\t{count} logs with method={method}")
        print(f"\t{special_logs_count} logs with method=GET and path=/status")
