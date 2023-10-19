#!/usr/bin/env python3
"""
Prints the location of a specific Github user
"""

import requests
import sys
import time

def get_user_location(api_url):
    """
    Prints the location of a specific Github user
    """
    response = requests.get(api_url)
    
    if response.status_code == 200:
        user_data = response.json()
        location = user_data.get('location')
        if location:
            print(f"User's Location: {location}")
        else:
            print("User's location not available.")
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        current_time = int(time.time())
        reset_minutes = (reset_time - current_time) // 60
        print(f"Reset in {reset_minutes} min")
    else:
        print("Unexpected error occurred.")

if __name__ == '__main__':
    """
    Prints the location of a specific Github user
    """
    if len(sys.argv) != 2:
        print("Usage: python 2-user_location.py <API_URL>")
    else:
        api_url = sys.argv[1]
        get_user_location(api_url)
