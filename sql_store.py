"""
Script for storing mfcc values according to genres in sql
"""

import numpy as np
import pandas as pd
import json
import sqlite3
import sqlalchemy
import pickle

#collect data from json file
with open('data.json', 'r') as datafile:
    data=json.load(datafile)

#provide specific names for genres
genre_name = []
for genre_id in range(len(data['labels'])):
    if data['labels'][genre_id] == 0:
        genre_name.append("disco")
    elif data['labels'][genre_id] == 1:
        genre_name.append("hiphop")
    elif data['labels'][genre_id] == 2:
        genre_name.append("classical")
    elif data['labels'][genre_id] == 3:
        genre_name.append("pop")
    elif data['labels'][genre_id] == 4:
        genre_name.append("metal")
    elif data['labels'][genre_id] == 5:
        genre_name.append("rock")
    elif data['labels'][genre_id] == 6:
        genre_name.append("blues")
    elif data['labels'][genre_id] == 7:
        genre_name.append("reggae")
    elif data['labels'][genre_id] == 8:
        genre_name.append("jazz")
    elif data['labels'][genre_id] == 9:
        genre_name.append("country")
print(len(genre_name))

#As the mfcc value has array of array having high dimension, values are converted
#into pickel object and later on added to the server as pickel object
#here, for individual samples, pickle files are being created
files_list=[]
for i in range(len(data['mfcc'])):
    with open('parrot'+str(i)+'.pkl', 'wb') as f:
        files_list.append(pickle.dump(df['mfcc'][i],f))

f.close()

#generate sql engnine and create table
engine=sqlalchemy.create_engine('sqlite:///genre.db')
conn = sqlite3.connect('engine')
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS genres( 
            labels Integer,
            genre_name text,
            mfcc PickleType)
            """)


#fill the table with information
for i in range(len(df["genre_name"])):
    c.execute("INSERT INTO genres VALUES(:labels, :genre_name, :mfcc)", {'labels': data['labels'][i], 'genre_name': genre_name[i], 'mfcc':files_list[i]})
    conn.commit()

conn.close()