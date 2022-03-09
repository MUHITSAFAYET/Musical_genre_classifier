import numpy as np
import pandas as pd
import json
import sqlite3
import sqlalchemy
import pickle


with open('data.json', 'r') as datafile:
    data=json.load(datafile)

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

files_list=[]
for i in range(len(data['mfcc'])):
    with open('parrot'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(df['mfcc'][i],f)
    files_list.append('parrot'+str(i)+'.pkl')

f.close()


