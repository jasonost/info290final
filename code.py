import pandas as pd, numpy as np, datetime, random, cPickle as pickle
from __future__ import division
pd.set_option('max_colwidth', 200)

base_path = './data/'
file_2011 = 'stark_2011_events.csv'
grades_2011 = 'anonymized11.csv'

events = pd.read_csv('%s%s' % (base_path,file_2011),
                     skiprows=1,
                     names=['userId',
                            'updatedAt',
                            'eventType',
                            'activityType',
                            'activityTitle',
                            'topicTitle',
                            'timeMinutes',
                            'scorePercent',
                            'anchorValue',
                            'anchorSelection',
                            'topicId',
                            'activityId',
                            'activityEventId',
                            'updateCount'],
                     parse_dates=["updatedAt"],
                     na_values=["\N"],
                     delimiter="\t"
                     )

grades = pd.read_csv('%s%s' % (base_path,grades_2011),
                     skiprows=1,
                     names=["dropcol",
                            "userId",
                            "Set0",
                            "Set1",
                            "Set2",
                            "Set3",
                            "Set4",
                            "Set5",
                            "Set6",
                            "Set7",
                            "Set8",
                            "Set9",
                            "Set10",
                            "Set11",
                            "Set12",
                            "Set13",
                            "Set14",
                            "Set15",
                            "Set16",
                            "Set17",
                            "Set18",
                            "Set19",
                            "Set20",
                            "Set21",
                            "Set22",
                            "Set23",
                            "Set24",
                            "Set25",
                            "Set26",
                            "Set27",
                            "Set28",
                            "demerit",
                            "s0",
                            "s1",
                            "s2",
                            "s3",
                            "s4",
                            "s5",
                            "s6",
                            "s7",
                            "s8",
                            "s9",
                            "s10",
                            "s11",
                            "s12",
                            "s13",
                            "s14",
                            "s15",
                            "s16",
                            "s17",
                            "s18",
                            "s19",
                            "s20",
                            "s21",
                            "s22",
                            "s23",
                            "s24",
                            "s25",
                            "s26",
                            "s27",
                            "s28",
                            "set2Adj",
                            "set3Adj",
                            "set4Adj",
                            "set5Adj",
                            "set9Adj",
                            "set13Adj",
                            "set23Adj",
                            "Cred.Code",
                            "hw",
                            "final",
                            "course",
                            "letter",
                            "Comment"])

grades = grades[grades.course.apply(lambda x: pd.notnull(x))]
del grades['dropcol']

events_coldict = events.to_dict()
events_rowdict = events.T.to_dict()
grades_coldict = grades.to_dict()
grades_rowdict = grades.T.to_dict()

events_rowdict[1378149]

# Vincent's Timing Features for User

# Based on panda's implementation of dayofweek
def day_of_week(number):
    if number == 0:
        return 'Monday'
    elif number == 1:
        return 'Tuesday'
    elif number == 2:
        return 'Wednesday'
    elif number == 3:
        return 'Thursday'
    elif number == 4:
        return 'Friday'
    elif number == 5:
        return 'Saturday'
    elif number == 6:
        return 'Sunday'

from collections import defaultdict

train_features = defaultdict(list)

for key in events_rowdict:
    user = events_rowdict[key]['userId']
    updated_at = events_rowdict[key]['updatedAt']
    time = events_rowdict[key]['timeMinutes']
    event_type = events_rowdict[key]['eventType']
    activity_type = events_rowdict[key]['activityType']
    if user in train_features: # user already in train_features
        if type(time) == float and str(time) != 'nan':
            train_features[user]['total_time'] += time
            train_features[user]['day_times'][day_of_week(updated_at.dayofweek)] += time
            train_features[user]['event_times'][event_type] += time
            train_features[user]['activity_times'][activity_type] += time
    else: # new user
        # Initialize features here
        user_features = dict()
        user_features['total_time'] = time if (type(time) == float and str(time) != 'nan') else 0.0
        user_features['day_times'] = {'Sunday':0, 'Monday':0, 'Tuesday':0, 'Wednesday':0, 'Thursday':0, 'Friday':0, 'Saturday':0}
        user_features['day_times'][day_of_week(updated_at.dayofweek)] += 1
        user_features['event_times'] = {'CLOSED': 0, 'OPENED': 0, 'SUSPENDED': 0, 'WORKED': 0}
        user_features['event_times'][event_type] = 1
        user_features['activity_times'] = {'ASSIGNMENT': 0, 'DISCUSS': 0, 'LISTEN': 0, 'PRACTICE': 0, 'READ': 0, 'WATCH': 0}
        user_features['activity_times'][activity_type] = 1
        train_features[user] = user_features
        
user_data = defaultdict(list)
key_data = defaultdict(list)

# add all user data to maps
for user in train_features:
    total_time = train_features[user]['total_time']
    user_data[user].append(('total_time', total_time))
    key_data['total_time'].append(total_time)
    del train_features[user]['event_times']['OPENED']
    del train_features[user]['activity_times']['DISCUSS']
    train_features[user]['activity_times']['LISTEN'] += train_features[user]['activity_times']['WATCH']
    del train_features[user]['activity_times']['WATCH']
    if total_time == 0:
        del train_features[user]
    for dictionary in train_features[user]:
        if dictionary != "total_time":
            for m in train_features[user][dictionary]:
                train_features[user][dictionary][m] = train_features[user][dictionary][m] / total_time
                user_data[user].append((m, train_features[user][dictionary][m]))
                key_data[m].append(train_features[user][dictionary][m])

# create number->zscore mappings for each key
from scipy import stats
mappings = {}
for key in key_data:
    inner_dict = {}
    maps = zip(key_data[key], stats.zscore(key_data[key]))
    for m in maps:
        inner_dict[m[0]] = m[1]
    mappings[key] = inner_dict
                
# standardize all other features
user_data = dict(user_data)
data_order = []
tmp = 0
for key in user_data:
    all_user_data = user_data[key]
    zdata = []
    for data in all_user_data:
        if tmp == 0:
            data_order.append(data[0])
        zdata.append(mappings[data[0]][data[1]])
    tmp = 1    
    user_data[key] = zdata

# adding features for scores
for key in grades_rowdict:
    user = grades_rowdict[key]['userId']
    if user in train_features: # user already in train_features
        # Set average score
        sum_set_scores = 0
        num_sets = 29
        for i in range(0, num_sets):
            sum_set_scores += grades_rowdict[key]['Set' + str(i)]
        train_features[user]['average_set_score'] = sum_set_scores / float(num_sets)
        # s average score
        sum_s_scores = 0
        for i in range(0, num_sets):
            sum_s_scores += grades_rowdict[key]['s' + str(i)]
        train_features[user]['average_s_score'] = sum_set_scores / float(num_sets)
        # rest of the features
        train_features[user]['course_score'] = grades_rowdict[key]['course']
        train_features[user]['final_exam_score'] = grades_rowdict[key]['final']
        train_features[user]['hw_score'] = grades_rowdict[key]['hw']
        train_features[user]['letter'] = grades_rowdict[key]['letter']
        train_features[user]['demerit'] = grades_rowdict[key]['demerit']
    else:
        pass


# Testing Vincent's Timing Features
from pprint import pprint

test_user = train_features['c9744297-65d9-42d8-9127-c42f4e0f1c9a']

print "\nTotal time on class:"

# Test user's total time spent on class
pprint(test_user['total_time'])

print "\nTime spend on class based on day of week:"

# Test user's time spent on class based on day of week
pprint(test_user['day_times'])

print "\nTime spent on class based on event type:"

# Test user's time spent on class based on event type
pprint(test_user['event_times'])

print "\nTime spent on class based on activity type:"

# Test user's time spent on class based on activity type
pprint(test_user['activity_times'])

print "\nAverage time between events:"

# Test user's average time spent on class between events
pprint(test_user['average_times'])

print "\nAll user features involving time:"

# Test user's features
pprint(test_user)


# MACHINE LEARNING CLUSTERING
# Vincent MACHINE LEARNING CLUSTERING
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
users = []
skl_features_array = []
def main():
    count = 0
    for user in user_data:
        # create array of features
#         if count > 300:
#             return
        if len(user_data[user]) != 1:
            users.append(user)
            skl_features_array.append(user_data[user])
        count += 1
main()

# inspect variables
# pprint(users)
# pprint(data_order)
# pprint(skl_features_array[0])

# # KMEANS
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
kmeans.fit(skl_features_array)
cluster = zip(users, kmeans.labels_)
cluster_to_users = defaultdict(list)
for c in cluster:
    cluster_to_users[c[1]].append(c[0])
cluster_to_users = dict(cluster_to_users)
from collections import Counter
c = Counter(kmeans.labels_)
# pprint(c) # overview of clusters
# pprint(data_order) # order of the data
# pprint(cluster_to_users) # all users in the given clusters
# example_cluster = 1
# pprint(user_data[cluster_to_users[example_cluster][0]]) # finding info of specific user

# SWITCH TO DBSCAN
# x = np.array(skl_features_array)
# cluster_hash = {}
# e = 0.1
# while e < 2:
#     dbscan = DBSCAN(eps=e, min_samples=10)
#     c = Counter(dbscan.fit_predict(x))
#     cluster_hash[e] = c
#     e += .1
# pprint(cluster_hash)