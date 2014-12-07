import pandas as pd, numpy as np, datetime, random, cPickle as pickle
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

# Vincent's Timing Features for User
# Based on panda's implementation of dayofweek
day_of_week = {0: 'Monday',
               1: 'Tuesday',
               2: 'Wednesday',
               3: 'Thursday',
               4: 'Friday',
               5: 'Saturday',
               6: 'Sunday'}

# calculate time features
# events = events[(events.activityType != 'DISCUSS') & 
#                (events.eventType != 'OPENED') &
#                (events.timeMinutes.apply(lambda x: pd.notnull(x)))]
#total_time = events.groupby('userId').timeMinutes.sum()
#day_time = events.groupby(['userId',events.updatedAt.apply(lambda x: day_of_week[x.dayofweek])]).timeMinutes.sum().unstack()
#event_time = events.groupby(['userId','eventType']).timeMinutes.sum().unstack()
#activity_time = events.groupby(['userId',events.activityType.apply(lambda x: 'WATCH' if x == 'LISTEN' else x)]).timeMinutes.sum().unstack()

# create full dataframe
#train_features = pd.DataFrame(total_time).join(day_time).join(event_time).join(activity_time).fillna(0)

#todo: add Jason's and Yu's features to 
for key in events:
    user = events[key]['userId']
    print str(user)


# calculate percents and z-score columns
#for feat in train_features.keys():
#    if feat != 'timeMinutes':
#        train_features[feat] = train_features[feat] / train_features.timeMinutes
#    train_features[feat] = stats.mstats.zscore(train_features[feat])

# # Jordeen adding features for scores
# for key in grades_rowdict:
#     user = grades_rowdict[key]['userId']
#     if user in train_features: # user already in train_features
#         # Set average score
#         sum_set_scores = 0
#         num_sets = 29
#         for i in range(0, num_sets):
#             sum_set_scores += grades_rowdict[key]['Set' + str(i)]
#         train_features[user]['average_set_score'] = sum_set_scores / float(num_sets)
#         # s average score
#         sum_s_scores = 0
#         for i in range(0, num_sets):
#             sum_s_scores += grades_rowdict[key]['s' + str(i)]
#         train_features[user]['average_s_score'] = sum_set_scores / float(num_sets)
#         # rest of the features
#         train_features[user]['course_score'] = grades_rowdict[key]['course']
#         train_features[user]['final_exam_score'] = grades_rowdict[key]['final']
#         train_features[user]['hw_score'] = grades_rowdict[key]['hw']
#         train_features[user]['letter'] = grades_rowdict[key]['letter']
#         train_features[user]['demerit'] = grades_rowdict[key]['demerit']
#     else:
#         pass

# MACHINE LEARNING CLUSTERING
#import numpy as np
#from sklearn.cluster import KMeans, DBSCAN
#kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
#kmeans.fit(train_features)

#from sklearn.cluster.bicluster import SpectralBiclustering
#model = SpectralBiclustering(n_clusters=5, method='log', random_state=0)
#model.fit(train_features)

#train_features.loc['bf7aa87b-444a-4eff-9f81-b4078e6dccd3']

#model.row_labels_


