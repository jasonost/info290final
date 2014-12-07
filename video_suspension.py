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

# Counting the number of suspension
user_eventType = events.groupby(['userId', 'eventType']).size().unstack()
user_suspended = user_eventType[['SUSPENDED']].fillna(value=0)

# Getting the grades
user_grades = grades[['userId', 'hw', 'final', 'course']].set_index('userId')

# Merge suspension and grade
user_data = user_suspended.join(user_grades).dropna()
user_data = user_data.sort(columns='SUSPENDED')

# Dividing users into groups according to the number of suspension
user_group = []
user_group.append(user_data[user_data.SUSPENDED == 0])

for i in range(5):
    subgroup = user_data[(user_data.SUSPENDED > i * 20) & (user_data.SUSPENDED <= (i+1)*20)]
    user_group.append(subgroup)

user_group.append(user_data[user_data.SUSPENDED > 100])

# Showing the result (space separated)
print ",hw,final,course,num_users"
for i in range(7):
    g =  user_group[i]
    print g.SUSPENDED.mean(),
    print g.hw.mean(),
    print g.final.mean(),
    print g.course.mean(),
    print g.SUSPENDED.size
