import pandas as pd, numpy as np, datetime, random, cPickle as pickle
pd.set_option('max_colwidth', 200)

events = pd.read_csv("stark_2011_events.csv",
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

grades = pd.read_csv("anonymized11.csv",
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
events = events[(events.activityType == 'WATCH') | (events.activityType == 'LISTEN')]

# Counting the number of video accessed by each user
video_map = events.groupby(['userId', 'topicId']).size().unstack().fillna(value=0)
user_video = video_map.applymap((lambda x: 1 if x > 0 else 0)).sum(axis="columns")
user_video = pd.DataFrame(user_video, columns=['VideoAccessed'])

# Counting the number of suspension
user_eventType = events.groupby(['userId', 'eventType']).size().unstack()
user_suspended = user_eventType[['SUSPENDED']].fillna(value=0)

# Getting the grades
user_grades = grades[['userId', 'hw', 'final', 'course']].set_index('userId')

# Drop "final == 0"
user_grades = user_grades[user_grades.final != 0]

# Merge VideoAccessed and NumSuspension, then calculate the average number of suspension per video
user_data = user_video.join(user_suspended)
user_data = user_data[user_data.VideoAccessed > 0]
user_data['SuspensionPerVideo'] = user_data['SUSPENDED'] / user_data['VideoAccessed']

# Merge grades
user_data = user_data.join(user_grades).dropna()
user_data = user_data.sort(columns='VideoAccessed')

#print user_data
#user_data.to_csv("video_suspend_grade.csv")



# Dividing users into groups according to the number of suspension
user_group = []
for i in range(6):
    subgroup = user_data[(user_data.SuspensionPerVideo >= i) & (user_data.SuspensionPerVideo < i+1)]
    user_group.append(subgroup)

# Showing the result (space separated)
print ",hw,final,course,num_users"
for i in range(6):
    g =  user_group[i]
    print g.SuspensionPerVideo.mean(),
    print g.hw.mean(),
    print g.final.mean(),
    print g.course.mean(),
    print g.SuspensionPerVideo.size
