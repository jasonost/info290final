import pandas as pd, numpy as np, datetime, random, cPickle as pickle

pd.set_option('max_colwidth', 200)

base_path = '../data/'
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

events[:10]

grades[:10]

events_coldict = events.to_dict()
events_rowdict = events.T.to_dict()

events_rowdict[10]

