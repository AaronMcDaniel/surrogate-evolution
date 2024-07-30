#import dataset_tracking
import pandas as pd
import pickle
#name = '../airborne-detection-starter-kit-master/data/ds_1_train_all.pkl'
name = 'predictions.pkl'
#df = pd.read_pickle(name)
with open('data/results/run0/result01.pkl', 'rb') as file:
    one = pickle.load(file)

with open('data/results/run0/result02.pkl', 'rb') as file:
    two = pickle.load(file)

with open('data/results/run0/result03.pkl', 'rb') as file:
    three = pickle.load(file)

with open('data/results/run0/result04.pkl', 'rb') as file:
    four = pickle.load(file)

with open('data/results/run0/result05.pkl', 'rb') as file:
    five = pickle.load(file)

with open('data/results/run0/result06.pkl', 'rb') as file:
    six = pickle.load(file)

with open('data/results/run0/result07.pkl', 'rb') as file:
    seven = pickle.load(file)

with open('data/results/run0/result08.pkl', 'rb') as file:
    eight = pickle.load(file)

with open('data/results/run0/result09.pkl', 'rb') as file:
    nine = pickle.load(file)

with open('data/results/run0/result10.pkl', 'rb') as file:
    ten = pickle.load(file)

combined = [one, two, three, four, five, six, seven, eight, nine, ten]
with open('data/results/run0/result_combined.pkl', 'wb') as fp:
            pickle.dump(combined, fp)
#print(name + ':', df)
# print(type(df))
# print(type(df[list(df.keys())[0]]))
# print(df[list(df.keys())[0]])
#print(df)
#print(type(df[0]))
#print(list(df[0].keys())[:5])
# print(list(df[1].keys())[:5])
# print(len(df), len(list(df[0].keys())), len(list(df[1].keys())))

#df.head/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/ds_transform_1.pkl