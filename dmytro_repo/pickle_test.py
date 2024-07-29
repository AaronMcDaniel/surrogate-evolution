#import dataset_tracking
import pandas as pd
import pickle
#name = '../airborne-detection-starter-kit-master/data/ds_1_train_all.pkl'
name = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/everything.pkl'
#df = pd.read_pickle(name)
with open(name, 'rb') as file:
    one = pickle.load(file)

print(one)
# with open('/gv1/projects/GRIP_Precog_Opt/data_loading/dmytro-airborne-detection-starter-kit-master/data/results/run0/result_combined.pkl', 'wb') as fp:
#             pickle.dump(combined, fp)
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