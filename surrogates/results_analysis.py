import numpy as np
import json
from scipy import stats
import statsmodels.stats.api as sms
import pandas as pd


scores_record_mode1 = {}
scores_record_mode2 = {}
num_samples_mode1 = 0
num_samples_mode2 = 0

mode1 = "Strong codec"
mode2 = "Old codec"

outcsv = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/strong_codec/oldstrongComparison.csv"
with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/strong_codec/scores.txt", 'r') as f:
# with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu_masked/scores_dynamic_mask.txt", 'r') as f:
    for line in f:
        # print(line.strip())
        scores = json.loads(line.strip())
        if not scores_record_mode1:
            for class_reg in scores:
                scores_record_mode1[class_reg] = {}
                for model_type in scores[class_reg]:
                    scores_record_mode1[class_reg][model_type] = {}
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_mode1[class_reg][model_type][cur_metric] = [scores[class_reg][model_type][cur_metric],]
        else:
            for class_reg in scores:
                for model_type in scores[class_reg]:
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_mode1[class_reg][model_type][cur_metric].append(scores[class_reg][model_type][cur_metric])
        num_samples_mode1 += 1

with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/old_codec/scores.txt", 'r') as f:
    for line in f:
        # print(line.strip())
        scores = json.loads(line.strip())
        if not scores_record_mode2:
            for class_reg in scores:
                scores_record_mode2[class_reg] = {}
                for model_type in scores[class_reg]:
                    scores_record_mode2[class_reg][model_type] = {}
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_mode2[class_reg][model_type][cur_metric] = [scores[class_reg][model_type][cur_metric],]
        else:
            for class_reg in scores:
                for model_type in scores[class_reg]:
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_mode2[class_reg][model_type][cur_metric].append(scores[class_reg][model_type][cur_metric])
        num_samples_mode2 += 1

metricsFinal = []
mode1means = []
mode2means = []
pvals=[]
for class_reg in scores_record_mode1:
    for model_type in scores_record_mode1[class_reg]:
        print(model_type)
        perc = 0
        for cur_metric in scores_record_mode1[class_reg][model_type]:
            if cur_metric in ['epoch_num', 'train_loss']:
                continue
            print("   ", cur_metric)
            metricsFinal.append(cur_metric)
            data = np.array(scores_record_mode1[class_reg][model_type][cur_metric])
            data_mode2 = np.array(scores_record_mode2[class_reg][model_type][cur_metric])
            mean = np.mean(data)
            mean_mode2 = np.mean(data_mode2)
            mode1means.append(mean)
            mode2means.append(mean_mode2)
            print(f"        {mode1}: sample mean:", mean)
            print(f"        {mode2}: sample mean:", mean_mode2)
            stdev = np.std(data, ddof=1)
            stdev_mode2 = np.std(data_mode2, ddof=1)
            print(f"        {mode1}: Sample stdev:", stdev)
            print(f"        {mode2}:    Sample stdev:", stdev_mode2)
            print(f"        {mode1}: Confidence interval of true mean:", mean - 1.96*stdev/(num_samples_mode1**0.5), mean + 1.96*stdev/(num_samples_mode1**0.5))
            print(f"        {mode2}:    Confidence interval of true mean:", mean_mode2 - 1.96*stdev_mode2/(num_samples_mode2**0.5), mean_mode2 + 1.96*stdev_mode2/(num_samples_mode2**0.5))
            res = stats.ttest_ind(data_mode2, data, equal_var=False, alternative="less")
            print(f"        p-value:", res.pvalue, f" (hypothesis test for mean of {mode2} being less)")
            pvals.append(res.pvalue)
            # mean2 = mean/3
            # stdev2 = np.std(data/3, ddof=1)
            # n, n2 = num_samples_mode1, num_samples_mode1
            # stderr = np.sqrt((stdev**2 / n) + (stdev2**2 / n2))
            # alpha = 0.05

            cm = sms.CompareMeans(sms.DescrStatsW(data_mode2), sms.DescrStatsW(data))
            print(f"        confidence interval for diff in means ({mode2} minus {mode1}):", cm.tconfint_diff(usevar='unequal'))
        


export_data = [mode1means, mode2means, np.array(mode1means) - np.array(mode2means), pvals]
df = pd.DataFrame(export_data, columns=metricsFinal)
df.to_csv(outcsv, index=False, float_format='%.5f')
