import numpy as np
import json
from scipy import stats
import statsmodels.stats.api as sms


scores_record_normal = {}
scores_record_uda = {}
num_samples_normal = 0
num_samples_uda = 0

with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/no_uda/scores.txt", 'r') as f:
# with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu_masked/scores_dynamic_mask.txt", 'r') as f:
    for line in f:
        # print(line.strip())
        scores = json.loads(line.strip())
        if not scores_record_normal:
            for class_reg in scores:
                scores_record_normal[class_reg] = {}
                for model_type in scores[class_reg]:
                    scores_record_normal[class_reg][model_type] = {}
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_normal[class_reg][model_type][cur_metric] = [scores[class_reg][model_type][cur_metric],]
        else:
            for class_reg in scores:
                for model_type in scores[class_reg]:
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_normal[class_reg][model_type][cur_metric].append(scores[class_reg][model_type][cur_metric])
        num_samples_normal += 1

with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu_masked/scores_dynamic_mask.txt", 'r') as f:
    for line in f:
        # print(line.strip())
        scores = json.loads(line.strip())
        if not scores_record_uda:
            for class_reg in scores:
                scores_record_uda[class_reg] = {}
                for model_type in scores[class_reg]:
                    scores_record_uda[class_reg][model_type] = {}
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_uda[class_reg][model_type][cur_metric] = [scores[class_reg][model_type][cur_metric],]
        else:
            for class_reg in scores:
                for model_type in scores[class_reg]:
                    for cur_metric in scores[class_reg][model_type]:
                        scores_record_uda[class_reg][model_type][cur_metric].append(scores[class_reg][model_type][cur_metric])
        num_samples_uda += 1


for class_reg in scores_record_normal:
    for model_type in scores_record_normal[class_reg]:
        print(model_type)
        perc = 0
        for cur_metric in scores_record_normal[class_reg][model_type]:
            if cur_metric in ['epoch_num', 'train_loss']:
                continue
            print("   ", cur_metric)
            data = np.array(scores_record_normal[class_reg][model_type][cur_metric])
            data_uda = np.array(scores_record_uda[class_reg][model_type][cur_metric])
            mean = np.mean(data)
            mean_uda = np.mean(data_uda)
            perc += (mean_uda - mean)/mean
            print("        Normal: sample mean:", mean)
            print("        UDA   : sample mean:", mean_uda)
            stdev = np.std(data, ddof=1)
            stdev_uda = np.std(data_uda, ddof=1)
            print("        Normal: Sample stdev:", stdev)
            print("        UDA:    Sample stdev:", stdev_uda)
            print("        Normal: Confidence interval of true mean:", mean - 1.96*stdev/(num_samples_normal**0.5), mean + 1.96*stdev/(num_samples_normal**0.5))
            print("        UDA:    Confidence interval of true mean:", mean_uda - 1.96*stdev_uda/(num_samples_uda**0.5), mean_uda + 1.96*stdev_uda/(num_samples_uda**0.5))
            res = stats.ttest_ind(data_uda, data, equal_var=False, alternative="less")
            print("        p-value:", res.pvalue, " (hypothesis test for mean of uda being less)")

            # mean2 = mean/3
            # stdev2 = np.std(data/3, ddof=1)
            # n, n2 = num_samples_normal, num_samples_normal
            # stderr = np.sqrt((stdev**2 / n) + (stdev2**2 / n2))
            # alpha = 0.05

            cm = sms.CompareMeans(sms.DescrStatsW(data_uda), sms.DescrStatsW(data))
            print("        confidence interval for diff in means:", cm.tconfint_diff(usevar='unequal'))
        print("    PERC overall decrease from normal to uda: ", perc)


            
