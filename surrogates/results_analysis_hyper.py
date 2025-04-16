import numpy as np
import json
from scipy import stats
import statsmodels.stats.api as sms


scores_record_normal = {}
scores_record_uda = {}
num_samples_normal = 0
num_samples_uda = 0

# Basic parsing of the scoress.txt output from training a surrogate N times.
# Each line in the scores file has the dictionary returned from one call of the surrogate training function
with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu/scores_lambda_search_lambda_is_0.txt", 'r') as f:
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

scores_records = []

with open("/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu/merged_lambda_results.txt", 'r') as f:
    for line in f:
        # print(line)
        if line == "\n":
            continue
        if line[0] == "R":
            scores_records.append({})
            continue
        # print(line.strip())
        scores = json.loads(line.strip())
        # print(scores_records)
        if not scores_records[-1]:
            for class_reg in scores:
                scores_records[-1][class_reg] = {}
                for model_type in scores[class_reg]:
                    scores_records[-1][class_reg][model_type] = {}
                    for cur_metric in scores[class_reg][model_type]:
                        scores_records[-1][class_reg][model_type][cur_metric] = [scores[class_reg][model_type][cur_metric],]
        else:
            for class_reg in scores:
                for model_type in scores[class_reg]:
                    for cur_metric in scores[class_reg][model_type]:
                        scores_records[-1][class_reg][model_type][cur_metric].append(scores[class_reg][model_type][cur_metric])
        num_samples_uda += 1

with open("/home/hice1/psomu3/scratch/surrogate-evolution-2/surrogates/lambda_search.csv", 'w') as f:
    pass

for class_reg in scores_records[0]:
    for model_type in scores_records[0][class_reg]:
        print(model_type)
        for x, d in enumerate(scores_records):
            cur_info = []
            if x == 0:
                with open("/home/hice1/psomu3/scratch/surrogate-evolution-2/surrogates/lambda_search.csv", 'a') as f:
                    f.write(model_type + "\n")
                    f.write(",".join([f"{x} mean" for x in d[class_reg][model_type].keys() if x not in ['epoch_num', 'train_loss']]) + "\n")
            for cur_metric in d[class_reg][model_type]:
                if cur_metric in ['epoch_num', 'train_loss']:
                    continue
                print("   ", cur_metric)
                data_uda = np.array(d[class_reg][model_type][cur_metric])
                mean_uda = np.mean(data_uda)
                print("        UDA   : sample mean:", mean_uda)
                stdev_uda = np.std(data_uda, ddof=1)
                print("        UDA:    Sample stdev:", stdev_uda)
                print("        UDA:    Confidence interval of true mean:", mean_uda - 1.96*stdev_uda/(num_samples_uda**0.5), mean_uda + 1.96*stdev_uda/(num_samples_uda**0.5))
                cur_info.append(str(mean_uda))
                # mean2 = mean/3
                # stdev2 = np.std(data/3, ddof=1)
                # n, n2 = num_samples_normal, num_samples_normal
                # stderr = np.sqrt((stdev**2 / n) + (stdev2**2 / n2))
                # alpha = 0.05

                # print("        confidence interval for diff in means:", cm.tconfint_diff(usevar='unequal'))
            # print("    PERC overall decrease from normal to uda: ", perc)
            with open("/home/hice1/psomu3/scratch/surrogate-evolution-2/surrogates/lambda_search.csv", 'a') as f:
                f.write(",".join(cur_info) + "\n")


            
