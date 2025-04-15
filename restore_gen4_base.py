

import os
import glob
import shutil

# dirs = ['ssi_retest_1', 'ssi_retest_2', 'ssi_retest_3']
dirs = ['testing_baseline_ssi']
for dir in dirs:
    ROOT_DIR = f"/storage/ice-shared/vip-vvk/data/AOT/psomu3/{dir}/testing_baseline"
    if dir == 'testing_baseline':
        ROOT_DIR = f"/storage/ice-shared/vip-vvk/data/AOT/psomu3/testing_baseline"
    TRUTH_DIR = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/gen4_base/testing_baseline"
    FILES_TO_COPY = [
        "elites_history.pkl", "elites.csv", "hall_of_fame.csv",
        "hof_history.pkl", "out.csv"
    ]
    ctr = 5
    candidate_dir = os.path.join(ROOT_DIR, f"generation_{ctr}")
    while os.path.exists(candidate_dir):
        shutil.rmtree(candidate_dir, ignore_errors=True)
        ctr += 1
        candidate_dir = os.path.join(ROOT_DIR, f"generation_{ctr}")
    shutil.rmtree(os.path.join(ROOT_DIR, "logs"), ignore_errors=True)
    shutil.copytree(os.path.join(TRUTH_DIR, "checkpoint"), os.path.join(ROOT_DIR, "checkpoint"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(TRUTH_DIR, "eval_inputs"), os.path.join(ROOT_DIR, "eval_inputs"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(TRUTH_DIR, "surrogate_weights"), os.path.join(ROOT_DIR, "surrogate_weights"), dirs_exist_ok=True)
    for file in FILES_TO_COPY:
        shutil.copy2(os.path.join(TRUTH_DIR, file), ROOT_DIR)