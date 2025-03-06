

import os
import glob
import shutil

ROOT_DIR = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/light_test_nsga/testing_baseline"
TRUTH_DIR = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/gen4_base/testing_baseline"
FILES_TO_COPY = [
    "elites_history.pkl", "elites.csv", "hall_of_fame.csv",
    "hof_history.pkl", "out.csv"
]

# shutil.rmtree(os.path.join(ROOT_DIR, "generation_5"), ignore_errors=True)
shutil.rmtree(os.path.join(ROOT_DIR, "logs"), ignore_errors=True)
shutil.copytree(os.path.join(TRUTH_DIR, "checkpoint"), os.path.join(ROOT_DIR, "checkpoint"), dirs_exist_ok=True)
shutil.copytree(os.path.join(TRUTH_DIR, "eval_inputs"), os.path.join(ROOT_DIR, "eval_inputs"), dirs_exist_ok=True)
shutil.copytree(os.path.join(TRUTH_DIR, "surrogate_weights"), os.path.join(ROOT_DIR, "surrogate_weights"), dirs_exist_ok=True)
for file in FILES_TO_COPY:
    shutil.copy2(os.path.join(TRUTH_DIR, file), ROOT_DIR)