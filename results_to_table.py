import os
import json
import utils
import numpy as np
from sys import argv
from glob import glob
from pathlib import Path


def get_metric_name(metric_index):
    metric_name_mapping = {
        0: "ATE",
        1: "AOE",
        2: "RTE",
        3: "RRE",
        4: "RIR",
        5: "Runtime"
    }
    return metric_name_mapping[metric_index]


def find_numerical_results_from_logs(filepath, outpath):
    for experiment in (sorted(glob(filepath))):
        with open(sorted(glob(f"{experiment}/logs/*.txt"))[-1]) as log:
            name = experiment.split("/")[-1]
            with open(f"{outpath}/{name}.txt", 'w') as out:
                for lines in log.readlines()[1:8]:
                    res = lines.split(":\t")[-1]
                    print(res, end="")
                    out.write(res)
            print("\n"+"-"*30)


def fill_in_latex_table(filepath, outpath, metric_index):
    shi, fast, sift, censure, akaze, orb = [], [], [], [], [], []
    for experiment in (sorted(glob(f"{filepath}/**"))):
        filename = experiment.split("/")[-1]
        info = filename.rstrip(".txt").split("_")
        method, detector = np.asarray(info[0:2]).ravel()
        descriptor = ""
        if len(info) > 2: # DESCRIPTOR_BASED
            descriptor = info[-1]
            print(detector, descriptor)
            with open(f"{filepath}/{filename}", 'r') as log:
                for lines in log.readlines()[metric_index+1:metric_index+2]:
                    res = lines.split(":\t")[-1]
                    #print(res)
                    if detector.lower() == "shi":
                        shi.append(res.strip())
                    elif detector.lower() == "fast":
                        fast.append(res.strip())
                    elif detector.lower() == "sift":
                        sift.append(res.strip())
                    elif detector.lower() == "censure":
                        censure.append(res.strip())
                    elif detector.lower() == "akaze":
                        akaze.append(res.strip())
                    elif detector.lower() == "orb":
                        orb.append(res.strip())

    with open(f"{outpath}/{get_metric_name(metric_index)}.txt", 'w') as out:
        out.write("\n\\begin{table}[] ")
        out.write("\n\\begin{tabular}{| c | c | c | c | c|}")
        out.write("\n\hline")
        out.write("\n                                & \\textbf{AKAZE}  & \\textbf{BRIEF}   & \\textbf{ORB}    & \\textbf{SIFT}      \\\\ \hline ")
        out.write("\n\\textbf{Shi - Tomasi} " + f"   & N/A              & {shi[0]}          & {shi[1]}         & {shi[2]}       " + "\\\\ \hline ")
        out.write("\n\\textbf{FAST}         " + f"   & N/A              & N/A               & N/A              & N/A            " + "\\\\ \hline ")
        out.write("\n\\textbf{SIFT}         " + f"   & N/A              & {sift[0]}         & N/A              & {sift[1]}      " + "\\\\ \hline ")
        out.write("\n\\textbf{CenSurE}      " + f"   & N/A              & {censure[0]}      & {censure[1]}     & {censure[2]}   " + "\\\\ \hline ")
        out.write("\n\\textbf{AKAZE}        " + f"   & {akaze[0]}       & {akaze[1]}        & {akaze[2]}       & {akaze[3]}     " + "\\\\ \hline ")
        out.write("\n\\textbf{ORB}          " + f"   & N/A              & {orb[0]}          & {orb[1]}         & {orb[2]}       " + "\\\\ \hline ")
        out.write("\n\end{tabular}")
        out.write("\n\end{table}")





if __name__ == '__main__':
    version_string  = f'v{argv[1]}'
    scenario_string = f'{(argv[2]).lower()}'
    metric_index = f'{argv[3]}'

    filepath        = f"results/images_{version_string}/{scenario_string}/**"
    values_path     = f"tables/{version_string}/{scenario_string}"
    tables_path     = f"latex_tables/{version_string}/{scenario_string}"
    utils.create_dir(values_path)
    utils.create_dir(tables_path)

    fill_in_latex_table(values_path, tables_path, int(metric_index))

