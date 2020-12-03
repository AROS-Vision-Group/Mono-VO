import os
import json
import utils
import numpy as np
from sys import argv
from glob import glob
from pathlib import Path


def get_metric_name(metric_index):
    metric_name_mapping = {
        0: "ATE",       # Absolute Trajectory Error(ATE)[m]
        1: "ARE",       # Absolute Rotation Error(AOE)[deg]
        2: "RTE",       # Relative Trajectory Error(RTE)[m]
        3: "RRE",       # Relative Rotation Error(RRE)[deg]
        4: "RIR",       # Ransac Inlier Ratio (RIR)
        5: "Runtime"    # Runtime [seconds/frame]
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
    shi, sift, censure, akaze, orb = [], [], [], [], []
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
                    elif detector.lower() == "sift":
                        sift.append(res.strip())
                    elif detector.lower() == "censure":
                        censure.append(res.strip())
                    elif detector.lower() == "akaze":
                        akaze.append(res.strip())
                    elif detector.lower() == "orb":
                        orb.append(res.strip())

    with open(f"{outpath}/{get_metric_name(metric_index)}.txt", 'w') as out:
        out.write("\\begin{table}[ht!]")
        out.write("\n\\centering")
        out.write("\n\\begin{tabular}{l|cccc}")
        out.write("\n                                & \multicolumn{4}{c}{\\textbf{Descriptor}} \\\\")
        out.write("\n\cline{2-5}")
        out.write("\n\\textbf{Detector}              & \\textit{BRIEF}    & \\textit{SIFT}  & \\textit{ORB}  & \\textit{AKAZE}      \\\\")
        out.write("\n\hline")
        out.write("\n\\textit{SIFT}         " + f"   & {sift[0]}          & {sift[1]}       & -              & -                 " + "\\\\")
        out.write("\n\\textit{Shi-Tomasi}   " + f"   & {shi[0]}           & {shi[2]}        & {shi[1]}       & -                 " + "\\\\")
        out.write("\n\\textit{CenSurE}      " + f"   & {censure[0]}       & {censure[2]}    & {censure[1]}   & -                 " + "\\\\")
        out.write("\n\\textit{ORB}          " + f"   & {orb[0]}           & {orb[2]}        & {orb[1]}       & -                 " + "\\\\")
        out.write("\n\\textit{AKAZE}        " + f"   & {akaze[1]}          & {akaze[3]}     & {akaze[2]}     & {akaze[0]}       " + "\\\\")

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



