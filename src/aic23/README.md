# AI City Challenge 2023 - Track 5
## Team ID 33

## Automation Lab, Sungkyunkwan University

---

---

#### I. Installation

1. Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html


2. Open new Terminal, create new conda environment named **aic22track1** and activate it with following commands:
```shell
conda env create -f setup/aic22track1_py3.9_torch1.10.1.yml 

conda activate aic22track1
```

---


#### II. Data preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2022-data-and-evaluation/

##### b. Video data import

Add video files to **AIC22_Track_1_MTMC/projects/tss/data/aic22_mtmc/dataset_a**.
   
The program folder structure should be as following:

```
AIC22_Track_1_MTMC
├── projects
│   ├──tss
│   │   ├── data
│   │   │   └── aic22_mtmc
│   │   │       ├── dataset_a
│   │   │       │   ├── c041
│   │   │       │   ├── c042
│   │   │       │   ├── c043
│   │   │       │   ├── c044
│   │   │       │   ├── c045
│   │   │       │   └── c046
│   │   │       ├── outputs
...
```

---

#### III. Reference

##### a. Download weight 

Download weight from [Release](https://o365skku-my.sharepoint.com/:u:/g/personal/duongtran_o365_skku_edu/EWSZTI_H2-VLshgNwa2RlmYBQW4DyCuk5WvV17cd1p9Zjw?e=LFViMC) then put it into **AIC22_Track_1_MTMC/models_zoo**.

The folder structure should be as following:
```
AIC22_Track_1_MTMC
├── models_zoo
│   └──pretrained
│       ├── reid
│       └── detector
```

##### b. Run inference

And the running script to get the result

```shell
bash projects/tss/runs/run_track_mtmc.sh 
```

##### c. Get the result
After more than 2-3 hours, we get the result:
```
AIC22_Track_1_MTMC/projects/tss/data/aic22_mtmc/outputs/mtmc_result.txt
```
