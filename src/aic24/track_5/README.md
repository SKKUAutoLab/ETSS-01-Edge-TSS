# AI City Challenge 2024 - Track 5
## Team ID - #5

## Automation Lab, Sungkyunkwan University

# A. Run from Docker

---

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2024-data-and-evaluation/

Download dataset to the folder **<folder_test_dataset>**

The dataset folder structure should be as following:

```
<folder_test_dataset>
│   ├── videos
│   │   ├── 001
│   │   ├── 002
...
```

##### b. Load Docker

Change the **<folder_test_dataset>** with your path and run:

```shell
docker run  \
    --ipc=host  \
    --gpus all   \
    -v <folder_test_dataset>:/usr/src/aic24-track_5/data   \
    -it supersugar/skku_automation_lab_aic24_track_5:latest
```

##### c. Run inference

And the running script to get the result

```shell
bash script/run_track_5_docker.sh 
```

##### d. Get the result
After more than 2-3 hours, we get the result:
```
<folder_test_dataset>/outputs_s2_v8_det_v8_iden/final_result_s2.txt
```

---

# B. Run from source

---

#### I. Installation

1. Download & install Miniconda or Anaconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html


2. Open new Terminal, create new conda environment named **aic23_track_5** and activate it with following commands:

```shell
conda create --name aic23_track_5 python=3.10

conda activate aic23_track_5

pip install -r requirements.txt
```

---


#### II. Data preparation

##### a. Data download

Go to the website of AI-City Challenge to get the dataset.

- https://www.aicitychallenge.org/2023-data-and-evaluation/

##### b. Video data import

Add video files to **Edge-TSS/src/aic23/track_5/data/**.
   
The program folder structure should be as following:

```
Edge-TSS
├── src
│   ├──aic23
│   │   ├── track_5
│   │   │   └── data
│   │   │       └── videos
│   │   │  
...
```

---

#### III. Reference

##### a. Download weight 

Download weight from [Release](https://o365skku-my.sharepoint.com/:f:/g/personal/duongtran_o365_skku_edu/Eo2nfe_g62VNocpi_6mOIjsBFPbXaDiVat1C7vaJ6HLJ_g?e=e5tjcB) then put it into **Edge-TSS/src/aic23/track_5/models_zoo**.

The folder structure should be as following:
```
Edge-TSS
├── src
│   ├──aic24
│   │   ├── track_5
│   │   │   ├── models_zoo
│   │   │   │   └──yolov8
│   │   │   │       ├── yolov8x6_1280_1cls_track_5_filtered_helmet
│   │   │   │       ├── yolov8x6_320_7cls_crop_both_v2
│   │   │   │       ├── yolov8x6_384_7cls_crop_both_v1
│   │   │   │       ├── yolov8x6_448_7cls_crop_both_v1
│   │   │   │       ├── yolov8x6_512_7cls_crop_both_v1
│   │   │   │       └── yolov8x6_576_7cls_crop_both_v1
```

##### b. Run inference

And the running script to get the result

```shell
cd Edge-TSS/src/aic23/track_5/

bash script/run_track_5_s2_docker.sh 
```

##### c. Get the result
After more than 2-3 hours, we get the result:
```
Edge-TSS/src/aic23/track_5/data/output_aic24/final_result.txt
```

---

# C. Training Dataset

---

##### a. Dataset for the Detector

Download dataset for Detector from [link](https://o365skku-my.sharepoint.com/:f:/g/personal/duongtran_o365_skku_edu/Eo2nfe_g62VNocpi_6mOIjsBFPbXaDiVat1C7vaJ6HLJ_g?e=e5tjcB).

##### b. Dataset for the Identifier

Download dataset for Identifier from [link](https://o365skku-my.sharepoint.com/:f:/g/personal/duongtran_o365_skku_edu/Eo2nfe_g62VNocpi_6mOIjsBFPbXaDiVat1C7vaJ6HLJ_g?e=e5tjcB).
