
# QUICKSTART

```python
python 3.8
pytorch 1.8.1
pytorchvideo
mmaction2
mmcv-full

```

Install pytorchvideo and mmaction2，please read [https://github.com/facebookresearch/pytorchvideo/blob/main/INSTALL.md ]、(https://github.com/facebookresearch/pytorchvideo/blob/main/INSTALL.md)[https://github.com/open-mmlab/mmaction2/blob/master/docs_zh_CN/install.md](https://github.com/open-mmlab/mmaction2/blob/master/docs_zh_CN/install.md)

# MMAction2

Selecte  pre-training model to evaluate their inference performance ,such as, SlowFast,I3D,C3D and TSN.
Get the required pre-trained model: [https://github.com/open-mmlab/mmaction2]

## 1.Start Profiling


- Run **run.sh** to evaluate the performance of the pre-trained model

```python
./run.sh
```

- Optional arguments:

```python
config           #test config file path
checkpoints The  #checkpoint file to be evaluated
data             #video file/url or rawframes directory
label            #path to the label set
batch_num
batch_size
```


- Run **run.sh** to evaluate the performance of SlowFast:

```python
python run.py configs/slowfast_r50_4x16x1_256e_kinetics400_rgb.py\
    checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth \
    example.mp4 map/label_map_k400.txt --batch_size 1 --batch_num 11
```


The model input data is example.mp4, run.py calls modelcl.py to evaluate the model. There are four indicators that will be acquired, **throughput, P50-latency, P95-latency and P99-latency.**

## 2.Result


Set batch_size=1,batch_num=11,data=example.mp4

- The throughput curves of the above four models are shown in the following figure. It can be seen that the throughput of SlowFast and ID3 is higher than that of TSN and C3D. The specific data is located in **mm_latency_throughput.xls**

![image](https://user-images.githubusercontent.com/72073969/136967007-df0e96af-fd97-4744-b1f4-11ad43f08992.png)


- The latency curves of the above four models are shown in the following figure. It can be seen that the latency of SlowFast and ID3 are lower than TSN and C3D, and the latency time of SlowFast and ID3 is almost the same. From the perspective of latency , SlowFast and ID3 are better than TSN and C3D.  TSN is better than C3D.  The specific data is located in **mm_latency_throughput.xls**

![image](https://user-images.githubusercontent.com/72073969/136967138-02974b79-fbe8-4a0d-9774-201758133339.png)


- The four indicators of the model (**result_mm.xlsx**):

![image](https://user-images.githubusercontent.com/72073969/136967651-6d0f5d34-5ea2-4412-a4f7-e262b006e678.png)



# PyTorchVideo


Selecte SlowFast to evaluate their inference performance.

## 1.Start Profiling


- Run **runcla.sh** to evaluate the performance of the pre-trained model

```python
./runcla.sh
```

- Optional arguments:

```python
data        #video file/url or rawframes directory
label       #path to the label set
batch_num
batch_size
```


- Run **runcla.sh** to evaluate the performance of SlowFast:

```python
python runcla.py example.mp4 lab_map_k400.txt --name slowfast --batch_size 1 --batch_num 11
```

The model input data is example.mp4, **runcla.sh** calls **modelcl.py** to evaluate the model. There are four indicators that will be acquired, **throughput, P50-latency, P95-latency and P99-latency.**

## 2.Result


Set batch_size=1,batch_num=11,data=example.mp4

- The throughput curves of the above model is shown in the following figure. The specific data is located in **pv_latency_throughput.xls**

![image](https://user-images.githubusercontent.com/72073969/136968141-b2d95049-aea7-4695-a649-4864e58b1839.png)


- The latency curves of the model is shown in the following figure.The specific data is located in **pv_latency_throughput.xls**

![image](https://user-images.githubusercontent.com/72073969/136968231-a24dc9cf-40a3-42b2-ac27-f6dbe52481fb.png)


- The four indicators of the model (**result.xlsx**):
![image](https://user-images.githubusercontent.com/72073969/136968302-512012d4-7bef-44e3-8f81-17edfa407d4b.png)
