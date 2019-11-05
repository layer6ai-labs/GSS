<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Guided Similarity Separation for Image Retrieval NeurIPS'2019
[Tensorflow](https://www.tensorflow.org/) implementation of
Guided Similarity Separation for Image Retrieval.


Authors: [Chundi Liu](http://www.cs.toronto.edu/~chundiliu/), [Guangwei Yu](http://www.cs.toronto.edu/~guangweiyu), Cheng Chang, Himanshu Rai, Junwei Ma, Satya Krishna Gorti, [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

# Prerequisites
tensorflow-gpu==1.13.1

numpy==1.16.0

# Dataset

We provide all the generated descriptors, dataset files and ground-truth files. To run the model, download the files from [here](https://s3.amazonaws.com/public.layer6.ai/GSS/GSS.tar.gz)
and extract them to a directory, refered to as `$DATA_DIR`. You should structure your data directory as follows:
```
GSS
  ├─ datasets
  │   ├─ instre
  |   |   ├─ gnd_instre.mat
  |   |   ├─ instre_gem.mat
  |   |   └─ instre_siamac.mat
  │   ├─ roxford5k
  |   |   ├─ gnd_roxford5k.mat
  |   |   └─ gnd_roxford5k.pkl
  |   └─ rparis6k
  |   |   ├─ gnd_rparis6k.mat
  |   |   └─ gnd_rparis6k.pkl
  ├─ features				
  │   ├─ instre_gem_index_ms_lw.npy
  │   ├─ instre_gem_query_ms_lw.npy
  │   ├─ roxford5k_resnet_rsfm120k_gem.mat
  │   └─ rparis6k_resnet_rsfm120k_gem.mat
  └─ graphs
      ├─ instre_index_ransac_graph.npy
      ├─ instre_query_ransac_graph.npy
      ├─ roxford5k_index_ransac_graph.npy
      ├─ roxford5k_query_ransac_graph.npy
      ├─ rparis6k_index_ransac_graph.npy
      └─ rparis6k_query_ransac_graph.npy
```
Provide `$DATA_DIR` as the argument to `--data-path` when running `train.py`.

# Descriptors

For all the experiments, we use pre-trained [GeM](https://arxiv.org/abs/1711.02512) model. The code and the pre-trained weights can be found in the author's [offical github repository](https://github.com/filipradenovic/cnnimageretrieval-pytorch).

# Examples
------

### roxford hard 57.5
```
python train.py --data-path $DATA_DIR --dataset roxford5k --num-layers 2 --k 5 --kq 5 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode descriptor --report-hard --beta-percentile 98
```

### roxford medium 77.8
```
python train.py --data-path $DATA_DIR --dataset roxford5k --num-layers 2 --k 5 --kq 5 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode descriptor --beta-percentile 98
```

### rparis hard 83.5
```
python train.py --data-path $DATA_DIR --dataset rparis6k --num-layers 2 --k 5 --kq 15 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode descriptor --report-hard --beta-percentile 98
```

### rparis medium 92.4
```
python train.py --data-path $DATA_DIR --dataset rparis6k --num-layers 2 --k 5 --kq 15 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode descriptor --beta-percentile 98
```

### instre 89.2
```
python train.py --data-path $DATA_DIR --dataset instre --num-layers 2 --k 10 --kq 10 --epoch 500 --lr 0.0001 --graph-mode descriptor --beta-percentile 98
```

### roxford hard ransac 64.7
```
python train.py --data-path $DATA_DIR --dataset roxford5k --num-layers 2 --k 5 --kq 10 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode ransac --report-hard --beta-percentile 98
```

### roxford medium ransac 80.6
```
python train.py --data-path $DATA_DIR --dataset roxford5k --num-layers 2 --k 5 --kq 10 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode ransac --beta-percentile 98
```

### rparis hard ransac 85.3
```
python train.py --data-path $DATA_DIR --dataset rparis6k --num-layers 2 --k 5 --kq 25 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode ransac --report-hard --beta-percentile 98
```

### rparis medium ransac 93.4
```
python train.py --data-path $DATA_DIR --dataset rparis6k --num-layers 2 --k 5 --kq 25 --epoch 200 --lr 0.0001 --gpu-id 0 --graph-mode ransac --beta-percentile 98
```

### instre ransac 92.4
```
python train.py --data-path $DATA_DIR --dataset instre --num-layers 2 --k 50 --kq 20 --epoch 500 --lr 0.0001 --graph-mode ransac --beta-percentile 98
```

# Citation

If you find this code useful to your research, please kindly cite the following publication.

  
    @inproceedings{liu2019guided,
      title={Guided Similarity Separation for Image Retrieval},
      author={Chundi Liu, Guangwei Yu, Cheng Chang, Himanshu Rai, Junwei Ma, Satya Krishna Gorti, Maksims Volkovs},
      booktitle={NeurIPS},
      year={2019}
    }


