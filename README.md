## 《Sekiro™ Shadows Die Twice》【DoubleDQN】【Conv3D】

<p align="center">
    <a>English</a>
    | 
    <a href="https://github.com/ricagj/pysekiro_with_RL/blob/main/CN/README.md">中文</a>
</p>

![demo.jpg](https://raw.githubusercontent.com/ricagj/pysekiro/main/imgs/adjustment_02.png)  

## Quick start

[Quick_start.ipynb](https://github.com/ricagj/pysekiro_with_RL/blob/main/Quick_start.ipynb)  

## Project structuret

[How_it_works.ipynb](https://github.com/ricagj/pysekiro_with_RL/blob/main/How_it_works.ipynb)  

- pysekiro
    - img_tools
        - \__init__.py
        - adjustment.py
        - get_status.py
        - get_vertices.py
        - grab_screen.py
    - key_tools
        - \__init__.py
        - actions.py
        - direct_keys.py
        - get_keys.py
    - \__init__.py
    - Agent.py (DoubleDQN)
    - model.py
    - on_policy.py
    - off_policy.py

## Installation

#### Install Anaconda3

https://www.anaconda.com/  

#### Create a virtual environment and install dependencies

~~~shell
conda create -n pysekiro python=3.8
conda activate pysekiro
conda install pandas
conda install matplotlib
conda install pywin32
pip install opencv-python>=4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow>=2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c conda-forge jupyterlab
~~~

## Reference
https://github.com/Sentdex/pygta5  
https://github.com/analoganddigital/sekiro_tensorflow  
https://github.com/analoganddigital/DQN_play_sekiro  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter10_atari/BreakoutDeterministic-v4_tf.ipynb  
https://github.com/ZhiqingXiao/rl-book/blob/master/chapter06_approx/MountainCar-v0_tf.ipynb  
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network  