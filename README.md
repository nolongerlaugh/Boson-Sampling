# Boson-Sampling


Three algorithms for generating a sample of indistinguishable bosons in a linear network have been implemented

##  Problem
> Indistinguishable photons are fed into the first n modes of the linear optical network from m modes. At the output, the number of bosons is recorded in each mode. An optical network can be characterized by a matrix ![equations](https://latex.codecogs.com/svg.image?U^{m\times&space;m})
> ![Alt text](https://github.com/nolongerlaugh/Boson-Sampling/raw/master/image.png )
> The state is described by a vector:
> ![equations](https://latex.codecogs.com/svg.image?S=|i_1i_2&space;\dots&space;i_m&space;\rangle), where ![equations](https://latex.codecogs.com/svg.image?i_k ) - the number of bosons in ![equations](https://latex.codecogs.com/svg.image?k) in fashion.
> Probability of condition ![equations](https://latex.codecogs.com/svg.image?S) can be written as:
>
> ![equations](https://latex.codecogs.com/svg.image?P_S(S&space;=&space;|i_1i_2\dots&space;i_n|)=\frac{Per(U_S)}{i1!i2!\dots&space;i_n!}),
> 
> where ![equations](https://latex.codecogs.com/svg.image?U_S) is the nxn matrix, by taking ![equations](https://latex.codecogs.com/svg.image?i_k ) copies ![equations](https://latex.codecogs.com/svg.image?k)-th row of the matrix U.
## Methods
+ Calculating the full distribution and sampling from it as from a discrete distribution.
+ Distribution generation using Markov chain Monte Carlo.
+ Tensor-train decomposition of the distribution tensor and generation using conditional probability

## Simple use
```python
    from Sampler_all import *
    smp = Sampler(m, n) #создание объекта класса сэмплера с m модами и n бозонами. Создается случайная унитарная матрица U(mxm)
    smp.eval_density() #вычисление полного распределения
    smp.sample_from_density(smp.density, k) #генерировать из полного распределения k точек
    smp.sample_mis(k, 100, 50) #генерировать k точек с помощью МСМС
    smp.do_MPS_prepare(max_rank, ort_period) #подготовка тензора
    smp.sample_mps(k) #генерировать k точек с помощью Tensor-train разложение тензора распределения
```
