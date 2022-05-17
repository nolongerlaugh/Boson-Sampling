# Boson-Sampling


Реализовано три алгоритма генерации выборки неразличимых бозонов в линейной сети

##  Задача
> В первые n мод линейной оптической сети из m мод подаются неразличимые фотоны. На выходе в каждой моде регистрируется количество бозонов. Оптическую сеть можно характеризовать матрицей ![equations](https://latex.codecogs.com/svg.image?U^{m\times&space;m})
> ![Alt-текст](https://github.com/nolongerlaugh/Boson-Sampling/raw/master/image.png)
> Состояние описывается вектором:
> ![equations](https://latex.codecogs.com/svg.image?S=|i_1i_2&space;\dots&space;i_m&space;\rangle), где ![equations](https://latex.codecogs.com/svg.image?i_k) - количество бозонов в ![equations](https://latex.codecogs.com/svg.image?k)-ой моде.
> Вероятность состояния ![equations](https://latex.codecogs.com/svg.image?S) можно записать как:
> 
> ![equations](https://latex.codecogs.com/svg.image?P_S(S&space;=&space;|i_1i_2\dots&space;i_n|)=\frac{Per(U_S)}{i1!i2!\dots&space;i_n!}),
> 
> где ![equations](https://latex.codecogs.com/svg.image?U_S) - матрица nxn, путем взятия ![equations](https://latex.codecogs.com/svg.image?i_k) копий ![equations](https://latex.codecogs.com/svg.image?k)-ой строки матрицы U.

## Методы
+ Вычисление полного распределения и сэмплирование из него как из дискретного распределения.
+ Генерация распределения с помощью Марковской цепи Монте-Карло.
+ Tensor-train разложение тензора распределения и генерация с помощью условной вероятности

## Основные функции
```python
    from Sampler_all import *
    smp = Sampler(m, n) #создание объекта класса сэмплера с m модами и n бозонами. Создается случайная унитарная матрица U(mxm)
    smp.eval_density() #вычисление полного распределения
    smp.sample_from_density(smp.density, k) #генерировать из полного распределения k точек
    smp.sample_mis(k, 100, 50) #генерировать k точек с помощью МСМС
    smp.do_MPS_prepare(max_rank, ort_period) #подготовка тензора
    smp.sample_mps(k) #генерировать k точек с помощью Tensor-train разложение тензора распределения
```
