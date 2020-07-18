<h3 align="center">Stacking Autoencoders</h3>

<p align="center">
  <a href="" rel="noopener">
 <img width=500px height=200px src=";/../readme_elmts_folder/img_auto_enc.png" alt="Bot logo"></a>
</p>

---

Centrale Lille | [Anas ESSOUNAINI](https://www.linkedin.com/in/anas-essounaini-b7514014a/) | [Contact us][anas-email]

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About <a name = "about"></a>](#-about)
- [🎥 Repository Structure  <a name = "repo-struct"></a>](#-repository-structure)
- [🎉 Acknowledgements <a name = "acknowledgement"></a>](#-acknowledgements)


## 🧐 About <a name = "about"></a>

In this project, we are interested in a popular __Ensemble Learning__ method called __Stacking__. We are going to use this approach to _stack autoencoders_.

An autoencoder is a special type of neural networks whose purpose is to reconstruct the inputs <a href="https://www.codecogs.com/eqnedit.php?latex=x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{i}" title="x_{i}" /></a> passing through several intermediate representations (or codes) corresponding to the outputs of the layers. The
objective function minimized by gradient descent during learning by the back-propagation algorithm is the mean square error. If <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{x}_{i}" title="\tilde{x}_{i}" /></a>
is the input reconstructed by the network, the objective function is : <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{n}&space;\sum&space;_i&space;\left&space;\|&space;x_i&space;-&space;\tilde{x}_i\right&space;\|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{n}&space;\sum&space;_i&space;\left&space;\|&space;x_i&space;-&space;\tilde{x}_i\right&space;\|" title="\frac{1}{n} \sum _i \left \| x_i - \tilde{x}_i\right \|" /></a> .

Here is the simple structure of the autoencoder that we are going to use :
> Input <a href="https://www.codecogs.com/eqnedit.php?latex=x_i&space;\Rightarrow" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i&space;\Rightarrow" title="x_i \Rightarrow" /></a> code 1 <a href="https://www.codecogs.com/eqnedit.php?latex=z_{i,1}&space;\Rightarrow" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{i,1}&space;\Rightarrow" title="z_{i,1} \Rightarrow" /></a> code 2 <a href="https://www.codecogs.com/eqnedit.php?latex=z_{i,1}&space;\Rightarrow" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{i,2}&space;\Rightarrow" title="z_{i,2} \Rightarrow" /></a> reconstructed code 1 
<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{z}_{i,1}&space;\Rightarrow" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{z}_{i,1}&space;\Rightarrow" title="\tilde{z}_{i,1} \Rightarrow" /></a> reconstructed input <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{x}_{i}" title="\tilde{x}_{i}" /></a>. 

The passage from one layers' representation to the other is parametered with weight matrix <a href="https://www.codecogs.com/eqnedit.php?latex={W}_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{W}_{i}" title="{W}_{i}" /></a> and a bias <a href="https://www.codecogs.com/eqnedit.php?latex={b}_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{b}_{i}" title="{b}_{i}" /></a>. 

Our goal is to learn the weight matrices and biases. In order to simplify and lower computational time necessary for training, we are going to use the _tied wheight hypothesis_. In other words, we consider that our auroencoder is symetrical i.e: <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_red&space;W_{4}&space;=&space;W_{1}^T&space;,&space;W_{3}&space;=&space;W_{2}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_red&space;W_{4}&space;=&space;W_{1}^T&space;,&space;W_{3}&space;=&space;W_{2}^T" title="W_{4} = W_{1}^T , W_{3} = W_{2}^T" /></a> .

## 🎥 Repository Structure  <a name = "repo-struct"></a>

This repository consists of one main directory `src` that contains two files :

1. `main_autoencoder.py`: We coded a 3-layer auto-encoder and train it on the MNIST dataset which contains images of handwritten numbers. The images are in
grayscale and size 28 × 28. This dataset contains 65,000 learning examples divided into 55,000 for
the train and in 10,000 for the test.

1. `main_stacked_autoencoders.py`: We trained the same network as in previous file but layer by layer. Once our weights are determined,
we will add a layer of the softmax regression type to obtain a usual MLP. We then launch
the back-propagation algorithm on this network, some of the weights of which are intelligently initialized.









## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used, especially my Ensemble Learning Professor John Klein

[anas-email]: mailto:essounaini97@gmail.com
