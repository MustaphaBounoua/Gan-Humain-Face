# Gan-Human-Face
Generating human faces with Adversarial Networks


A generative adversarial network is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014  [Paper](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)


Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. They are used widely in image ,video  and voice generation.
# Human Face Generator using GANs

A generative adversarial network is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014.
[Paper](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. They are used widely in image ,video  and voice generation.

* The generator learns to generate plausible data. The generated instances become negative training examples for the discriminator.

* The discriminator learns to distinguish the generator's fake data from real data. The discriminator penalizes the generator for producing implausible results.

![Gans Architecture](https://miro.medium.com/max/3286/1*Pvn9wuntqx3UMsNGS-FAvg.png)


# Data Set

We will use a celebrity images dataset [Kaggle Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)

# Env

Recommended execution env : Kaggle using GPU mode  [How ?](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu)



# Demo

The streamlit app is a demonstrator which allows to visualize the results [Demo] (https://gan-face-generator.herokuapp.com/)
