import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image



CODE_SIZE = 256
IMG_SHAPE = (36, 36, 3)

def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, CODE_SIZE)).astype('float32')



def generate_face(nb):
<<<<<<< HEAD
    generator = tf.keras.models.load_model('./generator',compile=False)
=======
    generator = tf.keras.models.load_model('./generator.h5',compile=False)
>>>>>>> origin/kaggle
    image_list = generator.predict(sample_noise_batch(bsize=nb))
    image_list = image_list.clip(0,255)
    return [image.reshape(IMG_SHAPE) for image in image_list]

 

nb_faces = st.sidebar.slider("Select Number of Faces to generate",1,6,value=1)



st.title('GAN Human Face Generator')

st.markdown("Welcome to  GAN Human Face Generator App. If you want to see more project and cool apps. Please visit my portofolio [page] (http://mustaphabounoua.ml/). ")

st.markdown(" Here’s the full  [code] (https://github.com/MustaphaBounoua/Gan-Human-Face) for this app if you would like to get a look.")

st.header("Generative Adversarial Networks")

<<<<<<< HEAD
st.markdown("A generative adversarial network is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014  [Paper](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).")
st.markdown("Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. \
=======
st.markdown("A generative adversarial network is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues \
            in 2014  [Paper](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).")
st.markdown("Generative adversarial networks (GANs) are algorithmic \
    architectures that use two neural networks, pitting one against the other \
            (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. \
>>>>>>> origin/kaggle
    They are used widely in image ,video  and voice generation.")

st.markdown("The generator learns to generate plausible data. The generated instances become negative training examples for the discriminator.")

st.markdown("The discriminator learns to distinguish the generator's fake data from real data. The discriminator penalizes the generator for producing implausible results.")

image ="https://miro.medium.com/max/3286/1*Pvn9wuntqx3UMsNGS-FAvg.png"

st.image(image, imagecaption='Gans Architecture' ,use_column_width=True)


st.header("Our Model")

st.markdown("In this app we will use Gan to generate plausible human faces. This might sounds as the first step of an evil plan for machines to take over." )


st.markdown("To train our Discriminator we will use as real images of people downloaded from [Link](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/#dev) " )


st.subheader("Discriminator")

st.write("TODO")


st.subheader("Generator")

st.write("TODO")

st.header("Demo")

if st.button('Generate Face'):
    if nb_faces >1:
        st.write('New {} faces generated'.format(nb_faces))
    else:
        st.write('New face generated')
    image_list = generate_face(nb_faces)
    for index, image in enumerate(image_list):
        st.image(image,width=200,clamp=True,caption="Face {} ".format(index))