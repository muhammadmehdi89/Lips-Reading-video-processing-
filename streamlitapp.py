import streamlit as st
import os
from moviepy.editor import VideoFileClip
from utils import load_data, num_to_char
from modelutil import load_model
import tensorflow as tf
import imageio


with st.sidebar:
    st.image('https://www.einfochips.com/blog/wp-content/uploads/2019/09/exploring-the-potential-of-computer-vision-across-industries-featured.jpg')
    st.title('Lipread')
    st.info('This app is developed using Lip-net architecture.')

st.title('LipNet App')

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)
col1, col2 = st.columns(2)


if options:
    with col1:
        with st.container():

            file_path = os.path.join("..", 'data', 's1', selected_video)
            converted_path = os.path.join("..", 'data', 's1', 'converted_video.mp4')

            # Convert MPG to MP4 using moviepy
            clip = VideoFileClip(file_path)
            clip.write_videofile(converted_path, codec='libx264')

            try:
                # Display the converted video
                video_ = open(converted_path, 'rb').read()
                st.video(video_)
            except Exception as e:
                st.error(f"Error displaying video: {e}")
    with st.container():
        video, alignments = load_data(tf.convert_to_tensor(file_path))
        model = load_model()
        yhat = model.predict(tf.expand_dims(video,axis=0))
        st.info('Actual Sentence')
        converted_prediction = tf.strings.reduce_join(num_to_char(alignments)).numpy().decode('utf-8')
        st.text(converted_prediction)
        decoder = tf.keras.backend.ctc_decode(yhat, [75],greedy=True)[0][0]
        st.info('Lip read Prediction')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

            
    with col2:
        import numpy as np
        import cv2
        st.info('This shows how each frames are moving')
        video, alignments = load_data(tf.convert_to_tensor(file_path))
        
        # Convert TensorFlow tensor to NumPy array
        video_np = video.numpy()
        
        resized_video = [cv2.resize(frame, (400, 100), interpolation=cv2.INTER_LINEAR) for frame in video_np]
       
        # Perform NumPy operations
        fv = (np.array(video_np).astype(np.uint8) * 255).squeeze()
        imageio.mimsave('./animation.gif', fv, duration=150)
        
        #imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif',width=400)

        st.text('Column1')
        

