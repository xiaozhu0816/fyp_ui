import os
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from clf_crop_dino import predict
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("LZ1: Tongue Image Classification using Vision Transformers Demo")
img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])


if img_file:
    path = os.getcwd()
    st.write(path)
    img = Image.open(img_file)
    genre = st.radio(label="Choose whether to crop", options=["Just original image", "Need to crop"])
    if genre == "Need to crop":      
        cropped_img = st_cropper(img, realtime_update=True, box_color='#12FF00',
                                        aspect_ratio=None)
    else:
        cropped_img = img

    text1 = '<p style="font-family:Courier; color:Red; font-size: 18px;">Important: Please crop the image before predicting. </p>'
    text2 = '<p style="font-family:Courier; color:Red; font-size: 18px;">Such that only the tongue is left in the image. </p>'
    text3 = '<p style="font-family:Courier; color:Red; font-size: 18px;">Or the result may be inaccurate. </p>'
    st.markdown(text1, unsafe_allow_html=True)
    st.markdown(text2, unsafe_allow_html=True)
    st.markdown(text3, unsafe_allow_html=True)

    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((300,300))
    st.image(cropped_img)

    predict_button = st.button(label="Predict")
    if(predict_button):
        st.write("Just a second...")
        labels_list = predict(cropped_img)
        for labels in labels_list:
            for i in labels:
                st.write("Prediction: ", i[0], ",   Score: ", i[1])
        st.write("Finished!")
