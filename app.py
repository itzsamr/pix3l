import os
import json
import streamlit as st
from PIL import Image
from app_funcs import instantiate_model, image_super_resolution, download_success
from streamlit_lottie import st_lottie
from streamlit_image_comparison import image_comparison
from st_clickable_images import clickable_images
import cv2
import numpy as np

# Set Streamlit page configuration
st.set_page_config(
    page_title="PiX3L",
    layout="wide",  # Use a wide layout for side-by-side images
    initial_sidebar_state="collapsed",  # Lock the sidebar by default
)

# Define upload and download paths
upload_path = "uploads/"
download_path = "downloads/"

# Function to authenticate the user
def authenticate(username, password):
    # Replace this with your actual authentication logic
    return username == "admin" and password == "password"

# Home Page
def home_page():
    st.title("Welcome to PiX3L (Image Enhancer App)")
    # Define a Lottie JSON URL
    def get(path:str):
        with open(path,"r") as p:
            return json.load(p)
    path = get("./Animation - 1699712164903.json")
    col1, col2 = st.columns(2)
    # Add content to the first column (adjust as needed)
    with col1:
        st.write("The Image Enhancer App, PiX3L, is a web application utilizing a hybrid AI model combining elements from StyleGAN, WGAN (Wasserstein Generative Adversarial Network), and CycleGAN for enhancing the quality of digital images. It features a multipage structure with an intuitive sidebar, including a home page, an upload and enhance page, a result comparison page, and a gallery page. Users can upload images, choose AI models, and download enhanced images in various formats. PiX3L aims to provide a user-friendly tool that combines machine learning, computer vision, and web development to deliver powerful image enhancement capabilities. The application is built on the Streamlit framework, offering a responsive and interactive user experience.")
    
    # Add the animation to the second column
    with col2:
        st_lottie(path)

# Upload and Enhance Page
def upload_and_enhance_page():
    st.title("Upload and Enhance Page")
    # Your existing code for Upload and Enhance page goes here
    # File uploader to upload the image for super resolution
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"])
        # Sidebar options
    st.sidebar.title("Options")

    # Radio button to choose the model for Image Super Resolution
    model_name = st.sidebar.radio("Choose Model for Image Super Resolution", ('HybridGAN model ✅', 'PSNR model ✅'))
    st.sidebar.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    
        with st.spinner(f"Enhancing..."):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, f"output_{uploaded_file.name}"))

        # Instantiate the selected model
            model = instantiate_model(model_name)

        # Perform image super resolution
            image_super_resolution(uploaded_image, downloaded_image, model)

        # Display the original and enhanced images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image 📷")
                original_image = Image.open(uploaded_image)
                st.image(original_image, use_column_width=True)

            with col2:
                st.subheader("Enhanced Image 🚀")
                enhanced_image = Image.open(downloaded_image)
                st.image(enhanced_image, use_column_width=True)

            # Provide download options for different image formats
            with open(downloaded_image, "rb") as file:
                supported_formats = {
                    '.jpg': 'image/jpg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.bmp': 'image/bmp',
                }
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                download_label = f"Download Enhanced Image 📷 ({file_extension})"
                mime_type = supported_formats.get(file_extension, 'image/jpeg')

                if st.download_button(
                    label=download_label,
                    data=file,
                    file_name=f"output_{uploaded_file.name}",
                    mime=mime_type
                ):
                    download_success()
    else:
        st.warning('⚠ Please upload your Image file')


# Result Comparison Page
def result_comparison_page():
    st.title("Result Comparison Page")
    # Your existing code for Upload and Enhance page goes here
    # File uploader to upload the image for super resolution
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"])
        # Sidebar options
    st.sidebar.title("Options")

    # Radio button to choose the model for Image Super Resolution
    model_name = st.sidebar.radio("Choose Model for Image Super Resolution", ('ESRGAN model ✅', 'PSNR model ✅'))
    st.sidebar.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    
        with st.spinner(f"Enhancing..."):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, f"output_{uploaded_file.name}"))

        # Instantiate the selected model
            model = instantiate_model(model_name)

        # Perform image super resolution
            image_super_resolution(uploaded_image, downloaded_image, model)

        # Display the original and enhanced images side by side
            col1, col2 = st.columns(2)

        # Add image comparison
            image_comparison(
                img1=uploaded_image,
                img2=downloaded_image,
                label1="Original Image",
                label2="Enhanced Image",
                )

        # Provide download options for different image formats
            with open(downloaded_image, "rb") as file:
                supported_formats = {
                    '.jpg': 'image/jpg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.bmp': 'image/bmp',
                }
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                download_label = f"Download Enhanced Image 📷 ({file_extension})"
                mime_type = supported_formats.get(file_extension, 'image/jpeg')

                if st.download_button(
                    label=download_label,
                    data=file,
                    file_name=f"output_{uploaded_file.name}",
                    mime=mime_type
                ):
                    download_success()
    else:
        st.sidebar.warning('⚠ Please upload your Image file')

# Gallery Page
def gallery_page():
    st.title("Enhanced Image Gallery")
    
    # Get a list of all files in the download folder
    enhanced_images = [f for f in os.listdir(download_path) if f.startswith("output_")]
    
    # Check if there are any enhanced images available
    if enhanced_images:
        selected_images = []
        view_images = []
        for image_file in enhanced_images:
            view_images.append(os.path.join(download_path, image_file))

        # Number of columns in the grid (you can adjust this as needed)
        n = 3

        groups = []
        for i in range(0, len(view_images), n):
            groups.append(view_images[i:i+n])
        for group in groups:
            cols = st.columns(n)
            for i, image_file in enumerate(group):
                # Use checkboxes to select images
                selected = cols[i].checkbox("", key=image_file)
                if selected:
                    selected_images.append(image_file)
                cols[i].image(image_file, use_column_width=True)

        if st.button("Remove Selected"):
            for image_file in selected_images:
                os.remove(image_file)
            st.success("Selected images removed successfully!")

        if st.button("Download Selected"):
            for image_file in selected_images:
                # Provide download option
                st.download_button(
                    label=f"Download {os.path.basename(image_file)}",
                    data=open(image_file, "rb").read(),
                    file_name=os.path.basename(image_file)
                )
    else:
        st.warning("No enhanced images available.")




# Main App
def main():
    # Sidebar options
    page = st.sidebar.radio("Navigation", ["Home", "Upload and Enhance", "Result Comparison", "Gallery"])

    if page == "Home":
        home_page()
    elif page == "Upload and Enhance":
        upload_and_enhance_page()
    elif page == "Result Comparison":
        result_comparison_page()
    elif page == "Gallery":
        gallery_page()
    else:
        st.warning("Invalid page selection.")

# Authentication
authenticated = st.sidebar.checkbox("Unlock Sidebar")

# Display login or logout options based on authentication status
if not authenticated:
    st.title("Login to your PiX3L Account :)")
    username = st.text_input("Username: ")
    password = st.text_input("Password: ", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success(f"Welcome, {username}!")
            authenticated = True
        else:
            st.error("Invalid credentials. Please try again.")

# Display content only if the user is authenticated
if authenticated:
    main()
else:
    st.warning("Please login to access PiX3L.")
