import os
import io
import json
import streamlit as st
from app_funcs import instantiate_model, image_super_resolution, download_success
from streamlit_lottie import st_lottie
from streamlit_image_comparison import image_comparison
import replicate
from PIL import Image, ImageEnhance, UnidentifiedImageError
from streamlit_image_select import image_select
import requests
import zipfile
import pyrebase
import segno
from PIL import Image
from io import BytesIO
import base64
import hashlib


# Set Streamlit page configuration
st.set_page_config(
    page_title="PiX3L | Samar",
    layout="wide",  # Use a wide layout for side-by-side images
    initial_sidebar_state="collapsed",  # Lock the sidebar by default
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define upload and download paths
upload_path = "static/uploads/"
download_path = "static/downloads/"


# Function to authenticate the user
def authenticate(username, password):
    # Replace this with your actual authentication logic
    return username == "admin" and password == "password"


def display_image_status(status_text, status_icon="⚙️"):
    with st.spinner(f"{status_icon} {status_text}"):
        st.empty()


# Home Page
def home_page():
    st.title(":rainbow[Welcome to PiX3L!]")

    # Define a Lottie JSON URL
    def get(path: str):
        with open(path, "r") as p:
            return json.load(p)

    path = get("./Animation - 1699712164903.json")
    col1, col2 = st.columns(2)
    # Add content to the first column (adjust as needed)
    with col1:

        st.write(
            "The magical web application that transforms ordinary images into extraordinary works of art! 🚀"
        )
        st.write(
            "1. **HybridGAN:** 🛠 An ensemble of StyleGAN, WGAN, and CycleGAN designed for advanced image enhancement."
        )
        st.write(
            "2. **HybridNet:** 🖼️ Integrating LIME, MSRNet, DRBL, and EG to excel in low-light image enhancement."
        )
        st.write(
            "3. **Multipage Structure:** 📚 Navigate seamlessly with dedicated pages for home, upload & enhance, HybridNet, and gallery."
        )
        st.write(
            "4. **Responsive & Interactive:** 🌐 Crafted using the Streamlit framework to ensure a user-friendly and interactive experience."
        )

    # Add the animation to the second column
    with col2:
        st_lottie(path)


# Result Comparison Page
def result_comparison_page():
    st.title(":rainbow[Upload->Enhance->Enjoy]🌈")
    st.write(
        "An ensemble of StyleGAN, WGAN, and CycleGAN designed for advanced image enhancement."
    )
    # Your existing code for Upload and Enhance page goes here
    # File uploader to upload the image for super resolution
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["png", "jpg", "bmp", "jpeg"]
    )
    # Sidebar options
    st.sidebar.title("Options")

    # Radio button to choose the model for Image Super Resolution
    model_name = st.sidebar.radio(
        "Choose Model for Image Super Resolution",
        ("HybridGAN model ✅", "PSNR model ✅"),
    )
    st.sidebar.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("⚙️ Processing..."):
            with st.status(
                "👩🏾‍🍳 Hooray! Your image is ready for the world! ", expanded=True
            ) as status:
                display_image_status("Model initiated")
                display_image_status("Stand up and stretch in the meantime")
                uploaded_image = os.path.abspath(
                    os.path.join(upload_path, uploaded_file.name)
                )
                downloaded_image = os.path.abspath(
                    os.path.join(download_path, f"output_{uploaded_file.name}")
                )

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
                        ".jpg": "image/jpg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".bmp": "image/bmp",
                    }
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    download_label = f"Download Enhanced Image 📷 ({file_extension})"
                    mime_type = supported_formats.get(file_extension, "image/jpeg")

                    if st.download_button(
                        label=download_label,
                        data=file,
                        file_name=f"output_{uploaded_file.name}",
                        mime=mime_type,
                    ):
                        download_success()

                # Hide the spinner
                st.spinner(False)
    else:
        st.warning("⚠ Please upload your Image file")


def bread_model_page():
    st.title(":rainbow[Upload->Enhance->Enjoy]🌈")
    st.write(
        "Integrating LIME, MSRNet, DRBL, and EG to excel in low-light image enhancement."
    )
    # File uploader to upload the image for super resolution
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["png", "jpg", "bmp", "jpeg"]
    )
    gamma = st.sidebar.number_input("Gamma Correction", value=1.0)
    strength = st.sidebar.number_input("Denoising Strength", value=0.05)

    # Set your Replicate API token
    token = ""
    os.environ["REPLICATE_API_TOKEN"] = token

    # Inputs for the Bread model
    if uploaded_file is not None:
        with st.spinner("⚙️Processing..."):
            output = replicate.run(
                "mingcv/bread:",
                input={"image": uploaded_file, "gamma": gamma, "strength": strength},
            )

            st.image(output, caption="Enhanced Image", use_column_width=True)

            # Provide download button for the enhanced image
            enhanced_image_filename = f"enhanced_{uploaded_file.name}"
            enhanced_image_extension = os.path.splitext(uploaded_file.name)[1].lower()
            enhanced_image_data = output.encode("utf-8")

            st.download_button(
                label=f"Download Enhanced Image 📷 ({enhanced_image_extension})",
                data=enhanced_image_data,
                file_name=enhanced_image_filename,
                mime=f"image/{enhanced_image_extension[1:]}",  # Remove the dot from the extension
            )

    else:
        st.warning("⚠ Please upload your Image file")


# Gallery Page
def gallery_page():
    st.title(":rainbow[Enhanced Image Gallery ]🖼️")
    st.write(
        "A visual symphony showcasing enhanced images that redefine ordinary into extraordinary."
    )

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
            groups.append(view_images[i : i + n])
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
                    file_name=os.path.basename(image_file),
                )
    else:
        st.warning("No enhanced images available.")


# HybridNet2
def edit_page():
    st.title(":rainbow[Upload->Enhance->Enjoy]🌈")
    st.write(
        "Integrating LIME, MSRNet, DRBL, and EG to excel in low-light image enhancement."
    )

    # File uploader to upload the image for editing
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["png", "jpg", "bmp", "jpeg"]
    )

    # Slider controls for image editing parameters
    brightness = st.sidebar.slider(
        "Brightness", min_value=-100, max_value=100, value=100
    )
    exposure = st.sidebar.slider("Exposure", min_value=-100, max_value=100, value=100)
    contrast = st.sidebar.slider("Contrast", min_value=-100, max_value=100, value=0)
    highlights = st.sidebar.slider(
        "Highlights", min_value=-100, max_value=100, value=100
    )
    shadows = st.sidebar.slider("Shadows", min_value=-100, max_value=100, value=100)

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("⚙️ Processing..."):
            uploaded_image = os.path.abspath(
                os.path.join(upload_path, uploaded_file.name)
            )
            edited_image = os.path.abspath(
                os.path.join(download_path, f"edited_{uploaded_file.name}")
            )

            # Perform image editing with specified parameters
            edit_image(
                uploaded_image,
                edited_image,
                brightness,
                exposure,
                contrast,
                highlights,
                shadows,
            )

            # Display the original and edited images side by side
            col1, col2 = st.columns(2)

            # Add image comparison
            image_comparison(
                img1=uploaded_image,
                img2=edited_image,
                label1="Original Image",
                label2="Enhanced Image",
            )

            # Provide download options for different image formats
            with open(edited_image, "rb") as file:
                supported_formats = {
                    ".jpg": "image/jpg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".bmp": "image/bmp",
                }
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                download_label = f"Download Edited Image 📷 ({file_extension})"
                mime_type = supported_formats.get(file_extension, "image/jpeg")

                if st.download_button(
                    label=download_label,
                    data=file,
                    file_name=f"edited_{uploaded_file.name}",
                    mime=mime_type,
                ):
                    st.success("Download Successful :)")

            # Hide the spinner
            st.spinner(False)
    else:
        st.warning("⚠ Please upload your Image file")


def edit_image(
    input_path, output_path, brightness, exposure, contrast, highlights, shadows
):
    # Implement the image editing logic using specified parameters
    # You can use libraries like PIL or OpenCV for image processing
    # Example using PIL:
    image = Image.open(input_path)

    # Apply enhancements
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance((brightness + 100) / 100)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance((contrast + 100) / 100)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance((exposure + 100) / 100)

    # Simple adjustment of highlights and shadows
    image = ImageEnhance.Brightness(image).enhance((highlights + 100) / 100)
    image = ImageEnhance.Brightness(image).enhance((shadows + 100) / 100)

    image.save(output_path)


# Edit Page
def display_image_editor():
    st.title(":rainbow[Upload->Edit->Enjoy]🌈")
    st.write(
        "Integrating LIME, MSRNet, DRBL, and EG to excel in low-light image enhancement."
    )

    # File uploader to upload the image for editing
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["png", "jpg", "bmp", "jpeg"]
    )

    # Slider controls for image editing parameters
    brightness = st.sidebar.slider("Brightness", min_value=-100, max_value=100, value=0)
    exposure = st.sidebar.slider("Exposure", min_value=-100, max_value=100, value=0)
    contrast = st.sidebar.slider("Contrast", min_value=-100, max_value=100, value=0)
    highlights = st.sidebar.slider("Highlights", min_value=-100, max_value=100, value=0)
    shadows = st.sidebar.slider("Shadows", min_value=-100, max_value=100, value=0)

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("⚙️ Processing..."):
            uploaded_image = os.path.abspath(
                os.path.join(upload_path, uploaded_file.name)
            )
            edited_image = os.path.abspath(
                os.path.join(download_path, f"edited_{uploaded_file.name}")
            )

            # Perform image editing with specified parameters
            apply_image_edits(
                uploaded_image,
                edited_image,
                brightness,
                exposure,
                contrast,
                highlights,
                shadows,
            )

            # Display the original and edited images side by side
            col1, col2 = st.columns(2)

            # Add image comparison
            image_comparison(
                img1=uploaded_image,
                img2=edited_image,
                label1="Original Image",
                label2="Enhanced Image",
            )

            # Provide download options for different image formats
            with open(edited_image, "rb") as file:
                supported_formats = {
                    ".jpg": "image/jpg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".bmp": "image/bmp",
                }
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                download_label = f"Download Edited Image 📷 ({file_extension})"
                mime_type = supported_formats.get(file_extension, "image/jpeg")

                if st.download_button(
                    label=download_label,
                    data=file,
                    file_name=f"edited_{uploaded_file.name}",
                    mime=mime_type,
                ):
                    st.success("Download Successful :)")

            # Hide the spinner
            st.spinner(False)
    else:
        st.warning("⚠ Please upload your Image file")


def apply_image_edits(
    input_path, output_path, brightness, exposure, contrast, highlights, shadows
):
    # Implement the image editing logic using specified parameters
    # You can use libraries like PIL or OpenCV for image processing
    # Example using PIL:
    image = Image.open(input_path)

    # Apply enhancements
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance((brightness + 100) / 100)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance((contrast + 100) / 100)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance((exposure + 100) / 100)

    # Simple adjustment of highlights and shadows
    image = ImageEnhance.Brightness(image).enhance((highlights + 100) / 100)
    image = ImageEnhance.Brightness(image).enhance((shadows + 100) / 100)

    image.save(output_path)


# Text2Image page

# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# Resources text, link, and logo
replicate_text = ""
replicate_link = ""
replicate_logo = ""

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application,
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):

            with st.expander(":rainbow[**Refine your output here**]"):
                # Advanced Settings (for the curious minds!)
                width = st.number_input("Width of output image", value=1024)
                height = st.number_input("Height of output image", value=1024)
                num_outputs = st.slider(
                    "Number of images to output", value=1, min_value=1, max_value=4
                )
                scheduler = st.selectbox(
                    "Scheduler",
                    (
                        "DDIM",
                        "DPMSolverMultistep",
                        "HeunDiscrete",
                        "KarrasDPM",
                        "K_EULER_ANCESTRAL",
                        "K_EULER",
                        "PNDM",
                    ),
                )
                num_inference_steps = st.slider(
                    "Number of denoising steps", value=50, min_value=1, max_value=500
                )
                guidance_scale = st.slider(
                    "Scale for classifier-free guidance",
                    value=7.5,
                    min_value=1.0,
                    max_value=50.0,
                    step=0.1,
                )
                prompt_strength = st.slider(
                    "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)",
                    value=0.8,
                    max_value=1.0,
                    step=0.1,
                )
                refine = st.selectbox(
                    "Select refine style to use (left out the other 2)",
                    ("expert_ensemble_refiner", "None"),
                )
                high_noise_frac = st.slider(
                    "Fraction of noise to use for `expert_ensemble_refiner`",
                    value=0.8,
                    max_value=1.0,
                    step=0.1,
                )
            prompt = st.text_area(
                ":orange[**Enter prompt: start typing, Shakespeare ✍🏾**]",
                value="An astronaut riding a rainbow unicorn, cinematic, dramatic",
            )
            negative_prompt = st.text_area(
                ":orange[**Party poopers you don't want in image? 🙅🏽‍♂️**]",
                value="the absolute worst quality, distorted features",
                help="This is a negative prompt, basically type what you don't want to see in the generated image",
            )

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True
            )

        return (
            submitted,
            width,
            height,
            num_outputs,
            scheduler,
            num_inference_steps,
            guidance_scale,
            prompt_strength,
            refine,
            high_noise_frac,
            prompt,
            negative_prompt,
        )


def main_page(
    submitted: bool,
    width: int,
    height: int,
    num_outputs: int,
    scheduler: str,
    num_inference_steps: int,
    guidance_scale: float,
    prompt_strength: float,
    refine: str,
    high_noise_frac: float,
    prompt: str,
    negative_prompt: str,
) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        scheduler (str): Scheduler type for the model.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        prompt_strength (float): Prompt strength when using img2img/inpaint.
        refine (str): Refine style to use.
        high_noise_frac (float): Fraction of noise to use for `expert_ensemble_refiner`.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
    if submitted:
        with st.status(
            "👩🏾‍🍳 Whipping up your words into art...", expanded=True
        ) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        output = replicate.run(
                            REPLICATE_MODEL_ENDPOINTSTABILITY,
                            input={
                                "prompt": prompt,
                                "width": width,
                                "height": height,
                                "num_outputs": num_outputs,
                                "scheduler": scheduler,
                                "num_inference_steps": num_inference_steps,
                                "guidance_scale": guidance_scale,
                                "prompt_stregth": prompt_strength,
                                "refine": refine,
                                "high_noise_frac": high_noise_frac,
                            },
                        )
                        if output:
                            st.toast("Your image has been generated!", icon="😍")
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(
                                        image,
                                        caption="Generated Image 🎈",
                                        use_column_width=True,
                                    )
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

                        # Create a BytesIO object
                        zip_io = io.BytesIO()

                        # Download option for each image
                        with zipfile.ZipFile(zip_io, "w") as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Write each image to the zip file with a name
                                    zipf.writestr(f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}",
                                        icon="🚨",
                                    )
                        # Create a download button for the zip file
                        st.download_button(
                            ":red[**Download All Images**]",
                            data=zip_io.getvalue(),
                            file_name="output_files.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                status.update(
                    label="✅ Images generated!", state="complete", expanded=False
                )
            except Exception as e:
                print(e)
                st.error(f"Encountered an error: {e}", icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass

    # Gallery display for inspo
    with gallery_placeholder.container():
        st.title(":rainbow[Imagine->Prompt->Download]🌈")
        st.write(
            "Text2Image: Unveiling visual serenity through the graceful dance of pixels using stable diffusion."
        )
        img = image_select(
            label="Like what you see? Right-click and save! It's not stealing if we're sharing! 😉",
            images=[
                "gallery/farmer_sunset.png",
                "gallery/astro_on_unicorn.png",
                "gallery/friends.png",
                "gallery/wizard.png",
                "gallery/puppy.png",
                "gallery/cheetah.png",
                "gallery/viking.png",
            ],
            captions=[
                "A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
                "A close-up portrait of a bearded viking warrior in a horned helmet. He stares intensely into the distance while holding a battle axe. Dramatic mood lighting, digital oil painting",
            ],
            use_container_width=True,
        )


def text2image_main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    (
        submitted,
        width,
        height,
        num_outputs,
        scheduler,
        num_inference_steps,
        guidance_scale,
        prompt_strength,
        refine,
        high_noise_frac,
        prompt,
        negative_prompt,
    ) = configure_sidebar()
    main_page(
        submitted,
        width,
        height,
        num_outputs,
        scheduler,
        num_inference_steps,
        guidance_scale,
        prompt_strength,
        refine,
        high_noise_frac,
        prompt,
        negative_prompt,
    )


# Image2QRCODE
config = {}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()


def generate_qrcode(data, output_filename):
    qr = segno.make(data)
    qr.save(output_filename, scale=15)


def qr_code_page():
    st.title(":rainbow[Upload->Scan->Download]🌈")
    st.write(
        "Transform your images into dynamic QR codes for seamless secured sharing."
    )

    # Upload Image
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image to upload", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        # Display uploaded image in column 1
        col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Upload Image to Firebase Storage
        storage.child(uploaded_file.name).put(uploaded_file)

        with col2:
            st.spinner("Generating...")

            # Generate QR code for the uploaded image URL
            image_url = storage.child(uploaded_file.name).get_url(
                token=None
            )  # Pass token as None

            # Generate QR code from image URL
            qr_code_filename = "output_qr.png"
            generate_qrcode(image_url, qr_code_filename)

            st.image(
                qr_code_filename, caption="QR Code for Image URL", use_column_width=True
            )


# Main App
def main():
    # Sidebar options
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "HybridGAN",
            "HybridNet1",
            "HybridNet2",
            "Gallery",
            "Text2Image",
            "Image2QRCODE",
            "Edit",
        ],
    )

    if page == "Home":
        home_page()
    elif page == "HybridGAN":
        result_comparison_page()
    elif page == "Gallery":
        gallery_page()
    elif page == "HybridNet1":
        bread_model_page()
    elif page == "HybridNet2":
        edit_page()
    elif page == "Edit":
        display_image_editor()
    elif page == "Text2Image":
        text2image_main()
    elif page == "Image2QRCODE":
        qr_code_page()
    else:
        st.warning("Invalid page selection.")


# Authentication
authenticated = st.sidebar.checkbox("Unlock Sidebar")

# Display login or logout options based on authentication status
if not authenticated:
    st.title(":rainbow[Login to your PiX3L Account :])")
    username = st.text_input("Username: ")
    password = st.text_input("Password: ", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success(f"Welcome, {username}! 🎉")
            authenticated = True
        else:
            st.error("Invalid credentials. Please try again.")

# Display content only if the user is authenticated
if authenticated:
    main()
else:
    st.warning("Please login to access PiX3L.")
