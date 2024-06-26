import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import matplotlib
from PIL import Image, ImageOps
from scipy.interpolate import splev, splprep
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="Crowding Assessment",
    page_icon=":tooth:",
)

# Define the target width and height for resizing
target_width = 224
target_height = 224

# Define a function to load and process an image and its keypoints
def load_and_process(image_file):
    # Read the uploaded file into a PIL image
    image = Image.open(image_file)

    # Apply any transformations specified in the metadata
    image = ImageOps.exif_transpose(image)

    # Convert the PIL image to a numpy array
    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check if the image is RGBA
    if image.shape[2] == 4:
        st.error('RGBA image detected, please ensure the image is in RGB format!', icon="🚨")
    # Get the original width and height of the image
    original_width = image.shape[1]
    original_height = image.shape[0]
    # Compute the scaling factors for width and height based on the smaller dimension
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)

    # Compute the new width and height after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image using cv2.resize function with interpolation=cv2.INTER_AREA
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Crop the resized image to match the target width and height from the center
    cropped_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    cropped_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # Normalize the cropped image pixels to range [0, 1]
    normalized_image = cropped_image / 255.0

    return tf.convert_to_tensor(normalized_image)

def visualize_keypoints(images, keypoints):
    # Define a list of 32 colors for each keypoint
    colors = list(matplotlib.colors.CSS4_COLORS.keys())[:32]

    # Loop through the images and keypoints tensors
    for i in range(len(images)):
        # Get the image and keypoints tensors for the i-th example
        image = images[i]
        keypoint = keypoints[i]

        # Convert the tensors to numpy arrays
        if tf.is_tensor(image):
          image = image.numpy()

        # Denormalize the image pixels to range [0, 255]
        image = image * 255.0

        # Plot the image and the keypoints
        plt.figure(figsize=(10, 10))
        plt.imshow(image.astype(np.uint8))

        # Get the N value from the shape of the keypoints array
        N = keypoint.shape[0]

        # Define a list of labels for each keypoint
        labels = [str(i + 1) for i in range(N)]

        for j in range(len(keypoint)):
            # Get the coordinates and visibility of the keypoint
            x, y = keypoint[j]
            plt.scatter(x, y, s=100, c=colors[j], marker="o")
            plt.text(x + 5, y + 5, labels[j], fontsize=12, color=colors[j])
        plt.show()
        st.pyplot()

def denormalize_keypoints(predictions, target_width, target_height):
    denormalzied_keypoints = predictions.numpy()
    denormalzied_keypoints = denormalzied_keypoints * np.array([target_width, target_height])
    # # Define a boolean mask to filter out rows that contain 0s
    # mask = (keypoint != 0).all(axis=1)

    # # Apply the mask to the keypoint array
    # keypoint = keypoint[mask]
    return denormalzied_keypoints

def B_spline_curve_fitting_visualize(images, keypoints):
    # Define a list of 32 colors for each keypoint
    colors = list(matplotlib.colors.CSS4_COLORS.keys())[:32]

    x_list, y_list = [], []

    # Loop through the images and keypoints tensors
    for i in range(len(images)):
        # Get the image and keypoints tensors for the i-th example
        image = images[i]
        keypoint = keypoints[i]

        # Convert the tensors to numpy arrays
        if tf.is_tensor(image):
          image = image.numpy()
        
        # Denormalize the image pixels to range [0, 255]
        image = image * 255.0

        # Plot the image and the keypoints
        plt.figure(figsize=(10, 10))
        plt.imshow(image.astype(np.uint8))

        # Get the N value from the shape of the keypoints array
        N = keypoint.shape[0]

        # Define a list of labels for each keypoint
        labels = [str(i + 1) for i in range(N)]

        for j in range(len(keypoint)):
            # Get the coordinates of the keypoint
            x, y = keypoint[j]
            plt.scatter(x, y, s=100, c=colors[j], marker="o")
            plt.text(x + 5, y + 5, labels[j], fontsize=12, color=colors[j])

        # Set a distance threshold
        threshold = 50

        # Calculate the Euclidean distance between each keypoint and the origin
        distances = np.sqrt(np.sum(keypoint ** 2, axis=1))

        # Find the index of the first keypoint that is below the distance threshold
        n_valid = np.argmax(distances < threshold)

        # If no keypoints are below the threshold, set n_valid to the number of keypoints
        if n_valid == 0:
            n_valid = keypoint.shape[0]

        with st.expander("Adjust curve"):
            smoothness = st.slider("Smoothness of curve", 300, 3000, value = 1000)
            coeff = st.slider("Curve size", -20.0, 20.0, value = -5.0)
            middle = st.slider("Midline point", -20.0, 20.0, value = 0.0)
            starting_point = st.slider("Starting point", -20.0, 20.0, value = 0.0)
            ending_point = st.slider("Ending point", -20.0, 20.0, value = 0.0)

        # Fit a B-spline curve to the first n_valid keypoints
        # tck, u = splprep(keypoint[:n_valid].T, u=None, s=smoothness)
        tck, u = splprep(keypoint.T, u=None, s=smoothness)
        u_new = np.linspace(u.min(), u.max(), 1000)
        # x_new, y_new = splev(u_new, tck)
        c_new = tck[1].copy()
        c_new[0][0] -= coeff # left x
        c_new[0][1] -= coeff # left upper x
        c_new[0][-2] += coeff # right upper x
        c_new[0][-1] += coeff # right x

        # adjust curve for middle point
        if (len(c_new[1]) % 2 == 0):
            middle_index1 = int(len(c_new[1])/2)
            middle_index2 = int(len(c_new[1])/2 - 1)
            c_new[1][middle_index1] -= coeff
            c_new[1][middle_index2] -= coeff
            c_new[1][middle_index1] -= middle
            c_new[1][middle_index2] -= middle
        else:
            middle_index = int(len(c_new[1])//2)
            c_new[1][middle_index] -= middle
            c_new[1][middle_index] -= coeff

        # adjust start point and end point
        c_new[1][0] += starting_point
        c_new[1][-1] += ending_point

        x_new, y_new = splev(u_new, (tck[0], c_new, tck[2]))
            
        # Plot the B-spline curve
        plt.plot(x_new, y_new)
        st.pyplot()
        x_list.append(x_new)
        y_list.append(y_new)

    return x_list, y_list

def calculate_tooth_width(keypoints):
    # Reshape the tensor to shape (14, 2, 2)
    reshaped_keypoints = tf.reshape(keypoints, (14, 2, 2))
    denormalized_reshaped_keypoints = reshaped_keypoints.numpy() * np.array([target_width, target_height])

    # Calculate the Euclidean distance for each tooth
    tooth_lengths_pixels = tf.norm(denormalized_reshaped_keypoints[:, 0, :] - denormalized_reshaped_keypoints[:, 1, :], axis=1)
    return tooth_lengths_pixels.numpy()

def check_concave_up_curve(keypoints):
    keypoint = keypoints[0]
    tck, u = splprep(keypoint.T, u=None, s=2000)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x2_new, y2_new = splev(u_new, tck, der=2)

    # if second derivative is always negative, means the curve is concave down
    # but since the y-axis increases downwards (computer graphics coordinate system), it will give opposite result, so i flip the sign
    if np.all(y2_new > 0):
        #  curve is concave down or n shape
        return False
    else:
        #  curve is concave up or u shape
        return True

st.sidebar.header("Note :warning:")
st.sidebar.write("1. Please upload image with 14 teeth only, as the current model only supports complete dentition excluding wisdom teeth.")
st.sidebar.write("2. For best result, please ensure the whole dental is centered at the middle.")
st.sidebar.image('sample.JPG', caption="Example image")

# Models are stored in google drive as streamlit deployment doesn't support file > 100MB
import gdown
# Define a function to download the model from the gdrive link
@st.cache_resource
def download_models():
    model_link = 'https://drive.google.com/uc?id=149JqmcSDXL2EANEXn_HWJiak6DTCTsdX'
    filtering_model_link = 'https://drive.google.com/uc?id=1QGiGq3LvuoxF7k9T6ezkC4KEasVruDtt'

    model_file_name = "intraoral_modelv2.h5"
    filtering_model_file_name = "filtering_modelv1.h5"

    # Use gdown to download the file from the gdrive link
    gdown.download(model_link, model_file_name, quiet =True, fuzzy = True, use_cookies = True)
    gdown.download(filtering_model_link, filtering_model_file_name, quiet =True, fuzzy = True, use_cookies = True)
    # Return the file name
    return model_file_name, filtering_model_file_name

# Define a function to load the model using keras
@st.cache_resource
def load_model(file_name):
  # Load the model from the local file
  model = tf.keras.models.load_model(file_name)
  # Return the model
  return model

@st.cache_resource
def load_filtering_model(file_name):
  # Load the model from the local file
  filtering_model = tf.keras.models.load_model(file_name, compile = False)
  filtering_model.compile(loss='mean_squared_error', optimizer='adam')
  # Return the model
  return filtering_model

# Call the download_model function and get the file name
model_file_name, filtering_model_file_name = download_models()
# Call the load_model function and get the model
model = load_model(model_file_name)
filtering_model = load_filtering_model(filtering_model_file_name)

image = st.file_uploader("Upload an intraoral photograph", type = ['png', 'jpg'])
st.set_option('deprecation.showPyplotGlobalUse', False)
if image is not None:
    image =  load_and_process(image)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

        # Check if the image has changed
        if 'last_image' not in st.session_state or (not (np.array_equal(image, st.session_state.last_image)) and not (np.array_equal(np.flip(image, axis=1), st.session_state.last_image))):
            for key in st.session_state.keys():
                del st.session_state[key]

            # validate image using filtering model
            filtering_predictions = filtering_model.predict(image)
            filtering_predictions = tf.reshape(filtering_predictions, (image.shape[0], -1, 2))
            # denormalize predictions
            denorm_filtering_keypoints = denormalize_keypoints(filtering_predictions, target_width, target_height)
            # validate if intraoral if detected from the input image
            # if first point is close to last point or points are close to each other, most likely the input image is not intraoral

            p1 = denorm_filtering_keypoints[0][0]
            p2 = denorm_filtering_keypoints[0][-1]
            distance_p1p2 = np.linalg.norm(p2 - p1)

            # if sum_tooth_widths is small means points close to each other
            tooth_lengths_pixels = calculate_tooth_width(filtering_predictions)
            sum_tooth_widths = sum(tooth_lengths_pixels)

            if ((distance_p1p2 < 50) or (sum_tooth_widths < 120)):
                st.error('Intraoral is not detected, please upload another image', icon="🚨")
                st.stop()
            
            if check_concave_up_curve(denorm_filtering_keypoints):
                # flip image vertically if concave up curve detected
                image = np.flip(image, axis=1)
                st.info("Image is vertically flipped")

            # prediction
            predictions = model.predict(image)
            predictions = tf.reshape(predictions, (image.shape[0], -1, 2))

            # Store the result in session state
            st.session_state.predictions = predictions
            st.session_state.last_image = image
        else:
            # Use the stored result
            predictions = st.session_state.predictions
            image = st.session_state.last_image

        if 'denormalized_predictions' not in st.session_state:
            st.session_state.denormalized_predictions = denormalize_keypoints(predictions, target_width, target_height)

        with st.expander("Adjust keypoints"):
            option = st.selectbox('Choose a point to adjust',[i for i in range(1, 29)])
            image_adjust = image[0] * 255.0
            st.write(f"Click on the image below to update the new position for point {option}")
            adjusted_point = streamlit_image_coordinates(
                image_adjust.astype(np.uint8),
                key=f"numpy_{option}",
                height = 672,
                width = 672, # size of image is tripled
            )
            if adjusted_point is not None:
                st.session_state.denormalized_predictions[0, option-1, 0] = adjusted_point['x']/3 #divide by 3 to obtain original point
                st.session_state.denormalized_predictions[0, option-1, 1] = adjusted_point['y']/3

        # display result
        x_list, y_list = B_spline_curve_fitting_visualize(image, st.session_state.denormalized_predictions)

        with st.form(key='my_form'):
            actual_length = st.number_input("Input central incisor tooth width (mm)", min_value = 0.0, value = None, placeholder = "Type a number...")
            submit_button = st.form_submit_button(label='Check result')
            if submit_button:
                
                # do calculations
                tooth_lengths_pixels = calculate_tooth_width(predictions)
                image_length = tooth_lengths_pixels[6]
                scale_factor = actual_length / image_length
                tooth_lengths_actual = tooth_lengths_pixels * scale_factor
                sum_tooth_widths = sum(tooth_lengths_actual)
                st.write("Sum of tooth widths: {:.2f} mm".format(sum_tooth_widths))
                curve_length_actual = 0
                for x, y in zip(x_list,y_list):
                    points = np.column_stack((x, y))
                    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
                    curve_length_pixels = np.sum(distances)
                    curve_length_actual = curve_length_pixels * scale_factor

                st.write("Arch form length: {:.2f} mm".format(curve_length_actual))
                st.write("Crowding: {:.2f} mm".format(sum_tooth_widths-curve_length_actual))

