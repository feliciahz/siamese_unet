import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from loss import FocalLoss, bce_dice_loss

layers_to_visualize = ['conv5_1_conv2', 'conv5_1_conv2']

# Iterate over all images in the directories
for image_filename in os.listdir(image_dir1):
    # Ensure the image exists in both directories
    if os.path.exists(os.path.join(image_dir2, image_filename)):

        # Construct the full paths to the images
        input_image_path1 = os.path.join(image_dir1, image_filename)
        input_image_path2 = os.path.join(image_dir2, image_filename)

        for layer_name in layers_to_visualize:
            try:
                print(f"Processing layer: {layer_name}")

                # Iterate over the input images
                for idx, input_image_path in enumerate([input_image_path1, input_image_path2]):
                    # Extract the output of the layer we want to visualize
                    layer_output = model.get_layer(layer_name).output

                    # Create a model that returns the layer output and final output
                    model_to_explain = tf.keras.models.Model(inputs=model.inputs, outputs=[layer_output, model.output])

                    # Set the input image
                    input_image = cv2.imread(input_image_path)
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    input_image = cv2.resize(input_image, (256, 256))
                    input_image = input_image / 255.0
                    input_image = np.expand_dims(input_image, axis=0)
                    input_image = np.expand_dims(input_image, axis=-1)  # Add an extra dimension to represent the grayscale channel
                    inputs = [input_image, input_image]  # Update the inputs for the current iteration

                    with tf.GradientTape() as tape:
                        # Cast inputs to tf.float32
                        inputs = [tf.cast(img, tf.float32) for img in inputs]
                        tape.watch(inputs)

                        # Get the layer output and final model output
                        layer_output, outputs = model_to_explain(inputs)

                        # Extract the output of the last class
                        class_output = outputs[0][:, -1]

                    # Calculate the gradients of the target class output w.r.t the specific layer output
                    grads = tape.gradient(class_output, layer_output)

                    # Get the gradient for the channel 0
                    channel_grads = grads[:, :, :, 0]

                    # Pool the gradients over all the axes leaving out the channel dimension
                    pooled_grads = K.mean(channel_grads, axis=(0, 1, 2))

                    # Weigh the output feature map with the computed gradient values
                    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, layer_output), axis=-1)

                    # Normalize the heatmap between 0 & 1 for visualization
                    heatmap /= tf.reduce_max(heatmap)

                    # Ensure it's a 2D array by removing singleton dimensions
                    heatmap = np.squeeze(heatmap)

                    # Convert the heatmap to uint8
                    heatmap = np.uint8(255 * heatmap)

                    # Apply the colormap
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    # Overlay the heatmap on the original image
                    img_original = cv2.imread(input_image_path)
                    heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))

                    superimposed_img = cv2.addWeighted(img_original, 0.3, heatmap, 0.7, 0)

                    # Save the result to a file
                    output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_{layer_name}_Channel_0_GradCAM_{idx}.jpg")
                    cv2.imwrite(output_path, superimposed_img)
            except Exception as e:
                print(f"Failed to process layer {layer_name} for image {image_filename}: {str(e)}")


layers_to_visualize = ['input_1', 'batch_normalization', 'batch_normalization_9', 'batch_normalization_8', 'batch_normalization_5', 'batch_normalization_4', 'batch_normalization_1', 'batch_normalization_13', 'batch_normalization_12', 'conv5_1_dropout1', 'conv5_1_conv2', 'conv5_1_conv1', 'conv5_1_bn2', 'conv5_1_bn1', 'conv4_1_dropout1', 'conv4_1_conv2', 'conv4_1_conv1', 'conv4_1_bn2', 'conv4_1_bn1', 'conv3_1_dropout1', 'conv3_1_conv2', 'conv3_1_conv1', 'conv3_1_bn2', 'conv3_1_bn1', 'conv2_1_dropout1', 'conv2_1_conv2', 'conv2_1_conv1', 'conv2_1_bn2', 'conv2_1_bn1', 'conv1_1_dropout1', 'conv1_1_conv2', 'conv1_1_conv1', 'conv1_1_bn2', 'conv1_1_bn1']

for layer_name in layers_to_visualize:
    try:
        print(f"Processing layer: {layer_name}")

        # Iterate over the input images
        for idx, input_image_path in enumerate([input_image_path1, input_image_path2]):
            # Extract the output of the layer we want to visualize
            layer_output = model.get_layer(layer_name).output

            # Create a model that returns the layer output and final output
            model_to_explain = tf.keras.models.Model(inputs=model.inputs, outputs=[layer_output, model.output])

            # Set the input image
            input_image = cv2.imread(input_image_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            input_image = cv2.resize(input_image, (256, 256))
            input_image = input_image / 255.0
            input_image = np.expand_dims(input_image, axis=0)
            input_image = np.expand_dims(input_image, axis=-1)  # Add an extra dimension to represent the grayscale channel
            inputs = [input_image, input_image]  # Update the inputs for current iteration

            with tf.GradientTape() as tape:
                # Cast inputs to tf.float32
                inputs = [tf.cast(img, tf.float32) for img in inputs]
                tape.watch(inputs)

                # Get the layer output and final model output
                layer_output, outputs = model_to_explain(inputs)

                # Extract the output of the last class
                class_output = outputs[0][:, -1]

            # Calculate the gradients of the target class output w.r.t the specific layer output
            grads = tape.gradient(class_output, layer_output)

            # Pool the gradients over all the axes leaving out the channel dimension
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

            # Weigh the output feature map with the computed gradient values
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, layer_output), axis=-1)

            # Normalize the heatmap between 0 & 1 for visualization
            heatmap /= tf.reduce_max(heatmap)

            # Ensure it's a 2D array by removing singleton dimensions
            heatmap = np.squeeze(heatmap)

            # Convert the heatmap to uint8
            heatmap = np.uint8(255 * heatmap)

            # Apply the colormap
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Adjust the size of the heatmap to 256x256
            heatmap = cv2.resize(heatmap, (256, 256))

            # Overlay the heatmap on the original image
            img_original = cv2.imread(input_image_path)
            img_original = cv2.resize(img_original, (256, 256))
            superimposed_img = cv2.addWeighted(img_original, 0.3, heatmap, 0.7, 0)

            # Save the result to a file
            output_path = os.path.join(output_dir, f"{layer_name}_GradCAM_{idx}.jpg")
            cv2.imwrite(output_path, superimposed_img)
    except Exception as e:
        print(f"Failed to process layer {layer_name}: {str(e)}")

