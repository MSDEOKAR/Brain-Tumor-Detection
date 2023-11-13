# Brain-Tumor-Detection

<h1>Purpose: This code aims to build a deep learning model for detecting brain tumors in MRI images.</h1>

1) Data Preparation
<ul>
<li>Load images from the 'dataset' directory.
<li>separate images into 'no' and 'yes' tumor categories.
<li>Resize all images to a uniform size of 64x64 pixels.
<li>Convert images to NumPy arrays.
<li>Create a dataset and corresponding label arrays.
</ul>
<br>
2) Model Building:
<ul>
<li>Define a sequential convolutional neural network (CNN) model.
<li>Add convolutional layers with 32 filters, ReLU activation, and 2x2 max-pooling.
<li>Flatten the output of the convolutional layers.
<li>Add a fully connected layer with 64 units and ReLU activation.
<li>Add a dropout layer with a 0.5 dropout rate.
<li>Add a final output layer with two units and softmax activation for binary classification.
<li>Compile the model using categorical cross-entropy loss, the Adam optimizer, and accuracy metrics.
</li>
</ul>
  
3) Training and Evaluation:
<ul>
  <li>Split the dataset into training and testing sets (80% and 20%, respectively).
<li>Normalize the training and testing data.
<li>Convert labels into one-hot encoded vectors.
<li>Train the model using the training data and evaluate its performance on the testing data.
<li>Save the trained model as 'BrainTumor10Epochs.keras'.</ul>

4) Key Points:
<ul>
<li>The model utilizes CNNs to extract relevant features from MRI images.
<li>The model employs dropout to prevent overfitting and improve generalization.
<li>The model is trained for 10 epochs.
<li>The trained model is saved for future use.
</ul>




