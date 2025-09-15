# Toxic Comment Classification using Deep Learning

## Overview 📖

This project focuses on building a deep learning model to detect and classify different types of toxicity in online comments. The model is designed to handle multi-label classification, identifying whether a comment falls into one or more of six categories: **toxic**, **severe toxic**, **obscene**, **threat**, **insult**, and **identity hate**.

The core of this project is a neural network built with TensorFlow/Keras, featuring an `Embedding` layer and a `Bidirectional LSTM` to effectively process the sequential nature of text data. The final model is deployed in a simple interactive web interface using Gradio.

-----

## 💻 Environment and Setup

This project was developed using Google Colab to leverage its free access to high-performance resources, especially GPUs.

### Why Google Colab?

Training deep learning models on large datasets is computationally intensive. A **GPU (Graphics Processing Unit)** can drastically reduce training time from hours to minutes. Google Colab provides a convenient, pre-configured environment with free GPU access, making it ideal for projects like this.

### Local Setup Instructions

If you wish to run this project on your local machine, you'll need to install the necessary Python packages.

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install the required packages:**
    Google Colab comes with many libraries pre-installed. For a local setup, you'll need to install the following:

    ```bash
    pip install tensorflow pandas matplotlib gradio
    ```

-----

## 🛠️ Project Steps

The project follows a standard machine learning workflow, from data preparation to model deployment.

1.  **Data Loading and Exploration**: The dataset (`train.csv`) is loaded into a pandas DataFrame. The features (`comment_text`) and the multi-label targets are separated.

2.  **Text Preprocessing**: The core of the preprocessing is the `TextVectorization` layer from TensorFlow. This layer handles:

      * **Vocabulary Creation**: It builds a vocabulary of the top 200,000 most frequent words from the comments.
      * **Integer Encoding**: It converts each comment into a sequence of integer IDs.
      * **Padding/Truncation**: It ensures every sequence has a uniform length of 1800 by padding shorter comments and truncating longer ones.

3.  **Building a Data Pipeline**: An efficient `tf.data.Dataset` pipeline is created to handle the data during training. This pipeline includes:

      * `.cache()`: To keep the data in memory for faster access across epochs.
      * `.shuffle()`: To randomize the data order and prevent the model from learning patterns based on data sequence.
      * `.batch()`: To group data into batches of 16 for efficient training.
      * `.prefetch()`: To prepare subsequent batches while the current one is being processed, optimizing GPU utilization.

4.  **Dataset Splitting**: The dataset is split into three parts:

      * **70%** for **Training**
      * **20%** for **Validation** (to monitor performance and prevent overfitting during training)
      * **10%** for **Testing** (to evaluate the final model's performance on unseen data)

5.  **Model Architecture**: A `Sequential` model is constructed with the following layers:

      * **Embedding Layer**: Converts the integer-encoded vocabulary into dense vectors of size 32.
      * **Bidirectional LSTM Layer**: Processes the text sequence both forwards and backward, capturing contextual information effectively.
      * **Dense Layers**: Three fully-connected layers with ReLU activation for feature extraction.
      * **Output Layer**: A final Dense layer with 6 neurons (one for each toxicity class) and a `sigmoid` activation function to output a probability between 0 and 1 for each class.

6.  **Model Training**: The model is compiled using the **`BinaryCrossentropy`** loss function (suitable for multi-label problems) and the **`Adam`** optimizer. It's then trained for 5 epochs, with performance monitored on the validation set.

7.  **Performance Evaluation**: The trained model is evaluated on the test set using `Precision`, `Recall`, and `CategoricalAccuracy` metrics to assess its real-world performance. The final evaluation metrics were:

      * **Precision**: $0.86$
      * **Recall**: $0.79$
      * **Accuracy**: $0.49$

8.  **Interactive Testing with Gradio**: A user-friendly web interface is created using the `Gradio` library. This allows anyone to input a comment and receive an instant toxicity score across the six categories.

-----

## 🚀 How to Use

To test the model with your own comments:

1.  Ensure you have completed the setup instructions.
2.  Run the `DL_Project_Toxic_Comments.ipynb` notebook in a Jupyter environment or Google Colab.
3.  The final cells of the notebook will save the trained model as `toxicity.h5` and launch the Gradio web interface.
4.  Enter any comment into the input box in the Gradio interface and click "Submit" to see the classification results.
