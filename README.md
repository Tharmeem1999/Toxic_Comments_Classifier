# Toxic Comments Classifier

## Overview

This project is a deep learning-based multi-label classification model designed to detect toxic comments from online discussions. Using TensorFlow and an LSTM architecture, the model classifies comments into six categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The dataset is sourced from the Kaggle Toxic Comment Classification Challenge (train.csv, not included in this repo—download it separately). The project includes data preprocessing, model training, evaluation using ROC-AUC scores, and an interactive Gradio interface for testing predictions. This serves as an introduction to NLP tasks in deep learning, handling imbalanced data, and deploying models for real-time inference.

## Project Steps

Here’s a step-by-step guide to setting up and running the project. This project was developed in Google Colab to leverage free GPU resources, which are crucial for deep learning tasks as training on CPU can take hours or even days for large datasets.

1. **Set Up Environment**

   - Use Google Colab for development: Open Google Colab and upload the `DL_Project_Toxic_Comments.ipynb` notebook.
   - Enable GPU acceleration: Go to Runtime &gt; Change runtime type &gt; Select GPU (e.g., T4 GPU). This significantly speeds up model training.
   - If running locally (e.g., on Jupyter Notebook or VS Code), ensure you have a compatible GPU and CUDA installed for TensorFlow to use hardware acceleration.

2. **Download and Load Data**

   - Download the dataset from Kaggle Toxic Comment Classification Challenge.
   - Place `train.csv` in your working directory or upload it to Colab.
   - Load the data using pandas: `df = pd.read_csv('train.csv')`.

3. **Install Dependencies (For Local Run)**\
   Google Colab comes with many pre-installed packages, but if running locally, install the required libraries via pip:

   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn gradio
   ```

   Note: TensorFlow includes Keras, so no separate installation is needed.

4. **Data Preprocessing**

   - Explore the dataset: Check for class imbalances and preview comments.
   - Tokenize text: Use `Tokenizer` from TensorFlow to convert comments into sequences (limited to 20,000 words).
   - Pad sequences: Ensure uniform input length (max 200 words) using `pad_sequences`.
   - Split data: Use `train_test_split` from scikit-learn for 90/10 train-validation split.

5. **Model Building and Training**

   - Build an LSTM model: Embedding layer (128 dimensions), Bidirectional LSTM (64 units), Dense layers with sigmoid activation for multi-label output.
   - Compile: Use binary cross-entropy loss and Adam optimizer.
   - Train: Fit the model for 2 epochs with batch size 32, monitoring validation loss.

6. **Evaluation**

   - Predict on validation set and compute ROC-AUC scores for each label using `roc_auc_score` from scikit-learn.
   - Visualize results: Plot ROC curves with matplotlib.

7. **Interactive Testing**

   - Define a scoring function to predict toxicity labels for new comments.
   - Launch a Gradio interface: Input a comment and get boolean predictions for each toxicity category.
   - Run `interface.launch(share=True)` to generate a public link for sharing.

8. **Run the Notebook**

   - Execute cells sequentially in Colab or locally.
   - For local runs without GPU, reduce batch size or epochs to manage training time.

## Python Packages Used

The following Python packages are used in this project:

- **tensorflow**: For building, training, and deploying the LSTM model (includes Keras for neural networks).
- **pandas**: For data loading and manipulation.
- **numpy**: For numerical operations and array handling.
- **matplotlib**: For plotting ROC curves and visualizations.
- **scikit-learn**: For data splitting and evaluation metrics.
- **gradio**: For creating an interactive web interface to test the model.

For a complete list, refer to the notebook imports. If running locally, install via the command in the Project Steps.

## Additional Information

- **Dataset**: The project uses `train.csv` from Kaggle's Toxic Comment Classification Challenge, containing \~160,000 comments labeled for toxicity. Download it and place it in the project directory. Note: The dataset may contain offensive language—handle with care.

- **Why Google Colab?**: Deep learning models like LSTM benefit from GPU acceleration. Colab provides free access to GPUs, reducing training time from hours to minutes. For local setups, ensure TensorFlow-GPU is installed and configured.

- **Challenges and Learnings**: This project highlights handling multi-label classification, sequence padding in NLP, and evaluating imbalanced datasets. It's a great starter for NLP in deep learning.
