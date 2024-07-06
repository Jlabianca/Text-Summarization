# Text Summarization Project

The Text Summarization Project uses transformer-based models to generate summaries for long texts, such as articles. This project specifically utilizes the T5 (Text-to-Text Transfer Transformer) model, which is designed to handle various text generation tasks, including summarization.

## The project involves several key steps:

1. Data Loading and Preprocessing: Loading the CNN/Daily Mail dataset and preparing it for training.
2. Model Initialization: Setting up the T5 model and tokenizer.
3. Training the Model: Training the model on the dataset to learn how to generate summaries.
4. Evaluating the Model: Evaluating the model's performance using metrics like ROUGE.
5. Inference: Generating summaries for new, unseen articles.

## Setup

1. Required Libraries
   ```bash
     transformers
     datasets
     torch
     pandas
     numpy
     matplotlib
     wordcloud
   ```

2. Clone the repository:
    ```bash
    git clone <repository-url>
    cd text_summarization_project
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Data Preprocessing**:
    Run the data preprocessing script:
    ```bash
    python data/preprocess.py
    ```

2. **Training**:
    Train the model:
    ```bash
    python scripts/train.py
    ```

3. **Evaluation**:
    Evaluate the trained model:
    ```bash
    python scripts/evaluate.py
    ```

4. **Inference**:
    Generate summaries for new documents:
    ```bash
    python scripts/inference.py
    ```

## Project Structure

- **data/**: Data preprocessing scripts.
- **models/**: Model definition and training scripts.
- **notebooks/**: Jupyter notebooks for data exploration.
- **scripts/**: Scripts for training, evaluation, and inference.
- **utils/**: Utility functions for data collation, metrics, and tokenization.
- **requirements.txt**: List of required Python packages.
- **README.md**: Project documentation.

## Authors

- Joseph LaBianca

## License

This project is licensed under the MIT License.
