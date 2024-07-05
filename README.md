# Text Summarization Project

This project implements a transformer-based model for text summarization using the CNN/Daily Mail or XSum dataset. The models used include BERTSUM and T5.

## Setup

1. Required Libraries
   ```bash
     transformers
     datasets
     torch
     pandas
     numpy
     matplotlib
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
