# Image-Included RAG Evaluation

This repository contains a Python script designed to evaluate the performance of an image included Retrieval-Augmented Generation (RAG) system. The script analyzes the presence and accuracy of image links within markdown text, comparing them against ground truth data to calculate various performance metrics.

## Files in the Repository

1. **image_included_rag_evaluation.py**: Main script that performs the evaluation.
2. **requirements.txt**: List of Python dependencies required to run the script.
3. **test_image_included_rag_evaluation.py**: Contains unit tests for the main script.

## Prerequisites

Ensure you have Python installed on your system. This script requires Python 3.10 or higher.

## Setup

1. **Install Dependencies**

   Use pip to install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**

   Ensure your CSV file follows the structure expected by the script. An example row in the CSV might look like this:

   | inputs.ground_truth              | inputs.answer                    | inputs.documents                 |
   | -------------------------------- | -------------------------------- | -------------------------------- |
   | This is an image. ![image](url1) | This is an image. ![image](url1) | This is an image. ![image](url1) |

   or

   | inputs.ground_truth | inputs.answer  | inputs.documents |
   | ------------------- | -------------- | ---------------- |
   | ![image](url1)      | ![image](url1) | ![image](url1)   |

## Usage

Run the `image_included_rag_evaluation.py` script to evaluate the images in your CSV file.

```sh
python image_included_rag_evaluation.py path/to/your/csvfile.csv
```

The script will load the specified CSV file and perform the following tasks:

1. Load the CSV file into a DataFrame.
2. Extract image links from the markdown text in the ground truths, input answers, and input documents.
3. Check if the links are valid Azure Blob Storage URLs and if they are accessible.
4. Calculate precision, recall, and other metrics.
5. Categorize hallucinations into broken links, non-existing resources, and others.
6. Output detailed results for each row and average metrics.

## Output

The script will print out the detailed results for each row and the average metrics, which include:

- Retrieval Score
- Precision
- Recall
- Number of Hallucination Links
- Hallucination Ratio
- Number of Broken Links
- Number of Resource-Not-Existing Links
- Number of Other Hallucination Links
