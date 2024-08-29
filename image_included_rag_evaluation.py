import argparse
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


def extract_image_links(markdown: str) -> List[str]:
    """
    Extract image links from the given markdown text.

    Args:
    markdown (str): The markdown text containing image links.

    Returns:
    list: A list of extracted image links.
    """
    return re.findall(r'!\[.*?\]\((.*?)\)', markdown)

def is_valid_azure_blob_link(link: str) -> bool:
    """
    Check if the given link is a valid Azure Blob Storage URL.

    Args:
    link (str): The URL to be validated.

    Returns:
    bool: True if the link is a valid Azure Blob Storage URL, False otherwise.
    """
    return re.match(r'^https://[^.]+\.blob\.core\.windows\.net/[^/]+(/.*)?$', link) is not None

def is_link_accessible(link: str) -> bool:
    """
    Check if the given link is accessible by sending a HEAD request.

    Args:
    link (str): The URL to be checked.

    Returns:
    bool: True if the link is accessible (HTTP status code 200), False otherwise.
    """
    try:
        response = requests.head(link, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def calculate_metrics(answer_images: List[str], ground_truth_images: List[str]) -> Tuple[float, float]:
    """
    Calculate precision and recall metrics using sklearn.

    Args:
    answer_images (list): List of image links in the input answer.
    ground_truth_images (list): List of image links in the ground truth.

    Returns:
    tuple: Precision and recall scores.
    """
    true_positive = sum(1 for img in answer_images if img in ground_truth_images)
    false_positive = len(answer_images) - true_positive
    false_negative = len(ground_truth_images) - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return precision, recall

def evaluate_images(input_answer: str, ground_truth: str, documents: List[str]) -> Dict[str, Any]:
    """
    Evaluate the images in the input answer against the ground truth.

    Args:
    input_answer (str): The input answer containing markdown text with image links.
    ground_truth (str): The ground truth containing markdown text with image links.
    documents (list): The list of valid document links.

    Returns:
    dict: A dictionary containing retrieval score, precision, recall, hallucination links count, and hallucination ratio.
    """
    answer_images = extract_image_links(input_answer)
    ground_truth_images = extract_image_links(ground_truth)

    hallucination_links = {
        "broken_link": [],
        "resource_not_existing": [],
        "others": []
    }

    for link in answer_images:
        if link not in documents:
            if not is_valid_azure_blob_link(link):
                hallucination_links["broken_link"].append(link)
            elif is_valid_azure_blob_link(link) and not is_link_accessible(link):
                hallucination_links["resource_not_existing"].append(link)
            else:
                hallucination_links["others"].append(link)

    hallucination_count = sum(len(lst) for lst in hallucination_links.values())
    hallucination_ratio = hallucination_count / len(answer_images) if answer_images else 0

    if not ground_truth_images:
        return {
            "retrieval_score": None,
            "precision": None,
            "recall": None,
            "hallucination_count": hallucination_count,
            "hallucination_ratio": hallucination_ratio,
            "hallucination_broken_link": len(hallucination_links["broken_link"]),
            "hallucination_resource_not_existing": len(hallucination_links["resource_not_existing"]),
            "hallucination_others": len(hallucination_links["others"])
        }

    retrieval_score = int(answer_images == ground_truth_images)
    precision, recall = calculate_metrics(answer_images, ground_truth_images)

    return {
        "retrieval_score": retrieval_score,
        "precision": precision,
        "recall": recall,
        "hallucination_count": hallucination_count,
        "hallucination_ratio": hallucination_ratio,
        "hallucination_broken_link": len(hallucination_links["broken_link"]),
        "hallucination_resource_not_existing": len(hallucination_links["resource_not_existing"]),
        "hallucination_others": len(hallucination_links["others"])
    }

def evaluate_all_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate all rows in the DataFrame.

    Args:
    df (DataFrame): The DataFrame containing input answers and ground truths.

    Returns:
    DataFrame: A DataFrame with evaluation results for each row.
    """
    evaluation_results = df.apply(
        lambda row: evaluate_images(row['inputs.answer'], row['inputs.ground_truth'], extract_image_links(row['inputs.documents'])), axis=1
    )
    evaluation_df = pd.DataFrame(evaluation_results.tolist(), index=df.index)
    return pd.concat([df, evaluation_df], axis=1)

def calculate_average_metrics(detailed_results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate average metrics from the detailed results DataFrame.

    Args:
    detailed_results_df (DataFrame): The DataFrame containing detailed evaluation results.

    Returns:
    dict: A dictionary containing average retrieval score, precision, recall, total hallucination links, and average hallucination ratio.
    """
    filtered_df = detailed_results_df.dropna(subset=['retrieval_score'])
    average_retrieval_score = filtered_df["retrieval_score"].mean()
    average_precision = filtered_df["precision"].mean()
    average_recall = filtered_df["recall"].mean()
    total_hallucination_count = detailed_results_df["hallucination_count"].sum()
    total_answer_images = detailed_results_df["inputs.answer"].apply(extract_image_links).apply(len).sum()
    average_hallucination_ratio = total_hallucination_count / total_answer_images if total_answer_images > 0 else 0
    total_hallucination_broken_link = detailed_results_df["hallucination_broken_link"].sum()
    total_hallucination_resource_not_existing = detailed_results_df["hallucination_resource_not_existing"].sum()
    total_hallucination_others = detailed_results_df["hallucination_others"].sum()

    return {
        "average_retrieval_score": average_retrieval_score,
        "average_precision": average_precision,
        "average_recall": average_recall,
        "total_hallucination_count": total_hallucination_count,
        "total_answer_images": total_answer_images,
        "average_hallucination_ratio": average_hallucination_ratio,
        "total_hallucination_broken_link": total_hallucination_broken_link,
        "total_hallucination_resource_not_existing": total_hallucination_resource_not_existing,
        "total_hallucination_others": total_hallucination_others
    }

def main():
    """Main function to execute the evaluation process."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results including image links")
    parser.add_argument("csv_file_path", type=str, help="Path to the CSV file containing evaluation results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file_path)
    results = evaluate_all_images(df)
    detailed_results_df = pd.DataFrame(results)

    average_metrics = calculate_average_metrics(detailed_results_df)

    print(detailed_results_df)
    print("Average Metrics:", average_metrics)

if __name__ == "__main__":
    main()
