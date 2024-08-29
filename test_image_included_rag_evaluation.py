from typing import Dict, List
from unittest.mock import patch

import pytest
import requests

from image_included_rag_evaluation import (evaluate_images,
                                           extract_image_links,
                                           is_link_accessible,
                                           is_valid_azure_blob_link)


@pytest.mark.parametrize("markdown, expected_links", [
    ("![image1](http://example.com/image1.png) and ![image2](http://example.com/image2.png)", ["http://example.com/image1.png", "http://example.com/image2.png"]),
    ("No images here!", []),
    ("![image1](http://example.com/image1.png) with some text ![image2](http://example.com/image2.png)", ["http://example.com/image1.png", "http://example.com/image2.png"])
])
def test_extract_image_links(markdown: str, expected_links: List[str]):
    """
    Test the extract_image_links function with various markdown strings.
    """
    assert extract_image_links(markdown) == expected_links

@pytest.mark.parametrize("link, expected_result", [
    ("https://example.blob.core.windows.net/container/blob", True),
    ("https://example.com/blob", False),
    ("https://example.blob.core.windows.net/", False)
])
def test_is_valid_azure_blob_link(link: str, expected_result: bool):
    """
    Test the is_valid_azure_blob_link function with valid and invalid links.
    """
    assert is_valid_azure_blob_link(link) == expected_result

@pytest.mark.parametrize("link, status_code, expected", [
    ("https://accessible.blob.core.windows.net/container/blob", 200, True),
    ("https://unreachable.blob.core.windows.net/container/blob", 404, False)
])
def test_is_link_accessible(link: str, status_code: int, expected: bool):
    """
    Test the is_link_accessible function with various scenarios.
    """
    with patch('requests.head') as mocked_head:
        mocked_head.return_value.status_code = status_code

        assert is_link_accessible(link) == expected

@pytest.mark.parametrize("input_answer, ground_truth, documents, link_accessible, expected_output", [
    (
        "![image1](https://example.blob.core.windows.net/container/image1.png) ![image2](https://example.blob.core.windows.net/container/image2.png)",
        "![image1](https://example.blob.core.windows.net/container/image1.png) ![image2](https://example.blob.core.windows.net/container/image2.png)",
        ["https://example.blob.core.windows.net/container/image1.png", "https://example.blob.core.windows.net/container/image2.png"],
        True,
        {
            "retrieval_score": 1,
            "precision": 1.0,
            "recall": 1.0,
            "hallucination_count": 0,
            "hallucination_ratio": 0.0,
            "hallucination_broken_link": 0,
            "hallucination_resource_not_existing": 0,
            "hallucination_others": 0
        }
    ),
    (
        "![image1](https://example.blob.core.windows.net/container/image1.png)",
        "![image1](https://example.blob.core.windows.net/container/image1.png) ![image2](https://example.blob.core.windows.net/container/image2.png)",
        ["https://example.blob.core.windows.net/container/image1.png", "https://example.blob.core.windows.net/container/image2.png"],
        True,
        {
            "retrieval_score": 0,
            "precision": 1.0,
            "recall": 0.5,
            "hallucination_count": 0,
            "hallucination_ratio": 0.0,
            "hallucination_broken_link": 0,
            "hallucination_resource_not_existing": 0,
            "hallucination_others": 0
        }
    ),
    (
        "![image1](https://example.blob.core.windows.net/container/image1.png) ![invalid](invalid_link)",
        "![image1](https://example.blob.core.windows.net/container/image1.png)",
        ["https://example.blob.core.windows.net/container/image1.png"],
        True,
        {
            "retrieval_score": 0,
            "precision": 0.5,
            "recall": 1.0,
            "hallucination_count": 1,
            "hallucination_ratio": 0.5,
            "hallucination_broken_link": 1,
            "hallucination_resource_not_existing": 0,
            "hallucination_others": 0
        }
    ),
    (
        "![image1](https://example.blob.core.windows.net/container/nonexistent.png)",
        "![image1](https://example.blob.core.windows.net/container/image1.png)",
        ["https://example.blob.core.windows.net/container/image1.png"],
        False,
        {
            "retrieval_score": 0,
            "precision": 0,
            "recall": 0,
            "hallucination_count": 1,
            "hallucination_ratio": 1.0,
            "hallucination_broken_link": 0,
            "hallucination_resource_not_existing": 1,
            "hallucination_others": 0
        }
    ),
    (
        "![image1](https://example.blob.core.windows.net/container/image1.png) ![extra](https://example.blob.core.windows.net/container/extra.png)",
        "![image1](https://example.blob.core.windows.net/container/image1.png)",
        ["https://example.blob.core.windows.net/container/image1.png"],
        True,
        {
            "retrieval_score": 0,
            "precision": 0.5,
            "recall": 1.0,
            "hallucination_count": 1,
            "hallucination_ratio": 0.5,
            "hallucination_broken_link": 0,
            "hallucination_resource_not_existing": 0,
            "hallucination_others": 1
        }
    )
])
def test_evaluate_images(input_answer: str, ground_truth: str, documents: List[str], link_accessible: bool, expected_output: Dict[str, float]):
    """
    Test the evaluate_images function with various input answers, ground truths, and link accessibility.
    """
    with patch('image_included_rag_evaluation.is_link_accessible', return_value=link_accessible):
        assert evaluate_images(input_answer, ground_truth, documents) == expected_output