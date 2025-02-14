import argparse
import datasets
import torch
import numpy as np
import hdbscan
import io
import base64
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, SiglipModel
from torchvision import transforms


# Import your model configuration, model, and weight-loading utility.
from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

# You already have a prefix for the question.
PREFIX = (
    "Analyze the chart carefully, consider both visual features and data values, "
    "and provide a precise answer without any additional explanation or formatting. "
)

# --- Relaxed Correctness (unchanged) ---
def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction == target

# --- Helper: Convert PIL image to base64 string ---
def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# --- Option B: (Optional) Use siglip from Hugging Face for image embeddings ---
# Uncomment and adjust the model name if you prefer using siglip.
#
# from transformers import AutoProcessor, AutoModel
# siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
# siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
siglip = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
# siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
siglip.to("cuda")

i_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])


def pil_to_compressed_base64(image, max_size=(200, 200), quality=10):
    """
    Resize and compress the image, then convert to a base64 string.
    
    Args:
      image (PIL.Image): The input image.
      max_size (tuple): Maximum width and height.
      quality (int): JPEG quality (1-95). Lower means more compression.
      
    Returns:
      str: Base64-encoded JPEG image.
    """
    # Create a copy so as not to modify the original image.
    img_copy = image.copy()
    img_copy.thumbnail(max_size)  # This maintains aspect ratio.
    
    buffer = io.BytesIO()
    # Save as JPEG with reduced quality.
    img_copy.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def get_image_embedding(image):
    with torch.no_grad():
        pixel_values = i_transform(image).unsqueeze(0).to("cuda")
        # Adjust pooling as necessary for your siglip model.
        embedding = siglip.get_image_features(pixel_values=pixel_values)
    return embedding
#

# Then, in the code below you can replace model.encode_image(image) with get_image_embedding(image).

# --- Evaluation Function ---
def eval_chartqa(model, debug=False, n_samples=-1):
    dataset = datasets.load_dataset("vikhyatk/chartqa", split="test")
    # If n_samples is positive, select only the first n_samples.
    if n_samples > 0:
        dataset = dataset.select(range(n_samples))
    
    correct = 0
    total = 0
    human_correct = 0
    human_total = 0
    # We'll store the per-image QA results in a list (so that each entry corresponds to one chart image)
    results = []

    for row in tqdm(dataset, disable=debug, desc="ChartQA"):
        image = row["image"]
        # Use your model’s image encoder (or replace with get_image_embedding(image) if using siglip)
        encoded_image = model.encode_image(image)

        row_result = []
        for qa in row["qa"]:
            question = PREFIX + qa["question"]
            answer = qa["answer"]
            model_answer = model.query(encoded_image, question)["answer"]

            # Try to parse answers as lists; fall back to string comparison if that fails.
            try:
                answer_list = json.loads(answer)
                model_answer_list = json.loads(model_answer)
                if not (isinstance(answer_list, list) and isinstance(model_answer_list, list) and len(answer_list) == len(model_answer_list)):
                    raise ValueError
            except Exception:
                answer_list = [answer]
                model_answer_list = [model_answer]

            total += 1
            if qa["source"] == "human":
                human_total += 1

            is_correct = False
            if all(
                relaxed_correctness(str(cur_answer).strip().lower(), str(cur_model_answer).strip().lower())
                for cur_answer, cur_model_answer in zip(answer_list, model_answer_list)
            ):
                correct += 1
                if qa["source"] == "human":
                    human_correct += 1
                is_correct = True

            row_result.append({
                "question": question,
                "ground_truth": answer_list,
                "model_answer": model_answer_list,
                "is_correct": is_correct,
                "source": qa["source"],
            })
        results.append(row_result)

    eval_stats = {
        "human_acc": human_correct * 100 / human_total if human_total > 0 else 0,
        "total_acc": correct * 100 / total if total > 0 else 0,
        "results": results,
        "dataset": dataset,  # We return the dataset so we can later match images with their results.
    }
    return eval_stats

# --- Clustering Function ---
def cluster_wrong_samples(model, dataset, results, min_cluster_size=3):
    """
    For each image in the dataset, if any of its QA pairs were answered incorrectly,
    obtain the image embedding (using siglip or model.encode_image) and store along with details.
    Then cluster these embeddings using HDBSCAN.
    """
    wrong_samples = []
    # Assumes that the order of `results` matches that of `dataset`
    for row, row_results in zip(dataset, results):
        # Gather only the QAs where the answer was wrong.
        wrong_qas = [qa for qa in row_results if not qa["is_correct"]]
        if wrong_qas:
            # Use model.encode_image or get_image_embedding(row["image"]) if using siglip.
            embedding = get_image_embedding(row["image"])#model.encode_image(row["image"])
            # Convert the embedding to a list (make sure to detach and move to CPU if needed)
            embedding_np = embedding.detach().cpu().numpy().tolist()
            image_b64 = pil_to_compressed_base64(row["image"], max_size=(400,400), quality=15)
            wrong_samples.append({
                "image_base64": image_b64,
                "embedding": embedding_np,
                "qas": wrong_qas,
            })

    if not wrong_samples:
        return {}

    # Stack embeddings into a NumPy array for clustering.
    embeddings = np.array([sample["embedding"] for sample in wrong_samples])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    embeddings = embeddings.squeeze(1)
    labels = clusterer.fit_predict(embeddings)

    # Add cluster labels to each sample.
    for sample, label in zip(wrong_samples, labels):
        sample["cluster"] = int(label)

    # Organize samples by cluster.
    clusters = {}
    for sample in wrong_samples:
        cluster_id = sample["cluster"]
        clusters.setdefault(cluster_id, []).append({
            "image_base64": sample["image_base64"],
            "qas": sample["qas"],
        })
    return clusters

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_json", type=str, default="moondream/cluster/clusters.json", help="Output JSON file for clusters")
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples to process (-1 for entire dataset)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
        device = "mps"

    config = MoondreamConfig()
    model = MoondreamModel(config).to(device)
    load_weights_into_model(args.model, model)
    # model.compile()

    # Evaluate ChartQA with a sample limit if specified.
    eval_stats = eval_chartqa(model, args.debug, n_samples=args.n_samples)
    print(f"Human Accuracy: {eval_stats['human_acc']:.2f}")
    print(f"Total Accuracy: {eval_stats['total_acc']:.2f}")

    # Cluster the wrong samples.
    dataset = eval_stats["dataset"]
    results = eval_stats["results"]
    clusters = cluster_wrong_samples(model, dataset, results, min_cluster_size=10)

    # Save the clusters as JSON.
    with open(args.output_json, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"Clusters saved to {args.output_json}")