from sentence_transformers import SentenceTransformer, util
import numpy as np
from static_data import *
from memory_profiler import profile
from tools import result_report, load_data, get_date_time
from transformers import BertModel, BertTokenizer

# @profile
def ZSL(test_data_path, result_dir, save_name):
    model = SentenceTransformer("all-MiniLM-L12-v2")
    
    # Create mapping between labels and their descriptions
    label_mapping = {
        "1 1": "all must be, all of the",
        "1 0": "100% of ,later 100 ,out of 100 ,longer than 100 ,be avilable 100 ,exceed 100 ,more than 100 ,support 100 ,up to 100 ,at least 100 ,minimum of 100 ,support a maximum of 100 ,after 100 ,be able to handle 100 ,be capable of handling 100 ,achieve 100",
        "1 -1": "a period of 100 ,on 100 ,be 100 ,approximately 100 ,reamain 100 ,to 100 ,every 100 ,contain the 100",
        "0 -1": "within 100 ,decreased by 100 ,in under 100 ,less than 100 ,maximum of 100 ,at most 100",
        "-1 -1": "in an acceptable time ,respond fast to"
    }
    sentences, real_labels = load_data(test_data_path)
    sentence_embeddings = model.encode(sentences)

    predicted_labels = []
    i = 0
    for st_emb in sentence_embeddings:
        i += 1
        similarities = []
        label_keys = []
        for label in label_mapping:
            label_keys.append(label)
            description = label_mapping[label]  # Extract label description
            lb_emb = model.encode(description)
            cos_sim = util.cos_sim(st_emb, lb_emb)  # Calculate cosine similarity between sentence and label embeddings
            similarities.append(cos_sim.item())  # Store the similarity value (convert tensor to float)
            
        # Sort labels by similarity in descending order
        sorted_similarities = sorted(zip(label_keys, similarities), key=lambda x: x[1], reverse=True) 
        predicted_labels.append(sorted_similarities[0][0])  # Save the most similar label

    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    result_report("ZSL", save_path, predicted_labels, real_labels)