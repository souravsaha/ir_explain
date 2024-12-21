from explainers import LirmePointwiseExplainer
# from explainers import EXSPointwiseExplainer
from visualize import TermVisualization
from eval import PointWiseCorrectness
# from metrics import PointWiseConsistency, PointWiseCorrectness
# from dataloaders import FetchDocuments
# from utils.perturbations import MaskingPerturbations
from sentence_transformers import CrossEncoder      
import torch

# Load your model from huggingface
model_name = "cross-encoder/ms-marco-electra-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = load_your_model()
model = CrossEncoder(model_name, max_length = 512, device = device)

# create an explainer 
explainer = LirmePointwiseExplainer(model, corpus_path = "/disk_a/junk/data/small-collection.tsv", indexer_type= "no-index")

# explainer = EXSPointwiseExplainer(model, corpus_path = "/disk_a/junk/data/small-collection.tsv", indexer_type= "no-index", exs_model = 'svm', num_samples = 100)

# Explain a prediction 
input_q = "what is the daily life of thai people"
input_d = "The following concepts are part of Thai everyday life: à¹ƒà¸ˆà¹€à¸¢à¹‡à¸™ or JAI YEN is more a way of life, it is to keep your temper whatever the situation. Thai people are educated in the family and in school to keep frustration inside. Instead of showing their anger toward a problem or situation, Thai people show JAI YEN, i.e. calm or patience. "

params = {
    "sampling_method" : "masking",
    "top_terms" : 20,
    "kernel_range" : [5,10]
}

explanation_vectors, ranked_lists = explainer.explain(input_q, input_d, params)

# Assuming you have information about top k retrieved list
# params = {
#     "doc_ids" : retrieved_dict["doc_ids"]
#     "rerank_scores" : retrieved_dict["rerank_scores"],
#     "rank" : rank,

# }
# explainer.explain(input_q, input_d, params= params, Method='topk-bin')


print(f"explanation vector: {explanation_vectors}")
print(f"ranked_lists : {ranked_lists}")

# visualize the explanation
termVisualization = TermVisualization()
termVisualization.visualize(explanation_vectors[0]["term_vector"], show_top=5)

pointWiseConsistency = PointWiseCorrectness(explainer)
pointWiseConsistency.evaluate(query_id = '1112341', doc_id = '8139258', explanation_vector = explanation_vectors[0]["term_vector"])
