from explainers import LirmePointwiseExplainer
from visualize import TermVisualization
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

# Explain a prediction 
input_q = "what is the daily life of thai people"
input_d = "An important thing in everyday life is SANUK . Thai people love to have fun together . SANUK can represent many things : eat together , to be with friends and chat , to go out with friends . For Thai people SANUK happens with several persons . "

params = {
    "sampling_method" : "masking",
    "top_terms" : 20,
    "kernel_range" : [5,10]
}

explanation_vectors, ranked_lists = explainer.explain(input_q, input_d, params)

print(f"explanation vector: {explanation_vectors}")
print(f"ranked_lists : {ranked_lists}")

termVisualization = TermVisualization()
termVisualization.visualize(explanation_vectors[0]["term_vector"], show_top=5)
# visualize the explanation
# visualize(explanation)
