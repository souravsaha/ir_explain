from explainers import LirmePointwiseExplainer
# from visualize import TermWeightVisualizer
# from metrics import PointWiseConsistency, PointWiseCorrectness
# from dataloaders import FetchDocuments
# from utils.perturbations import MaskingPerturbations


# Load your model from huggingface
# model = load_your_model()

model = "test"

# create an explainer 
explainer = LirmePointwiseExplainer(model, corpus_path = "/disk_a/junk/data/collection.tsv", indexer_type= "no-index")

# Explain a prediction 
input_q = "what is the daily life of thai people"
input_d = "An important thing in everyday life is SANUK . Thai people love to have fun together . SANUK can represent many things : eat together , to be with friends and chat , to go out with friends . For Thai people SANUK happens with several persons . "

params = {
    "sampling_method" : "masking",
    "top_terms" : 20,
    "kernel_range" : [5,10]
}

explanation = explainer.explain(input_q, input_d, params)

# visualize the explanation
# visualize(explanation)
