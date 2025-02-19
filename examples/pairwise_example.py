from explainers import PairwiseAxiomaticExplainer
index_path = ""

query_text = "Exons Definition Biology"
doc1 = "Herbivory: Definition & Examples 4:44 Next Lesson. Allometric Growth: Definition & Example; Isometric Growth: Definition & Example; Go to AP Biology - Animal Behavior: Help and Review"
doc2 = "Chitin (biology) Definition, meaning online encyclopedia mimi. (2) a polysaccharide which makes up the cell walls of fungi and exoskeleton of insects a tough, semitransparent substance that is the main component of the exoskeletons of art"

# Create an instance of the pairwise class
pairwise_instance = PairwiseAxiomaticExplainer(query_text, doc1, doc2, index_path)
axiom_classes = [pairwise_instance.TFC1(), pairwise_instance.LB1(), pairwise_instance.AND(), pairwise_instance.DIV(), pairwise_instance.PROX1()]

explanation = pairwise_instance.explain(query_text, doc1, doc2, axiom_classes)
print(explanation)

# Different arithmatic operations of the selected Axioms
axiom_classes = [4*pairwise_instance.LNC1() + 3*pairwise_instance.TFC1(), (-1)*pairwise_instance.DIV(), pairwise_instance.PROX1() + pairwise_instance.PROX2() + pairwise_instance.PROX3() ]
explanation = pairwise_instance.explain(query_text, doc1, doc2, axiom_classes)
print(explanation)

# Provide more explanation with the selected Axioms
axiom_name = "PROX1"
explanation = pairwise_instance.explain_details(query_text, doc1, doc2, axiom_name)
print(explanation)
