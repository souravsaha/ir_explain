import math
def union(list1, list2):
    final_list = list(set(list1) | set(list2))
    return final_list

def intersection(list1, list2):
    return list(set(list1) & set(list2))

def compute_jaccard(list1, list2):
    return float(len(intersection(list1, list2))) / len(union(list1, list2))
