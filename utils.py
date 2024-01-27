import re
# from nltk import ngrams
from datasketch import MinHash, MinHashLSH
from collections import defaultdict

# Retain English, and underscores, excluding punctuation marks
NON_CHAR = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")

def _get_doc_mini_hash(doc: list[str] | str, num_perm: int) -> MinHash:
    '''
    Get the mini hash of a text segment
    '''
    mini_hash = MinHash(num_perm=num_perm)
    for s in doc:
        mini_hash.update(s.encode('utf-8'))
    return mini_hash

class DropDatasetDuplicate:

    def __init__(self,  threshold: float=0.85, num_perm: int=256) -> None:
        '''
        Get all duplicate (similar over threshold) indexes in a dataset, input as: list[str], where a str element is a text segment (doc)
        For example, input: [a, b, c, d, c, d, e] returns: {4, 5} (indexes of the latter two c, d)
        '''
        self.similar_index_cluster = defaultdict(set)
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) 
        self.num_perm = num_perm

    def add_doc(self, index: object, doc: str,) -> set[int]:
        '''
        Add a document,
        index: Index of the document
        doc: The document itself
        '''

        # Retain only English, and underscores, excluding punctuation marks
        doc = ''.join(NON_CHAR.split(doc))
        # doc = [''.join(t) for t in list(ngrams(doc, 3))]

        doc_hash = _get_doc_mini_hash(doc, self.num_perm)
        close duplicates = self.data_lsh.query(doc_hash)

        self.data_lsh.insert(index, doc_hash)

        # All similar docs in similar_index_cluster have the key as the earliest appearing index
        # For example: if indexes 2, 7, 8, 9, 10, 12 in data are similar, then it is represented in similar_index_cluster as {2: {8, 9, 10, 12}}
        if len(close_duplicates) > 0:
            min_idx= min(close_duplicates)
            self.similar_index_cluster[min_idx].add(index)
    
    def get_duplicate_indexs(self):
        '''
        Return all duplicate document indexes
        '''
        similar_index_cluster = self.similar_index_cluster
        need_to_remove_idx = set()
        
        for key_idx in similar_index_cluster.keys():
            need_to_remove_idx |= similar_index_cluster[key_idx]

        return need_to_remove_idx

def get_dataset_duplicate_index(data: list[str], threshold: float=0.85, num_perm: int=256) -> set[int]:
    '''
    Get all duplicate (similar over threshold) indexes in a dataset, input as: list[str], where a str element is a text segment (doc)
    For example, input: [a, b, c, d, c, d, e] returns: {4, 5} (indexes of the latter two c, d)
    '''
    similar_index_cluster = defaultdict(set)
    data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for i, doc in enumerate(data):

        # Retain only English, and underscores, excluding punctuation marks
        doc = ''.join(NON_CHAR.split(doc))
        # doc = [''.join(t) for t in list(ngrams(doc, 3))]

        doc_hash = _get_doc_mini_hash(doc, num_perm)
        close_duplicates = data_lsh.query(doc_hash)

        data_lsh.insert(i, doc_hash)

        # All similar docs in similar_index_cluster have the key as the earliest appearing index
        # For example: if indexes 2, 7, 8, 9, 10, 12 in data are similar, then it is represented in similar_index_cluster as {2: {8, 9, 10, 12}}
        if len(close_duplicates) > 0:
            min_idx= min(close_duplicates)
            similar_index_cluster[min_idx].add(i)
    
    need_to_remove_idx = set()
    for key_idx in similar_index_cluster.keys():
        need_to_remove_idx |= similar_index_cluster[key_idx]

    return need_to_remove_idx
