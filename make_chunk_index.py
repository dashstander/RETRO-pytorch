
from datasets import load_dataset
from retro_pytorch.retrieval import text_dataset_to_chunks_, chunks_to_precalculated_knn_
import torch.nn.functional as F


dataset = load_dataset("wikipedia", "20200501.en")['train']


chunk_size = 64
knn = 2
seq_len = 2048
chunks_memmap_path = './train.chunks.dat'
seqs_memmap_path = './train.seq.dat'
doc_ids_memmap_path = './train.doc_ids.dat'
max_seqs = 6_078_422 * 5    # number of wikipedia articles in train, say it's an average of 5 sequences of 2048 tokens (overkill)
max_chunks = max_seqs * 32  # actual maximum number of sequences
knn_extra_neighbors = 100
processed_stats_json_path = './processed-stats.json'
faiss_index_filename = 'knn.index'


stats = text_dataset_to_chunks_(
    dataset = dataset, 
    chunks_memmap_path = chunks_memmap_path,
    seqs_memmap_path = seqs_memmap_path,
    doc_ids_memmap_path = doc_ids_memmap_path,
    chunk_size = chunk_size,
    seq_len = seq_len,
    max_chunks = max_chunks,
    max_seqs = max_seqs
)

print('###########################################')
print(f'Chunked all of the documents in wikipedia!')
print('###########################################')

num_chunks = stats['chunks']
num_seqs = stats['seqs']

# calculate knn memmap path and get the faiss index
# todo - make sure if faiss_index_filename is found, do not reprocess unless flag is given

knn_memmap_path, faiss_index = chunks_to_precalculated_knn_(
    num_chunks = num_chunks,
    chunk_size = chunk_size,
    chunk_memmap_path = chunks_memmap_path,
    doc_ids_memmap_path = doc_ids_memmap_path,
    num_nearest_neighbors = knn,
    num_extra_neighbors = knn_extra_neighbors,
    index_file = faiss_index_filename,
    force_reprocess = True,
)
