import torch
import numpy as np

def compute_similarity(feats1, feats2):
    """计算两组特征的相似度矩阵"""
    similarity_matrix = torch.matmul(feats1, feats2.t())
    return similarity_matrix


def compute_avg_similarity(feats1, feats2):
    """计算两组特征的平均相似度"""
    similarity_matrix = compute_similarity(feats1, feats2)
    return similarity_matrix.mean().item()


def find_best_match(query_feat, gallery_feats, threshold=0.4):
    """为单个query找到最佳匹配"""
    similarities = torch.matmul(query_feat, gallery_feats.t()).squeeze()
    
    if similarities.dim() == 0:
        similarities = similarities.unsqueeze(0)
    
    max_sim, max_idx = similarities.max(dim=0)
    
    if max_sim.item() >= threshold:
        return max_idx.item(), max_sim.item()
    
    return None, max_sim.item()


def greedy_matching(similarity_matrix, threshold=0.4):
    """贪心匹配算法
    
    Args:
        similarity_matrix: [N_query, N_gallery]
        threshold: 最低相似度阈值
    
    Returns:
        matches: [(query_idx, gallery_idx, similarity), ...]
        unmatched_query: 未匹配的query索引
        unmatched_gallery: 未匹配的gallery索引
    """
    n_query, n_gallery = similarity_matrix.shape
    
    flat_similarities = []
    for i in range(n_query):
        for j in range(n_gallery):
            flat_similarities.append((similarity_matrix[i, j], i, j))
    
    flat_similarities.sort(reverse=True, key=lambda x: x[0])
    
    matched_query = set()
    matched_gallery = set()
    matches = []
    
    for similarity, q_idx, g_idx in flat_similarities:
        if q_idx not in matched_query and g_idx not in matched_gallery:
            if similarity >= threshold:
                matches.append((q_idx, g_idx, similarity))
                matched_query.add(q_idx)
                matched_gallery.add(g_idx)
    
    unmatched_query = [i for i in range(n_query) if i not in matched_query]
    unmatched_gallery = [j for j in range(n_gallery) if j not in matched_gallery]
    
    return matches, unmatched_query, unmatched_gallery

