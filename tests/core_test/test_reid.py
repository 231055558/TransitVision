import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.core.reid import ReIDFeatureExtractor, apply_nfc, greedy_matching
from transit_vision.utils import collect_images_from_folder
import torch
import numpy as np

MODEL_PATH = "/mnt/mydisk/My_project/TransitVision/ckpt/transformer_20.pth"
CFG_PATH = "/mnt/mydisk/My_project/TransitVision/tests/pose2id_scheme/Pose2ID/IPG/cfg_transreid.pkl"
TEST_DIR = "/mnt/mydisk/My_project/TransitVision/tests/pose2id_scheme/bus_data/reid_test"

def test_feature_extraction():
    print("=== 测试特征提取 ===\n")
    
    extractor = ReIDFeatureExtractor(MODEL_PATH, CFG_PATH)
    
    pre_folder = Path(TEST_DIR) / "pre_feature_1"
    images = collect_images_from_folder(pre_folder)[:5]
    
    print(f"提取 {len(images)} 张图像特征...")
    features = extractor.extract_batch(images, batch_size=4)
    
    print(f"✓ 特征维度: {features.shape}")
    print(f"✓ 特征范数: {torch.norm(features, dim=1).mean():.4f}\n")


def test_nfc():
    print("=== 测试NFC ===\n")
    
    extractor = ReIDFeatureExtractor(MODEL_PATH, CFG_PATH)
    
    pre_folder = Path(TEST_DIR) / "pre_feature_1"
    images = collect_images_from_folder(pre_folder)[:10]
    
    print(f"提取特征...")
    features = extractor.extract_batch(images)
    
    print(f"应用NFC...")
    features_nfc = apply_nfc(features, k1=2, k2=2)
    
    print(f"✓ 原始特征范数: {torch.norm(features, dim=1).mean():.4f}")
    print(f"✓ NFC后范数: {torch.norm(features_nfc, dim=1).mean():.4f}\n")


def test_matching():
    print("=== 测试匹配 ===\n")
    
    extractor = ReIDFeatureExtractor(MODEL_PATH, CFG_PATH)
    
    pre_folders = [f"pre_feature_{i}" for i in range(1, 4)]
    cap_folders = [f"cap_feature_{i}" for i in range(1, 4)]
    
    pre_feats = []
    cap_feats = []
    
    for folder in pre_folders:
        images = collect_images_from_folder(Path(TEST_DIR) / folder)[:5]
        if images:
            feat = extractor.extract_batch(images)
            pre_feats.append(feat.mean(dim=0))
    
    for folder in cap_folders:
        images = collect_images_from_folder(Path(TEST_DIR) / folder)[:5]
        if images:
            feat = extractor.extract_batch(images)
            cap_feats.append(feat.mean(dim=0))
    
    pre_feats = torch.stack(pre_feats)
    cap_feats = torch.stack(cap_feats)
    
    similarity = torch.matmul(pre_feats, cap_feats.t())
    
    print("相似度矩阵:")
    print(similarity.numpy())
    
    matches, unmatched_pre, unmatched_cap = greedy_matching(similarity.numpy(), threshold=0.4)
    
    print(f"\n匹配结果: {len(matches)} 对")
    for pre_idx, cap_idx, sim in matches:
        print(f"  pre_{pre_idx+1} <-> cap_{cap_idx+1}: {sim:.4f}")
    
    print(f"\n未匹配: pre={unmatched_pre}, cap={unmatched_cap}\n")


if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        print(f"模型文件不存在: {MODEL_PATH}")
        print("请下载模型到ckpt目录")
        exit(1)
    
    test_feature_extraction()
    test_nfc()
    test_matching()
    
    print("✓ 所有测试完成")

