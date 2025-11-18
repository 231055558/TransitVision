import numpy as np

def select_frames(total_frames, n=7):
    if total_frames <= n:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, n, dtype=int).tolist()
    
    if len(indices) > 2:
        return indices[1:-1]
    return indices

def select_frame_indices(frame_list, n=7):
    total = len(frame_list)
    selected_idx = select_frames(total, n)
    return [frame_list[i] for i in selected_idx]

