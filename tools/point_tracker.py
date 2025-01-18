import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
import cv2

def preprocess_frames(frames):
    """Preprocess frames to model inputs.
    Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8
    Returns:
        frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames

def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)
    return points

def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles

def inference(frames, query_points, model):
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    outputs = model(frames, query_points)
    tracks, occlusions, expected_dist = (
        outputs['tracks'][0],
        outputs['occlusion'][0],
        outputs['expected_dist'][0],
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles

def convert_select_points_to_query_points(frame, points):
    """Convert select points to query points.
    Args:
        points: [num_points, 2], in [x, y]
    Returns:
        query_points: [num_points, 3], in [t, y, x]
    """
    points = np.stack(points)
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points

def main():
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load('tapnet/checkpoints/bootstapir_checkpoint_v2.pt'))
    model = model.to(device)
    model = model.eval()
    torch.set_grad_enabled(False)

    # Video loading and preprocessing
    cap = cv2.VideoCapture('tapnet/examplar_videos/horsejump-high.mp4')
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    video = np.array(frames)

    # Parameters
    resize_height = 256
    resize_width = 256
    num_points = 50  # Number of random points to track

    # Resize frames
    frames_resized = []
    for frame in video:
        frame_resized = cv2.resize(frame, (resize_width, resize_height))
        frames_resized.append(frame_resized)
    frames_resized = np.array(frames_resized)

    # Generate random points to track
    query_points = sample_random_points(0, frames_resized.shape[1], frames_resized.shape[2], num_points)
    
    # Convert to torch tensors
    frames_tensor = torch.tensor(frames_resized).to(device)
    query_points_tensor = torch.tensor(query_points).to(device)

    # Run inference
    tracks, visibles = inference(frames_tensor, query_points_tensor, model)

    # Convert back to numpy for visualization
    tracks = tracks.cpu().detach().numpy()
    visibles = visibles.cpu().detach().numpy()

    # Scale tracks back to original video size
    height, width = video.shape[1:3]
    tracks = transforms.convert_grid_coordinates(
        tracks, (resize_width, resize_height), (width, height)
    )

    # Visualize tracks
    video_viz = viz_utils.paint_point_track(video, tracks, visibles)
    
    # Save visualization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_tracks.mp4', fourcc, 10.0, (width, height))
    
    for frame in video_viz:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print("Tracking visualization saved as 'output_tracks.mp4'")

if __name__ == '__main__':
    main()