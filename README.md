# 📚 pose-format Tutorials

Practical, end-to-end notebooks that teach the `pose-format` library from basics to advanced workflows (conversion, visualization, augmentation, and web usage). Written for researchers and developers working with sign language and human pose data.

## Prerequisites
- Python 3.9+
- `ffmpeg` available on PATH for video handling (recommended)
- GPU optional for deep learning examples

## Installation
```bash
pip install -r requirements.txt

# or minimal core
pip install pose-format
```

## Tutorial Notebooks
| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [01_extract_landmarks_from_video.ipynb](01_extract_landmarks_from_video.ipynb) | Extract landmarks from video with MediaPipe Holistic and save `.pose` files |
| 02 | [02_convert_pose_formats.ipynb](02_convert_pose_formats.ipynb) | Convert `.pose` to JSON/NPZ/Parquet/CSV and back |
| 03 | [03_read_pose_files.ipynb](03_read_pose_files.ipynb) | Load pose data with NumPy, PyTorch, TensorFlow backends |
| 04 | [04_visualize_pose.ipynb](04_visualize_pose.ipynb) | Render videos, GIFs, and images from pose data |
| 05 | [05_advanced_features.ipynb](05_advanced_features.ipynb) | Normalize, augment, interpolate, and slice components |
| 06 | [06_web_javascript_usage.ipynb](06_web_javascript_usage.ipynb) | Use `pose-format` in JavaScript/TypeScript for web apps |

## Quick Start (Python)
```python
from pose_format import Pose

with open("sign.pose", "rb") as f:
    pose = Pose.read(f.read())

print("Frames", len(pose.body.data))
print("FPS", pose.body.fps)
print("Components", [c.name for c in pose.header.components])
```

## CLI Cheatsheet
```bash
# Extract from video
video_to_pose --format mediapipe -i video.mp4 -o output.pose

# Batch extract
videos_to_poses --format mediapipe -i ./original_videos -o ./pose_files

# Visualize
visualize_pose -i pose_files/sample.pose -o visualized_video/sample.mp4
```

## Repository Layout
- Notebooks: 01–06 (see table above)
- [original_videos/](original_videos/) — raw input videos (if available)
- [pose_files/](pose_files/) — generated `.pose` files (examples provided)
- [visualized_video/](visualized_video/) — rendered previews
- [requirements.txt](requirements.txt) — pinned dependencies for notebooks
- [LICENSE](LICENSE) — MIT license

## Contributing
Issues and pull requests are welcome to expand coverage, fix bugs, or improve examples.