# Data preparation

Raw PCLMMPLUS videos and comments are not included. They may contain platform
identifiers or personal information. Public releases should prioritize
de-identified text, labels, split files and pre-extracted features.

Create `data/index.csv` with one row per video:

```csv
video_id,label,split,group,asr_text_path,comments_path,text_feature_path,video_feature_path,audio_feature_path,face_feature_path,skg_cache_path
```

Feature files are pickled NumPy arrays with shape `[T, D]`. Missing modalities
may use an empty path; the loader supplies a zero tensor and a false mask. The
published split contains 600 train, 66 development and 165 test videos. Never
split comments or feature streams independently of their `video_id`.

```text
data/
  index.csv
  TXT/<group>/<video_id>.txt
  comments/<group>/<video_id>.csv
  skg/mined_skg_chinese_with_label.json
features/
  TEXT_features/
  VIT_features/
  AUDIO_features/
  FACE_features/
  QWEN_SKG/
```
