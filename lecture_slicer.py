"""
lecture_slicer.py — Deterministic Lecture Editor
Reads brain_voxel_data.json, identifies timestamps where Default Mode Network
(Cognitive Drift) exceeds threshold, and uses FFmpeg to cut out low-focus segments.
"""
import json
import os
import subprocess

DATA_PATH = os.path.join("dashboard", "public", "brain_voxel_data.json")
INPUT_VIDEO = os.path.join("dashboard", "public", "input_video.mp4")
OUTPUT_DIR = "sliced_lectures"
DRIFT_THRESHOLD = 0.55  # DMN activation above this = mind-wandering


def identify_drift_segments(data, threshold=DRIFT_THRESHOLD):
    """Find contiguous time ranges where Default Mode > threshold."""
    segments = []
    in_drift = False
    start = None
    
    for frame in data:
        ts = frame["timestamp_seconds"]
        dmn = frame["networks"]["default_mode"]["activation"]
        
        if dmn > threshold and not in_drift:
            start = ts
            in_drift = True
        elif dmn <= threshold and in_drift:
            segments.append({"start": start, "end": ts, "type": "drift"})
            in_drift = False
    
    # Close any open segment
    if in_drift:
        segments.append({"start": start, "end": data[-1]["timestamp_seconds"], "type": "drift"})
    
    return segments


def identify_engagement_peaks(data, threshold=0.5):
    """Find timestamps where Dorsal Attention is highest (best learning moments)."""
    peaks = []
    for frame in data:
        ts = frame["timestamp_seconds"]
        focus = frame["networks"]["dorsal_attention"]["activation"]
        if focus > threshold:
            peaks.append({"timestamp": ts, "focus_level": round(focus, 3)})
    return peaks


def slice_video(drift_segments, video_path, output_dir):
    """Use FFmpeg to extract only the high-focus segments (removing drift periods)."""
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found. Skipping video slicing.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build FFmpeg filter to select only non-drift segments
    # Strategy: cut the video at drift boundaries, keeping engaged segments
    print(f"\nSlicing video to remove {len(drift_segments)} low-focus segments...")
    
    for i, seg in enumerate(drift_segments):
        print(f"  Drift segment {i+1}: {seg['start']:.1f}s – {seg['end']:.1f}s (removing)")
    
    # For each drift segment, extract the video BEFORE it as a "good" clip
    if not drift_segments:
        print("No drift segments found — entire lecture maintains focus!")
        return
    
    # Build the segments to KEEP (inverse of drift)
    keep_segments = []
    prev_end = 0
    for seg in drift_segments:
        if seg["start"] > prev_end:
            keep_segments.append({"start": prev_end, "end": seg["start"]})
        prev_end = seg["end"]
    # Add final segment after last drift
    keep_segments.append({"start": prev_end, "end": 9999})
    
    # Use FFmpeg select filter
    for i, seg in enumerate(keep_segments):
        output_path = os.path.join(output_dir, f"engaged_segment_{i+1}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(seg["start"]),
            "-to", str(min(seg["end"], 300)),  # Cap at 5 min
            "-c", "copy",
            output_path
        ]
        print(f"  Extracting engaged segment {i+1}: {seg['start']:.1f}s – {seg['end']:.1f}s")
        try:
            subprocess.run(cmd, capture_output=True, shell=True, timeout=30)
        except Exception as e:
            print(f"  FFmpeg error: {e}")
    
    print(f"\nExtracted {len(keep_segments)} high-focus segments to {output_dir}/")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run mock_inference.py first.")
        return
    
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} frames of neural data")
    print(f"Drift threshold: {DRIFT_THRESHOLD}")
    
    # Identify cognitive drift segments
    drift_segments = identify_drift_segments(data)
    print(f"\nFound {len(drift_segments)} cognitive drift segments:")
    for seg in drift_segments:
        duration = seg["end"] - seg["start"]
        print(f"  {seg['start']:.1f}s – {seg['end']:.1f}s ({duration:.1f}s of mind-wandering)")
    
    # Identify engagement peaks
    peaks = identify_engagement_peaks(data)
    print(f"\nFound {len(peaks)} high-focus timestamps (Dorsal Attention > 0.5)")
    if peaks:
        best = max(peaks, key=lambda p: p["focus_level"])
        print(f"  Best moment: t={best['timestamp']}s (focus={best['focus_level']})")
    
    # Export analysis report
    report = {
        "drift_segments": drift_segments,
        "engagement_peaks": peaks[:10],  # Top 10
        "total_drift_time": sum(s["end"] - s["start"] for s in drift_segments),
        "total_engaged_time": len(data) - sum(s["end"] - s["start"] for s in drift_segments),
        "drift_threshold": DRIFT_THRESHOLD,
    }
    
    report_path = os.path.join("dashboard", "public", "lecture_analysis.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nExported analysis report to {report_path}")
    
    # Slice video
    slice_video(drift_segments, INPUT_VIDEO, OUTPUT_DIR)


if __name__ == "__main__":
    main()
