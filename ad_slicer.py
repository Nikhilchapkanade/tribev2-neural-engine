"""
ad_slicer.py — Deterministic Peak Engagement B-Roll Extractor
Reads brain_voxel_data.json, identifies timestamps where BOTH Ventral Attention
(Hook/Surprise) AND Limbic (Emotional Resonance) breach threshold simultaneously,
then uses FFmpeg to extract those segments as B-roll.
"""
import json, os, subprocess, sys

DATA_PATH = os.path.join("dashboard", "public", "brain_voxel_data.json")
OUTPUT_DIR = "b_roll_output"
HOOK_THRESHOLD = 0.5
EMOTION_THRESHOLD = 0.4


def find_peak_segments(data):
    """Find contiguous segments where both Hook AND Emotion exceed thresholds."""
    segments = []
    in_peak = False
    start = None

    for frame in data:
        ts = frame["timestamp_seconds"]
        hook = frame["networks"]["ventral_attention"]["activation"]
        emotion = frame["networks"]["limbic"]["activation"]

        if hook > HOOK_THRESHOLD and emotion > EMOTION_THRESHOLD:
            if not in_peak:
                start = ts
                in_peak = True
        else:
            if in_peak:
                segments.append({"start": start, "end": ts})
                in_peak = False

    if in_peak:
        segments.append({"start": start, "end": data[-1]["timestamp_seconds"]})

    return segments


def extract_broll(segments, video_path, output_dir):
    """Use FFmpeg to extract peak engagement segments."""
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found. Generating report only.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        out = os.path.join(output_dir, f"peak_engagement_{i+1}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            "-c", "copy", out
        ]
        print(f"  Extracting: {seg['start']:.1f}s – {seg['end']:.1f}s → {out}")
        try:
            subprocess.run(cmd, capture_output=True, shell=True, timeout=30)
        except Exception as e:
            print(f"  FFmpeg error: {e}")

    print(f"\nExtracted {len(segments)} B-roll clips to {output_dir}/")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("dashboard", "public", "input_video.mp4")

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run mock_inference.py first.")
        return

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} frames | Hook threshold: {HOOK_THRESHOLD} | Emotion threshold: {EMOTION_THRESHOLD}")

    segments = find_peak_segments(data)
    print(f"\nFound {len(segments)} Peak Engagement segments:")
    total_duration = 0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        total_duration += dur
        print(f"  {seg['start']:.1f}s – {seg['end']:.1f}s ({dur:.1f}s)")

    print(f"\nTotal B-roll duration: {total_duration:.1f}s")

    # Export report
    report = {
        "peak_segments": segments,
        "total_broll_seconds": total_duration,
        "thresholds": {"hook": HOOK_THRESHOLD, "emotion": EMOTION_THRESHOLD},
    }
    report_path = os.path.join("dashboard", "public", "broll_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    extract_broll(segments, video_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()
