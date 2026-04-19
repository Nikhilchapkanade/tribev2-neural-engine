"""
mock_inference.py — Neuromarketing Ad-Testing Neural Simulator
Generates brain_voxel_data.json with Yeo 7 network activations
focused on Hook/Surprise (Ventral Attention) & Emotional Resonance (Limbic).

Usage: python mock_inference.py [video_path]
If video_path provided, reads its duration. Otherwise uses 30s default.
"""
import json, os, sys, subprocess
import numpy as np

OUTPUT_PATH = os.path.join("dashboard", "public", "brain_voxel_data.json")

NETWORKS = {
    "visual":            {"label": "Visual Cortex",        "marketing": "Visual Salience"},
    "somatomotor":       {"label": "Somatomotor",          "marketing": "Physical Response"},
    "dorsal_attention":  {"label": "Dorsal Attention",     "marketing": "Sustained Attention"},
    "ventral_attention": {"label": "Ventral Attention",    "marketing": "Hook / Surprise"},
    "limbic":            {"label": "Limbic",               "marketing": "Emotional Resonance"},
    "frontoparietal":    {"label": "Frontoparietal",       "marketing": "Brand Recall"},
    "default_mode":      {"label": "Default Mode",         "marketing": "Mind Wandering"},
}


def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, shell=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 30.0  # Default 30s


def generate_ad_profile(n_frames):
    """
    Simulates brain activity during a commercial ad with realistic engagement:
    - 0-3s:  Hook/opening — high Ventral Attention spike (surprise grab)
    - 3-8s:  Brand intro — moderate Visual + Dorsal Attention
    - 8-15s: Emotional storytelling — Limbic peak + sustained attention
    - 15-20s: Product reveal — second Ventral Attention spike
    - 20-25s: Call to action — Frontoparietal (brand recall) peaks
    - 25-30s: Closing — attention decay, DMN creeps in
    """
    t = np.linspace(0, 2 * np.pi, n_frames)
    np.random.seed(42)
    noise = lambda s=0.04: np.random.normal(0, s, n_frames)
    
    signals = {}

    # Ventral Attention ("Hook / Surprise") — sharp spikes at key moments
    hook = np.zeros(n_frames) + 0.15
    hook[1:4] += 0.85    # Opening hook (0-3s)
    hook[15:19] += 0.75  # Product reveal spike
    hook[8:10] += 0.4    # Story twist
    hook += noise(0.05)
    signals["ventral_attention"] = np.clip(hook, 0, 1.2)

    # Limbic ("Emotional Resonance") — builds during storytelling
    emotion = np.zeros(n_frames) + 0.2
    emotion += 0.7 * np.exp(-((np.linspace(0, n_frames, n_frames) - n_frames*0.4)**2) / (n_frames*3))
    emotion[8:16] += 0.5   # Emotional storytelling peak
    emotion[20:24] += 0.3  # Emotional callback in CTA
    emotion += noise(0.04)
    signals["limbic"] = np.clip(emotion, 0, 1.2)

    # Visual Salience — high during visual-heavy moments
    visual = 0.4 + 0.3 * np.sin(1.8 * t + 0.3) + noise(0.05)
    visual[0:5] += 0.3   # Opening visuals
    visual[15:20] += 0.25  # Product reveal visuals
    signals["visual"] = np.clip(visual, 0, 1.0)

    # Dorsal Attention ("Sustained Attention")
    sustained = 0.3 + 0.25 * np.sin(t + 1.0) + noise(0.04)
    sustained[3:15] += 0.3   # Mid-ad sustained focus
    sustained[25:] -= 0.15   # End decay
    signals["dorsal_attention"] = np.clip(sustained, 0, 1.0)

    # Somatomotor ("Physical Response") — low baseline, spikes on action
    somato = 0.1 + 0.1 * np.sin(0.8 * t) + noise(0.03)
    somato[2:5] += 0.25
    signals["somatomotor"] = np.clip(somato, 0, 0.7)

    # Frontoparietal ("Brand Recall") — highest during CTA
    brand = 0.25 + 0.2 * np.sin(1.2 * t + 2.0) + noise(0.04)
    brand[20:26] += 0.5  # CTA = peak brand encoding
    brand[15:20] += 0.3  # Product reveal
    signals["frontoparietal"] = np.clip(brand, 0, 1.1)

    # Default Mode ("Mind Wandering") — inverse of engagement
    dmn = 0.15 + noise(0.04)
    dmn[25:] += 0.4      # End-of-ad disengagement
    dmn[5:8] += 0.2      # Brief lull before story
    signals["default_mode"] = np.clip(dmn, 0, 0.9)

    return signals


def main():
    # Determine video duration
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if video_path and os.path.exists(video_path):
        duration = get_video_duration(video_path)
        print(f"Video: {video_path} ({duration:.1f}s)")
    else:
        duration = 30.0
        print(f"No video provided, using default {duration}s duration")

    n_frames = int(duration)
    signals = generate_ad_profile(n_frames)

    frames = []
    for i in range(n_frames):
        ts = round(float(i), 2)
        frame = {"timestamp_seconds": ts, "networks": {}}

        for net_name, net_info in NETWORKS.items():
            frame["networks"][net_name] = {
                "activation": round(float(signals[net_name][i]), 4),
                "marketing_label": net_info["marketing"],
            }

        all_acts = [signals[n][i] for n in NETWORKS]
        frame["global_mean"] = round(float(np.mean(all_acts)), 4)
        frame["peak_activation"] = round(float(np.max(all_acts)), 4)

        # Key marketing metrics
        hook = signals["ventral_attention"][i]
        emotion = signals["limbic"][i]
        frame["engagement_score"] = round(float((hook + emotion) / 2), 4)
        frame["hook_emotion_product"] = round(float(hook * emotion), 4)
        frame["hemisphere_asymmetry"] = round(float(
            signals["frontoparietal"][i] - signals["default_mode"][i]
        ), 4)

        frames.append(frame)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(frames, f, indent=2)

    print(f"Exported {len(frames)} frames to {OUTPUT_PATH}")

    # Key insights
    hook_arr = signals["ventral_attention"]
    emo_arr = signals["limbic"]
    combined = hook_arr * emo_arr
    
    peak_idx = np.argmax(combined)
    print(f"\nPeak Engagement: t={peak_idx}s (Hook={hook_arr[peak_idx]:.3f} × Emotion={emo_arr[peak_idx]:.3f})")
    
    threshold = 0.15
    hot_frames = np.where(combined > threshold)[0]
    if len(hot_frames) > 0:
        print(f"High-engagement segments: {hot_frames[0]}s – {hot_frames[-1]}s ({len(hot_frames)} frames)")


if __name__ == "__main__":
    main()
