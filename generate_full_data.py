"""
generate_full_data.py — Generates ALL data needed for the complete Rewire dashboard.
Exports a single comprehensive JSON with:
  - Per-frame Yeo 7 network activations
  - Demographic breakdowns (Adults, Gen Z, Kids, Older)
  - Peak scene detection
  - ROI rankings
  - Inter-subject variance matrix
  - Neural narrative transcript
"""
import json, os
import numpy as np

OUTPUT = os.path.join("dashboard", "public", "brain_data_full.json")
N_FRAMES = 33  # 0-32 seconds (Old Spice ad length)
np.random.seed(42)

DEMOGRAPHICS = ["Gen Z", "Adults", "Kids", "Older"]
COGNITIVE_DIMS = ["Visual", "Auditory", "Reward", "Memory", "Attention", "Narrative", "Personal", "Action"]

ROIS = [
    {"id": "V4", "name": "Visual Area 4", "desc": "Processes color contrast and object boundaries. High activation indicates the visual content has rich chromatic information demanding perceptual parsing."},
    {"id": "MT+", "name": "Medial Superior Temporal", "desc": "Decodes complex visual motion. Spikes here mean fast camera movements, action sequences, or dynamic scene transitions captured viewer attention."},
    {"id": "TPQJ3", "name": "Temporo-Parietal Junction 3", "desc": "Theory of mind and social cognition. Activation suggests viewers are mentalizing — trying to understand characters' intentions or social narratives."},
    {"id": "V8", "name": "Visual Area 8", "desc": "Face-color processing area. High activation means the ad's faces and skin tones are being deeply encoded — critical for spokesperson effectiveness."},
    {"id": "STS", "name": "Superior Temporal Sulcus", "desc": "Voice identity and prosody. Tracks how distinctive or emotionally modulated the narrator's voice is."},
    {"id": "IFG", "name": "Inferior Frontal Gyrus", "desc": "Language comprehension and brand name encoding. Activation during taglines indicates successful verbal memory formation."},
    {"id": "AMY", "name": "Amygdala Complex", "desc": "Emotional arousal and threat detection. Spikes indicate the content triggered a strong emotional response — positive or negative."},
    {"id": "NAcc", "name": "Nucleus Accumbens", "desc": "Reward anticipation. Active when viewers experience desire for the product or positive brand associations."},
]

# Neural narrative transcript (Old Spice style)
TRANSCRIPT = [
    {"time": 0, "text": "Hello, ladies.", "event": "Hook — direct address"},
    {"time": 2, "text": "Look at your man.", "event": "Attention redirect"},
    {"time": 4, "text": "Now back to me.", "event": "Re-engagement"},
    {"time": 6, "text": "Now back at your man.", "event": "Pattern establish"},
    {"time": 8, "text": "Now BACK to me.", "event": "Pattern break — emphasis"},
    {"time": 10, "text": "Sadly, he isn't me.", "event": "Emotional contrast"},
    {"time": 12, "text": "But if he stopped using lady-scented body wash...", "event": "Problem statement"},
    {"time": 15, "text": "...he could smell like he's me.", "event": "Product promise — Limbic spike"},
    {"time": 18, "text": "Look down. Back up.", "event": "Scene transition"},
    {"time": 20, "text": "Where are you? You're on a boat.", "event": "Surprise — Ventral Attention peak"},
    {"time": 22, "text": "With the man your man could smell like.", "event": "Brand reinforcement"},
    {"time": 25, "text": "What's in your hand? Tickets to that thing you love.", "event": "Reward — NAcc spike"},
    {"time": 27, "text": "Look again — the tickets are now diamonds.", "event": "Surprise escalation"},
    {"time": 29, "text": "Anything is possible when your man smells like Old Spice.", "event": "CTA — brand encoding"},
    {"time": 31, "text": "I'm on a horse.", "event": "Final hook — memory anchor"},
]


def gen_network_signals(n):
    """Generate per-frame Yeo 7 activations for a 30s ad."""
    t = np.linspace(0, 2*np.pi, n)
    noise = lambda s=0.04: np.random.normal(0, s, n)
    
    s = {}
    # Ventral Attention (Hook/Surprise) — sharp spikes
    hook = np.zeros(n) + 0.15
    hook[0:3] += 0.8;  hook[8:10] += 0.5;  hook[15:18] += 0.75
    hook[20:22] += 0.9;  hook[27:29] += 0.7;  hook[31:] += 0.6
    s["ventral_attention"] = np.clip(hook + noise(0.05), 0, 1.4)

    # Limbic (Emotional Resonance)
    emo = np.zeros(n) + 0.2
    emo += 0.5 * np.exp(-((np.arange(n) - 15)**2) / 20)
    emo[10:14] += 0.4;  emo[20:25] += 0.5;  emo[29:] += 0.3
    s["limbic"] = np.clip(emo + noise(0.04), 0, 1.3)

    # Visual
    vis = 0.5 + 0.3 * np.sin(1.5*t) + noise(0.05)
    vis[0:5] += 0.3;  vis[18:22] += 0.35;  vis[27:30] += 0.25
    s["visual"] = np.clip(vis, 0, 1.2)

    # Dorsal Attention (Sustained)
    att = 0.4 + 0.2 * np.sin(t + 1) + noise(0.04)
    att[2:18] += 0.25;  att[26:] -= 0.1
    s["dorsal_attention"] = np.clip(att, 0, 1.0)

    # Somatomotor
    som = 0.15 + 0.1 * np.sin(0.8*t) + noise(0.03)
    som[1:4] += 0.2;  som[18:21] += 0.15
    s["somatomotor"] = np.clip(som, 0, 0.7)

    # Frontoparietal (Brand Recall)
    brand = 0.3 + 0.2 * np.sin(1.2*t + 2) + noise(0.04)
    brand[22:28] += 0.45;  brand[29:32] += 0.55
    s["frontoparietal"] = np.clip(brand, 0, 1.1)

    # Default Mode (Mind Wandering)
    dmn = 0.12 + noise(0.04)
    dmn[5:8] += 0.2;  dmn[26:] += 0.3
    s["default_mode"] = np.clip(dmn, 0, 0.8)

    return s


def gen_demographic_scores():
    """Generate cognitive dimension scores per demographic."""
    base = {
        "Gen Z":  [0.82, 0.71, 0.88, 0.65, 0.91, 0.73, 0.85, 0.79],
        "Adults": [0.74, 0.68, 0.72, 0.78, 0.76, 0.81, 0.69, 0.65],
        "Kids":   [0.89, 0.55, 0.93, 0.42, 0.84, 0.58, 0.77, 0.91],
        "Older":  [0.61, 0.73, 0.58, 0.82, 0.63, 0.85, 0.72, 0.54],
    }
    return {demo: {dim: round(val + np.random.normal(0, 0.03), 3)
                   for dim, val in zip(COGNITIVE_DIMS, vals)}
            for demo, vals in base.items()}


def gen_demographic_trajectories(n):
    """Generate per-demographic mean activation over time."""
    t = np.linspace(0, 2*np.pi, n)
    trajectories = {}
    for demo in DEMOGRAPHICS:
        base = 0.4 + np.random.uniform(-0.1, 0.1)
        amp = 0.2 + np.random.uniform(0, 0.15)
        phase = np.random.uniform(0, np.pi)
        signal = base + amp * np.sin(t + phase) + np.random.normal(0, 0.04, n)
        # Gen Z peaks higher in middle, Older decays
        if demo == "Gen Z":
            signal[8:20] += 0.15
        elif demo == "Older":
            signal[20:] -= 0.1
        elif demo == "Kids":
            signal += 0.05 * np.sin(3*t)  # More volatile
        trajectories[demo] = [round(float(v), 4) for v in np.clip(signal, 0, 1.2)]
    return trajectories


def gen_peak_scenes(signals, n):
    """Detect the top 6 high-impact scene frames."""
    combined = signals["ventral_attention"] * signals["limbic"]
    top_indices = np.argsort(combined)[-6:][::-1]
    scenes = []
    for rank, idx in enumerate(top_indices):
        scenes.append({
            "rank": rank + 1,
            "timestamp": int(idx),
            "peak_intensity": round(float(combined[idx]), 4),
            "hook_score": round(float(signals["ventral_attention"][idx]), 3),
            "emotion_score": round(float(signals["limbic"][idx]), 3),
            "visual_score": round(float(signals["visual"][idx]), 3),
            "label": TRANSCRIPT[min(idx, len(TRANSCRIPT)-1)]["text"] if idx < len(TRANSCRIPT) else "Scene",
        })
    global_max = max(s["peak_intensity"] for s in scenes)
    for s in scenes:
        s["pct_of_max"] = round(s["peak_intensity"] / global_max * 100, 1)
    return scenes, round(float(global_max), 4)


def gen_roi_rankings():
    """Generate ranked ROI activation scores."""
    scores = np.array([0.92, 0.87, 0.83, 0.78, 0.74, 0.69, 0.64, 0.58])
    scores += np.random.normal(0, 0.02, len(scores))
    ranked = []
    for i, roi in enumerate(ROIS):
        ranked.append({**roi, "score": round(float(scores[i]), 3)})
    return sorted(ranked, key=lambda r: r["score"], reverse=True)


def gen_variance_matrix():
    """Generate inter-subject variance heatmap: demographics x ROIs."""
    matrix = {}
    for demo in DEMOGRAPHICS:
        matrix[demo] = {}
        for roi in ROIS[:6]:
            # Lower variance = more consistent response
            variance = round(np.random.uniform(0.05, 0.45), 3)
            matrix[demo][roi["id"]] = variance
    return matrix


def gen_audio_insights():
    """Generate high-impact audio moments per demographic."""
    moments = {}
    for demo in DEMOGRAPHICS:
        demo_moments = []
        indices = np.random.choice(range(len(TRANSCRIPT)), size=3, replace=False)
        for rank, idx in enumerate(sorted(indices)):
            demo_moments.append({
                "rank": rank + 1,
                "timestamp": TRANSCRIPT[idx]["time"],
                "text": TRANSCRIPT[idx]["text"],
                "intensity": round(np.random.uniform(0.6, 1.0), 3),
            })
        moments[demo] = demo_moments
    return moments


def main():
    signals = gen_network_signals(N_FRAMES)

    # Build per-frame data
    frames = []
    for i in range(N_FRAMES):
        frame = {"timestamp_seconds": float(i), "networks": {}}
        for net in signals:
            frame["networks"][net] = {"activation": round(float(signals[net][i]), 4)}
        
        all_a = [signals[n][i] for n in signals]
        frame["global_mean"] = round(float(np.mean(all_a)), 4)
        frame["peak_activation"] = round(float(np.max(all_a)), 4)
        frame["engagement_score"] = round(float((signals["ventral_attention"][i] + signals["limbic"][i]) / 2), 4)
        frame["hemisphere_asymmetry"] = round(float(signals["frontoparietal"][i] - signals["default_mode"][i]), 4)
        frames.append(frame)

    # Peak scenes
    peak_scenes, global_max = gen_peak_scenes(signals, N_FRAMES)

    # Full export
    data = {
        "frames": frames,
        "demographics": {
            "scores": gen_demographic_scores(),
            "trajectories": gen_demographic_trajectories(N_FRAMES),
        },
        "peak_scenes": peak_scenes,
        "global_max_score": global_max,
        "rois": gen_roi_rankings(),
        "variance_matrix": gen_variance_matrix(),
        "audio_insights": gen_audio_insights(),
        "transcript": TRANSCRIPT,
        "metadata": {
            "duration_seconds": N_FRAMES - 1,
            "top_demographic": "Gen Z (13–24)",
            "model": "TRIBE v2 + Rewire Pipeline",
        }
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported comprehensive dashboard data to {OUTPUT}")
    print(f"  {N_FRAMES} frames, {len(peak_scenes)} peak scenes, {len(ROIS)} ROIs")
    print(f"  Peak engagement: t={peak_scenes[0]['timestamp']}s (intensity={peak_scenes[0]['peak_intensity']})")


if __name__ == "__main__":
    main()
