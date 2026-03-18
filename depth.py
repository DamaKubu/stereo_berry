import cv2
import numpy as np
import openai
import subprocess
import os
import sys
import tempfile

# --- OPENAI ---
openai.api_key = "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"  # set your key in env


# --- Parameters to tune ---
params = {
    "numDisparities": 64,
    "blockSize": 5,
    "uniquenessRatio": 5,
    "speckleWindowSize": 50
}

# --- Camera capture code template ---
CAPTURE_CODE_TEMPLATE = """
import cv2
import numpy as np

capL = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
capR = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

retL, frameL = capL.read()
retR, frameR = capR.read()

frameL = cv2.rotate(frameL, cv2.ROTATE_90_COUNTERCLOCKWISE)
frameR = cv2.rotate(frameR, cv2.ROTATE_90_CLOCKWISE)

grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoSGBM_create(
    numDisparities={numDisparities},
    blockSize={blockSize},
    uniquenessRatio={uniquenessRatio},
    speckleWindowSize={speckleWindowSize}
)
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# Save disparity for evaluation
np.save("disparity.npy", disparity)
"""

def run_capture(params):
    """Run capture code in a temporary file."""
    code = CAPTURE_CODE_TEMPLATE.format(**params)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    subprocess.run([sys.executable, fname], check=True)
    disparity = np.load("disparity.npy")
    os.remove(fname)
    return disparity

def evaluate_disparity(disparity):
    """Return fraction of valid pixels."""
    valid_frac = np.count_nonzero(disparity > 0) / disparity.size * 100
    print(f"Valid depth fraction: {valid_frac:.1f}%")
    return valid_frac

def ask_llm_for_params(params, valid_frac):
    """Ask OpenAI to adjust parameters."""
    prompt = f"""
    Current stereo parameters: {params}
    Depth quality (valid fraction): {valid_frac:.1f}%
    Suggest new numDisparities, blockSize, uniquenessRatio, speckleWindowSize
    to improve depth. Return only a Python dictionary.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    suggested = response.choices[0].message.content
    # Evaluate dictionary safely
    try:
        new_params = eval(suggested)
        return new_params
    except Exception as e:
        print("Failed to parse LLM output:", e)
        return params

# --- Iterative loop ---
max_iters = 10
for i in range(max_iters):
    print(f"--- Iteration {i} ---")
    disparity = run_capture(params)
    valid_frac = evaluate_disparity(disparity)
    
    if valid_frac > 50:  # arbitrary threshold
        print("Depth looks good! Stopping.")
        break
    else:
        print("Depth poor -> asking LLM to adjust parameters...")
        params = ask_llm_for_params(params, valid_frac)
