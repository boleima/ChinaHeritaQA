"""
Run this file to automatically generate questions in the order specified by q_list.
Comment out entries in q_list to skip the corresponding question types.
"""
import os
import sys
import glob
import subprocess

# ── Question types to generate ────────────────────────────────────────────────
q_list = [
    'q1',  # Which heritage site is shown in this image?
    'q2',  # Which image was likely taken at the given heritage site?
    'q3',  # Which brief introduction correctly matches this image?
    'q4',  # In which dynasty was the architectural complex in this image built?
    'q5',  # Which historical background description is correct for this image?
    'q6',  # Which main-function description is correct for this image?
    'q7',  # Which architectural-usage description is correct for this image?
]

# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    success, failed = [], []

    for q in q_list:
        matches = glob.glob(os.path.join(script_dir, f"{q}_*.py"))
        if not matches:
            print(f"[SKIP] No script found for {q}")
            failed.append(q)
            continue

        script_path = matches[0]
        script_name = os.path.basename(script_path)
        print(f"\n{'=' * 55}")
        print(f"[{q_list.index(q) + 1}/{len(q_list)}] Generating {q}: {script_name}")
        print('=' * 55)

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,   # ensures ../DATA/ paths resolve correctly
        )

        if result.returncode == 0:
            print(f"[OK] {q} generated successfully")
            success.append(q)
        else:
            print(f"[ERROR] {q} generation failed (exit code {result.returncode})")
            failed.append(q)

    print(f"\n{'=' * 55}")
    print(f"Done: {len(success)}/{len(q_list)}")
    if failed:
        print(f"Failed: {failed}")
