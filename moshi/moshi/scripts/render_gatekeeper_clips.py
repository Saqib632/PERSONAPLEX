# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""One-time offline utility to render the PersonaFlex gatekeeper's two fixed
rejection messages to WAV clips, served at runtime by `gatekeeper.RejectionClips`.

Not part of the server runtime -- run this once, commit the resulting WAV
files under assets/gatekeeper/, and point --gatekeeper-reject-*-clip at them
(the server's defaults already point there).

Requires: pip install pyttsx3
Usage:    python -m moshi.scripts.render_gatekeeper_clips
"""

from pathlib import Path

from ..gatekeeper import REJECTION_TEXT, Verdict

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "assets" / "gatekeeper"

FILENAMES = {
    Verdict.FAIL_LANGUAGE: "reject_language.wav",
    Verdict.FAIL_TOPIC: "reject_topic.wav",
}


def main() -> None:
    import pyttsx3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    for verdict, filename in FILENAMES.items():
        out_path = OUTPUT_DIR / filename
        engine.save_to_file(REJECTION_TEXT[verdict], str(out_path))
    engine.runAndWait()
    for filename in FILENAMES.values():
        print(f"wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
