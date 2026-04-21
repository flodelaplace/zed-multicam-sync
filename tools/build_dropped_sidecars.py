"""tools/build_dropped_sidecars.py

Génère rétroactivement les sidecars `*.dropped.json` pour un run pipeline_sync
existant, sans rejouer les étapes 1-6. Utile pour ajouter les sidecars à un
output déjà calculé.

Usage :
    python tools/build_dropped_sidecars.py --input-dir input_videos/test_02
    python tools/build_dropped_sidecars.py --input-dir output/run_X --fps 30
"""
import os
import sys
import argparse
import logging

# Permet `python tools/build_dropped_sidecars.py` depuis la racine du repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_sync import (  # noqa: E402
    setup_logging,
    step_detect_fps,
    step_write_dropped_sidecars,
)


log = logging.getLogger("build_dropped_sidecars")


def main():
    parser = argparse.ArgumentParser(
        description="Génère les sidecars *.dropped.json pour un run existant."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Dossier de sortie d'un run pipeline_sync "
                             "(contient Analyse_Timestamps_Global.csv, "
                             "reference_frames.csv, MP4_repares/, MP4_synced/)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Forcer un FPS cible (sinon redétection auto)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    out_dir = args.input_dir
    csv_timestamps = os.path.join(out_dir, "Analyse_Timestamps_Global.csv")
    csv_selected = os.path.join(out_dir, "reference_frames.csv")
    mp4_repaired_dir = os.path.join(out_dir, "MP4_repares")
    mp4_synced_dir = os.path.join(out_dir, "MP4_synced")

    for label, path in [
        ("Analyse_Timestamps_Global.csv", csv_timestamps),
        ("reference_frames.csv", csv_selected),
        ("MP4_repares/", mp4_repaired_dir),
        ("MP4_synced/", mp4_synced_dir),
    ]:
        if not os.path.exists(path):
            log.error("Fichier ou dossier manquant : %s (%s)", label, path)
            return 1

    fps = args.fps if args.fps is not None else step_detect_fps(csv_timestamps)
    log.info("FPS cible : %s", fps)

    step_write_dropped_sidecars(
        csv_timestamps, csv_selected, mp4_repaired_dir, mp4_synced_dir,
        fps, overwrite=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
