"""cut_sync.py

Découpe et aligne les MP4 réparés en utilisant un CSV de frames de référence
(colonnes : video, selected_frame).

Correctifs par rapport à la version précédente :
- Coupe frame-accurate : on utilise `-ss` APRÈS `-i` avec réencodage (ou un
  fallback OpenCV qui skip linéairement les frames). L'ancienne version utilisait
  `-ss` avec `-c copy`, qui seek au keyframe le plus proche (jusqu'à ±2s d'erreur
  sur du H.264 GOP long) — ça cassait silencieusement l'alignement construit par
  les étapes précédentes.
- Comptage de frames rapide : on tente d'abord `nb_frames` dans le header du
  stream (instantané), puis `duration * fps` comme fallback. `-count_frames`
  (qui décode toute la vidéo) n'est plus utilisé.
- Logging structuré (module `logging`) à la place des prints.

Algorithme :
- Charge les frames sélectionnées par vidéo.
- Prend comme référence la frame sélectionnée la plus précoce (pour que tous les
  start_trim soient >= 0).
- Pour chaque vidéo : start_trim = frame_selected - ref_frame.
- Calcule la longueur commune max = min(frame_counts - start_trim).
- Coupe chaque vidéo : [start_trim, start_trim + common_length).
"""
import os
import csv
import subprocess
import logging

import cv2


log = logging.getLogger(__name__)


# =============================================================================
# Utilitaires
# =============================================================================
def read_selected_frames(csv_path):
    sel = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sel[row["video"]] = int(row["selected_frame"])
    return sel


def _fast_frame_count(path, fps_fallback):
    """Compte les frames rapidement.

    Priorité :
    1. `ffprobe ... stream=nb_frames` — lit le header du conteneur, instantané.
       Renvoie parfois "N/A" ou rien, selon le muxer.
    2. `ffprobe ... format=duration` * fps — approximation raisonnable.
    3. OpenCV `CAP_PROP_FRAME_COUNT` — dernier recours.

    On évite volontairement `-count_frames` qui décode toute la vidéo.
    """
    # 1) nb_frames du stream (header)
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=30,
        )
        val = proc.stdout.strip()
        if val and val != "N/A":
            n = int(val)
            if n > 0:
                return n
    except (subprocess.SubprocessError, ValueError):
        pass

    # 2) duration * fps
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            capture_output=True, text=True, timeout=30,
        )
        dur = float(proc.stdout.strip())
        return int(round(dur * fps_fallback))
    except (subprocess.SubprocessError, ValueError):
        pass

    # 3) OpenCV
    try:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass

    return None


def _cut_opencv(inp, out, start_frame_idx, common_length, fps):
    """Fallback frame-accurate via OpenCV : skip linéaire puis lecture-écriture.

    On n'utilise PAS `CAP_PROP_POS_FRAMES` qui peut seek au keyframe précédent
    sur H.264 — on skip avec `.grab()` qui est beaucoup plus rapide que `.read()`
    (pas de décode image) et garantit l'index exact.
    """
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        log.error("OpenCV ne peut pas ouvrir %s", inp)
        return False

    cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    # Skip frame-accurate
    for _ in range(int(start_frame_idx)):
        if not cap.grab():
            cap.release()
            log.error("EOF atteint pendant le skip de %d frames dans %s", start_frame_idx, inp)
            return False

    ret, first = cap.read()
    if not ret:
        cap.release()
        log.error("Aucune frame lue à partir de l'index %d dans %s", start_frame_idx, inp)
        return False

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, cap_fps, (w, h))
    writer.write(first)
    written = 1
    target = common_length if common_length is not None else float("inf")

    while written < target:
        ret, fr = cap.read()
        if not ret:
            break
        writer.write(fr)
        written += 1

    writer.release()
    cap.release()
    log.info("OpenCV a écrit %d frames dans %s", written, out)
    return written > 0


# =============================================================================
# API principale
# =============================================================================
def cut_videos_to_align(dossier_videos, csv_selected, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    selected = read_selected_frames(csv_selected)
    videos = sorted(selected.keys())
    if not videos:
        raise ValueError("Aucune vidéo sélectionnée dans le CSV")

    # Référence = frame sélectionnée la plus précoce -> tous les start_trim >= 0
    ref_frame = min(selected[v] for v in videos)
    offsets = {v: selected[v] - ref_frame for v in videos}

    # Comptage rapide
    frame_counts = {
        v: _fast_frame_count(os.path.join(dossier_videos, v), fps)
        for v in videos
    }

    log.debug("Frames sélectionnées : %s", selected)
    log.debug("Offsets (vs frame la plus précoce) : %s", offsets)
    log.debug("Frame counts : %s", frame_counts)

    # Trim de début par vidéo
    start_trim = {v: max(0, offsets[v]) for v in videos}
    eff_lengths = {
        v: (frame_counts[v] - start_trim[v]) if frame_counts[v] is not None else None
        for v in videos
    }
    known = [l for l in eff_lengths.values() if l is not None]
    common_length = min(known) if known else None

    log.info("start_trim=%s  common_length=%s", start_trim, common_length)

    # Découpage frame-accurate
    for v in videos:
        inp = os.path.join(dossier_videos, v)
        out = os.path.join(output_dir, v.replace(".mp4", "_synced.mp4"))
        start_frame_idx = int(start_trim[v])

        # CRITIQUE : `-ss` APRÈS `-i` + ré-encodage pour garantir la frame exacte.
        # Avec `-c copy`, ffmpeg seek au keyframe le plus proche => erreur possible
        # allant jusqu'à la durée du GOP (parfois >1s).
        start_sec = start_frame_idx / fps

        cmd = ["ffmpeg", "-y", "-i", inp, "-ss", f"{start_sec:.6f}"]
        if common_length is not None:
            duration_sec = common_length / fps
            cmd += ["-t", f"{duration_sec:.6f}"]
        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", out,
        ]

        log.info("Découpage %s : start_frame=%d, length=%s", v, start_frame_idx, common_length)
        log.debug("Commande : %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            log.error("ffmpeg a échoué (rc=%d) sur %s : %s",
                      proc.returncode, inp, (proc.stderr or "")[:500])
            log.info("Tentative de fallback OpenCV pour %s", inp)
            if not _cut_opencv(inp, out, start_frame_idx, common_length, fps):
                log.error("Échec complet pour %s", inp)
            continue

        # Vérif post-ffmpeg : sortie non vide ?
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            log.warning("Sortie vide pour %s, tentative OpenCV", out)
            _cut_opencv(inp, out, start_frame_idx, common_length, fps)

    log.info("Découpage et synchronisation terminés : %s", output_dir)
    return True


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("dossier", help="Dossier contenant les MP4 réparés")
    parser.add_argument("csv", nargs="?", default=None,
                        help="CSV des frames sélectionnées (ou out si --extract-only)")
    parser.add_argument("out", nargs="?", default=None, help="Dossier de sortie")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--extract-only", action="store_true",
                        help="Simple extraction d'une plage sans alignement")
    parser.add_argument("--extract-start-frame", type=int, default=None)
    parser.add_argument("--extract-end-frame", type=int, default=None)
    args = parser.parse_args()

    # En mode --extract-only, on tolère la forme : cut_sync.py <dossier> <out>
    if args.extract_only and args.out is None and args.csv is not None:
        args.out = args.csv
        args.csv = None

    if args.extract_only:
        os.makedirs(args.out, exist_ok=True)
        for fname in os.listdir(args.dossier):
            if not fname.lower().endswith(".mp4"):
                continue
            inp = os.path.join(args.dossier, fname)
            out = os.path.join(args.out, fname.replace(".mp4", "_extracted.mp4"))

            start_frame_idx = args.extract_start_frame or 0
            start_sec = start_frame_idx / args.fps

            cmd = ["ffmpeg", "-y", "-i", inp, "-ss", f"{start_sec:.6f}"]
            if args.extract_end_frame is not None:
                duration_sec = (args.extract_end_frame - start_frame_idx) / args.fps
                cmd += ["-t", f"{duration_sec:.6f}"]
            # Ré-encodage pour une extraction frame-accurate.
            cmd += [
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", out,
            ]

            log.info("Extraction : %s", " ".join(cmd))
            subprocess.run(cmd)
        log.info("Extraction terminée")
    else:
        if args.csv is None:
            raise SystemExit("Erreur : un CSV de frames sélectionnées est requis "
                             "(sauf avec --extract-only)")
        cut_videos_to_align(args.dossier, args.csv, args.out, args.fps)
