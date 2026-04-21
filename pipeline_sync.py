"""pipeline_sync.py

Orchestrateur du pipeline de synchronisation multi-caméras ZED.

Refonte :
- Fusion des anciennes étapes 1 (extraction timestamps) et 3 (conversion SVO->MP4)
  en une seule passe par SVO : chaque fichier .svo n'est ouvert qu'une fois.
- Parallélisation par caméra (ProcessPoolExecutor) pour les deux passes lourdes
  (SVO -> MP4 brut, puis MP4 brut -> MP4 réparé avec frames noires).
- Logging structuré (module `logging`) avec niveaux et sortie fichier optionnelle,
  à la place des prints + emojis.

Flux :
  SVO  --(pass 1 parallèle)-->  MP4 brut + timestamps
                                      |
                                      v
                             CSV global + graphes + détection FPS
                                      |
                                      v
  MP4 brut --(pass 2 parallèle)--> MP4 réparé (frames noires pour drops)
                                      |
                                      v
                            Sélection GUI + découpage synchro
"""
import os
import sys
import glob
import json
import logging
import argparse
import threading
import multiprocessing as mp
from queue import Empty
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


log = logging.getLogger("pipeline_sync")


# =============================================================================
# Suivi de progression partagé entre workers (ProcessPoolExecutor)
# =============================================================================
# Queue initialisée côté worker par `_init_worker_progress` (passé en
# `initializer` du ProcessPoolExecutor). Les workers y publient :
#   ("total", cam_name, N)  : nombre total de frames attendues
#   ("inc",   cam_name, k)  : +k frames traitées depuis le dernier envoi
#   ("done",  cam_name, _)  : worker terminé
_PROGRESS_Q = None
_PROGRESS_REPORT_EVERY = 20  # frames entre deux reports pour limiter l'IPC


def _init_worker_progress(q):
    global _PROGRESS_Q
    _PROGRESS_Q = q


def _progress_send(kind, cam_name, value=None):
    q = _PROGRESS_Q
    if q is None:
        return
    try:
        q.put_nowait((kind, cam_name, value))
    except Exception:
        pass


def _drain_progress(progress_q, n_workers_target, stop_event, step_label):
    """Consomme la queue de progression et affiche une barre tqdm par caméra."""
    bars = {}
    positions_used = 0
    done_count = 0

    while True:
        try:
            msg = progress_q.get(timeout=0.25)
        except Empty:
            if stop_event.is_set() and done_count >= n_workers_target:
                break
            continue
        if msg is None:
            break

        kind, cam, value = msg
        if kind == "total":
            if cam not in bars:
                pos = positions_used
                positions_used += 1
                bars[cam] = tqdm(
                    total=value if value else None,
                    desc=f"{step_label} {cam}",
                    position=pos,
                    leave=True,
                    unit="frame",
                    dynamic_ncols=True,
                )
        elif kind == "inc":
            bar = bars.get(cam)
            if bar is not None and value:
                bar.update(value)
        elif kind == "done":
            bar = bars.get(cam)
            if bar is not None:
                bar.close()
            done_count += 1
            if done_count >= n_workers_target:
                break

    for bar in bars.values():
        if not getattr(bar, "disable", False):
            bar.close()


# =============================================================================
# Logging
# =============================================================================
def setup_logging(level=logging.INFO, log_file=None):
    """Configure le logger racine : console + fichier optionnel."""
    fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True
    )


# =============================================================================
# ÉTAPE 1+3 fusionnées : SVO -> (timestamps, MP4 brut), parallélisée
# =============================================================================
def _extract_and_convert_worker(
    svo_path, output_mp4_path, start_frame=None, end_frame=None,
    crop_left_half=True, debug=False,
):
    """Worker exécuté dans un process séparé.

    Ouvre le SVO une seule fois, extrait les images (VIEW.LEFT), les écrit dans
    un MP4 et collecte les timestamps.

    Retour : (cam_name, timestamps_list, error_or_None)
    """
    # Imports locaux : le worker tourne dans un process distinct, et certains
    # imports (pyzed) peuvent avoir besoin d'un contexte frais à chaque spawn.
    import pyzed.sl as sl

    cam_name = os.path.basename(svo_path)

    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    # On ne retrieve que VIEW.LEFT : pas besoin du calcul depth (NEURAL par
    # défaut charge un réseau de neurones qui sature la GPU pour rien).
    init.depth_mode = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        return (cam_name, [], f"Ouverture SVO impossible : {svo_path}")

    # Total pour la barre de progression (peut être 0 si SDK ne sait pas)
    try:
        total_svo_frames = int(zed.get_svo_number_of_frames())
    except Exception:
        total_svo_frames = 0
    if end_frame is not None and total_svo_frames:
        total_svo_frames = min(total_svo_frames, end_frame + 1)
    _progress_send("total", cam_name, total_svo_frames)

    img = sl.Mat()
    writer = None
    timestamps = []
    frame_idx = 0
    last_reported = 0
    need_half_crop = False

    try:
        while True:
            # Stop-early si end_frame demandé
            if end_frame is not None and frame_idx > end_frame:
                break

            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                break

            if start_frame is not None and frame_idx < start_frame:
                frame_idx += 1
                continue

            zed.retrieve_image(img, sl.VIEW.LEFT)
            arr = img.get_data()
            if arr is None:
                frame_idx += 1
                continue

            # BGRA -> BGR si besoin
            frame = arr[:, :, :3] if (arr.ndim == 3 and arr.shape[2] == 4) else arr

            # Détection side-by-side au 1er frame (avant init du writer)
            if writer is None:
                src_h, src_w = frame.shape[:2]
                aspect = (src_w / src_h) if src_h else 0.0
                need_half_crop = crop_left_half and (aspect > 1.7) and (src_w > 2000)

            if need_half_crop:
                frame = frame[:, : frame.shape[1] // 2]

            # Init writer sur la première frame valide
            if writer is None:
                out_h, out_w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # FPS "placeholder" dans le header : la réparation (étape 4)
                # réécrira le MP4 au FPS cible en utilisant les timestamps.
                writer = cv2.VideoWriter(output_mp4_path, fourcc, 30.0, (out_w, out_h))
                if not writer.isOpened():
                    return (cam_name, [], f"VideoWriter ne s'ouvre pas : {output_mp4_path}")

            writer.write(frame)
            timestamps.append(
                zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            )
            frame_idx += 1

            if frame_idx - last_reported >= _PROGRESS_REPORT_EVERY:
                _progress_send("inc", cam_name, frame_idx - last_reported)
                last_reported = frame_idx

    except Exception as e:
        return (cam_name, timestamps, f"Exception : {e}")
    finally:
        if frame_idx > last_reported:
            _progress_send("inc", cam_name, frame_idx - last_reported)
        _progress_send("done", cam_name)
        if writer is not None:
            writer.release()
        zed.close()

    return (cam_name, timestamps, None)


def step_extract_and_convert_parallel(
    svo_files, output_mp4_dir, start_frame=None, end_frame=None,
    n_workers=None, debug=False,
):
    """Lance les workers SVO -> MP4 en parallèle. Retourne dict cam_name -> timestamps."""
    os.makedirs(output_mp4_dir, exist_ok=True)

    if n_workers is None:
        n_workers = min(len(svo_files), os.cpu_count() or 1)
    n_workers = max(1, n_workers)

    log.info(
        "Extraction+conversion : %d workers sur %d SVO -> %s",
        n_workers, len(svo_files), output_mp4_dir,
    )

    tasks = []
    for svo_path in svo_files:
        base = os.path.splitext(os.path.basename(svo_path))[0]
        mp4_out = os.path.join(output_mp4_dir, f"{base}.mp4")
        tasks.append((svo_path, mp4_out))

    all_timestamps = {}
    mgr = mp.Manager()
    progress_q = mgr.Queue()
    stop_event = threading.Event()
    drain_thread = threading.Thread(
        target=_drain_progress,
        args=(progress_q, len(tasks), stop_event, "Extraction"),
        daemon=True,
    )
    drain_thread.start()

    try:
        with logging_redirect_tqdm():
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker_progress,
                initargs=(progress_q,),
            ) as ex:
                futures = {
                    ex.submit(
                        _extract_and_convert_worker,
                        svo, mp4, start_frame, end_frame, True, debug,
                    ): (svo, mp4)
                    for svo, mp4 in tasks
                }
                for fut in as_completed(futures):
                    svo, mp4 = futures[fut]
                    try:
                        cam_name, timestamps, err = fut.result()
                    except Exception as e:
                        log.error("Worker crash sur %s : %s", svo, e)
                        continue
                    if err:
                        log.error("[%s] %s", cam_name, err)
                        continue
                    if not timestamps:
                        log.warning("[%s] aucune frame extraite", cam_name)
                        continue
                    all_timestamps[cam_name] = timestamps
                    log.info("[%s] %d frames -> %s", cam_name, len(timestamps), mp4)
    finally:
        stop_event.set()
        drain_thread.join(timeout=5)
        mgr.shutdown()

    return all_timestamps


def step_write_csv_and_graphs(all_timestamps, csv_path, graph_path):
    """Sauvegarde le CSV global + les deux graphes de diagnostic."""
    log.info("CSV global -> %s", csv_path)
    df = pd.DataFrame({k: pd.Series(v) for k, v in all_timestamps.items()})
    df.index.name = "Frame_Index"
    df.to_csv(csv_path)

    log.info("Graphiques -> %s", graph_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Delta-t par caméra
    for cam_name, times in all_timestamps.items():
        if len(times) > 1:
            deltas = [times[i] - times[i - 1] for i in range(1, len(times))]
            ax1.plot(range(1, len(times)), deltas, label=cam_name, alpha=0.7)
    ax1.set_title("Écart entre images (pics = drops / saturation Jetson)")
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Delta (ms)")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.grid(True)

    # Temps absolu (dérive)
    min_t = min(t[0] for t in all_timestamps.values() if t)
    for cam_name, times in all_timestamps.items():
        if times:
            rel = [(t - min_t) / 1000.0 for t in times]
            ax2.plot(range(len(times)), rel, label=cam_name)
    ax2.set_title("Évolution du temps (écarts = désynchro / courbes divergentes = dérive)")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Secondes écoulées")
    ax2.legend(loc="upper left", fontsize="small")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()


# =============================================================================
# ÉTAPE 2 : analyse timestamps + détection du FPS
# =============================================================================
def step_detect_fps(csv_path):
    """Calcule les métriques par caméra et retourne le FPS global détecté."""
    df = pd.read_csv(csv_path, index_col=0)
    all_deltas = []
    metrics = []

    for col in df.columns:
        ts = df[col].dropna().values
        if len(ts) < 2:
            continue
        d = np.diff(ts)
        all_deltas.extend(d)
        median_d = np.median(d)
        metrics.append({
            "Caméra": col,
            "Total Frames": len(ts),
            "Durée (sec)": round((ts[-1] - ts[0]) / 1000.0, 2),
            "Images Perdues": int(np.sum(d > 1.5 * median_d)),
            "Pire Gel (ms)": round(float(np.max(d)), 1),
        })

    log.info("Métriques par caméra :\n%s", pd.DataFrame(metrics).to_string(index=False))

    first = df.iloc[0].dropna()
    last = df.apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan)
    durations = last - first
    log.info("Écart max au démarrage (frame 0) : %.2f ms", first.max() - first.min())
    log.info("Dérive max accumulée : %.2f ms", durations.max() - durations.min())

    global_median = float(np.median(all_deltas))
    fps = round(1000.0 / global_median)
    log.info("FPS détecté : %d (médiane delta = %.2f ms)", fps, global_median)
    return float(fps)


# =============================================================================
# ÉTAPE 4 : réparation (insertion de frames noires + rotation), parallélisée
# =============================================================================
def parse_rotations(spec):
    """Parse une spec `--rotate` en dict {pattern: angle}.

    Formats acceptés :
      - "180"                    -> {"all": 180}  (toutes caméras)
      - "22516499=180"           -> {"22516499": 180}
      - "cam1=90,cam2=180"       -> {"cam1": 90, "cam2": 180}
      - "all=180,cam3=0"         -> toutes à 180° sauf cam3 à 0°
    """
    if not spec:
        return {}

    result = {}
    # Cas court : un seul angle sans '=' -> applique à tout
    if "=" not in spec:
        try:
            return {"all": int(spec.strip())}
        except ValueError:
            log.warning("Rotation ignorée (entier attendu) : %s", spec)
            return {}

    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            log.warning("Rotation ignorée (pas de '=') : %s", pair)
            continue
        key, val = pair.split("=", 1)
        try:
            angle = int(val.strip())
        except ValueError:
            log.warning("Angle invalide pour %s : %s", key, val)
            continue
        if angle not in (0, 90, 180, 270):
            log.warning("Angle non supporté pour %s : %d (attendu 0/90/180/270)", key, angle)
            continue
        result[key.strip()] = angle
    return result


def get_rotation_for_file(filename, rotation_map):
    """Priorité : nom exact > stem exact > substring > 'all' / '*'."""
    if not rotation_map:
        return 0
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    if base in rotation_map:
        return rotation_map[base]
    if stem in rotation_map:
        return rotation_map[stem]
    for pattern, angle in rotation_map.items():
        if pattern in ("all", "*"):
            continue
        if pattern in base:
            return angle
    return rotation_map.get("all", rotation_map.get("*", 0))


def _apply_rotation(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _compute_n_missing_per_gap(timestamps, fps_target):
    """Pour chaque gap ts[i] -> ts[i+1], nombre de frames noires à insérer.

    Logique strictement identique à `_repair_mp4_worker` ; partagée pour que les
    sidecars de drops décrivent exactement les frames effectivement insérées."""
    delta_target = 1000.0 / fps_target
    out = []
    for i in range(len(timestamps) - 1):
        n = int(round((timestamps[i + 1] - timestamps[i]) / delta_target)) - 1
        out.append(max(0, n))
    return out


def compute_repaired_dropped_indices(timestamps, fps_target):
    """Indices 0-based des frames noires dans le MP4 réparé."""
    n_missing_list = _compute_n_missing_per_gap(timestamps, fps_target)
    dropped = []
    pos = 0
    for i in range(len(timestamps)):
        pos += 1  # raw frame i écrite à l'index (pos - 1)
        if i < len(n_missing_list):
            for _ in range(n_missing_list[i]):
                dropped.append(pos)
                pos += 1
    return dropped


def _repair_mp4_worker(mp4_path, timestamps, fps_target, output_path, rotate_angle=0):
    """Worker : lit le MP4 brut, applique la rotation éventuelle, écrit un MP4
    "réparé" en insérant des frames noires là où les timestamps indiquent des drops."""
    cam_name = os.path.basename(mp4_path)
    n_missing_list = _compute_n_missing_per_gap(timestamps, fps_target)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return (cam_name, 0, f"Ouverture MP4 impossible : {mp4_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Rotation 90/270 swap les dimensions de sortie
    if rotate_angle in (90, 270):
        out_w, out_h = h, w
    else:
        out_w, out_h = w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_target, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        return (cam_name, 0, f"VideoWriter ne s'ouvre pas : {output_path}")

    black = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    frames_added = 0
    idx = 0
    last_reported = 0

    _progress_send("total", cam_name, len(timestamps))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if rotate_angle:
                frame = _apply_rotation(frame, rotate_angle)
            out.write(frame)

            if idx < len(n_missing_list):
                for _ in range(n_missing_list[idx]):
                    out.write(black)
                    frames_added += 1
            idx += 1

            if idx - last_reported >= _PROGRESS_REPORT_EVERY:
                _progress_send("inc", cam_name, idx - last_reported)
                last_reported = idx
    finally:
        if idx > last_reported:
            _progress_send("inc", cam_name, idx - last_reported)
        _progress_send("done", cam_name)
        cap.release()
        out.release()

    return (cam_name, frames_added, None)


def step_repair_parallel(csv_path, mp4_input_dir, mp4_output_dir, fps_target,
                         n_workers=None, rotation_map=None):
    """Lance la réparation en parallèle sur tous les MP4 bruts."""
    os.makedirs(mp4_output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)
    rotation_map = rotation_map or {}

    tasks = []
    for col in df.columns:
        base = col.replace(".svo", "")
        mp4_in = os.path.join(mp4_input_dir, f"{base}.mp4")
        mp4_out = os.path.join(mp4_output_dir, f"{base}_repaired.mp4")
        if not os.path.exists(mp4_in):
            log.warning("MP4 introuvable, ignoré : %s", mp4_in)
            continue
        timestamps = df[col].dropna().values.tolist()
        rot = get_rotation_for_file(mp4_in, rotation_map)
        if rot:
            log.info("[%s] rotation %d° appliquée", base, rot)
        tasks.append((mp4_in, timestamps, fps_target, mp4_out, rot))

    if not tasks:
        log.warning("Aucun MP4 à réparer")
        return

    if n_workers is None:
        n_workers = min(len(tasks), os.cpu_count() or 1)
    n_workers = max(1, n_workers)

    log.info("Réparation : %d workers sur %d MP4 -> %s", n_workers, len(tasks), mp4_output_dir)

    mgr = mp.Manager()
    progress_q = mgr.Queue()
    stop_event = threading.Event()
    drain_thread = threading.Thread(
        target=_drain_progress,
        args=(progress_q, len(tasks), stop_event, "Réparation"),
        daemon=True,
    )
    drain_thread.start()

    try:
        with logging_redirect_tqdm():
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker_progress,
                initargs=(progress_q,),
            ) as ex:
                futures = {ex.submit(_repair_mp4_worker, *t): t for t in tasks}
                for fut in as_completed(futures):
                    t = futures[fut]
                    try:
                        cam_name, frames_added, err = fut.result()
                    except Exception as e:
                        log.error("Worker crash sur %s : %s", t[0], e)
                        continue
                    if err:
                        log.error("[%s] %s", cam_name, err)
                    else:
                        log.info("[%s] %d frames noires insérées -> %s", cam_name, frames_added, t[3])
    finally:
        stop_event.set()
        drain_thread.join(timeout=5)
        mgr.shutdown()


# =============================================================================
# ÉTAPE 7 : sidecars JSON listant les frames noires du MP4 synced final
# =============================================================================
def step_write_dropped_sidecars(csv_timestamps, csv_selected, mp4_repaired_dir,
                                mp4_synced_dir, fps_target, overwrite=False):
    """Écrit un sidecar `<stem>.dropped.json` à côté de chaque MP4 synced.

    Le sidecar liste les indices 0-based des frames noires dans le MP4 synced
    final, calculés de façon déterministe à partir des timestamps SVO et de la
    sélection GUI — pas de détection pixel."""
    from cut_sync import _fast_frame_count, read_selected_frames

    if not os.path.isfile(csv_timestamps):
        log.error("Sidecars : CSV timestamps introuvable : %s", csv_timestamps)
        return
    if not os.path.isfile(csv_selected):
        log.error("Sidecars : CSV sélection introuvable : %s", csv_selected)
        return

    df = pd.read_csv(csv_timestamps, index_col=0)
    selected = read_selected_frames(csv_selected)
    if not selected:
        log.warning("Sidecars : aucune vidéo sélectionnée")
        return

    # Référence = sélection la plus précoce (identique à cut_sync) -> start_trim >= 0
    ref_frame = min(selected.values())

    per_cam = {}
    for col in df.columns:
        base = col.replace(".svo", "")
        repaired_name = f"{base}_repaired.mp4"
        if repaired_name not in selected:
            log.warning("[%s] absent de reference_frames.csv, ignoré", base)
            continue
        repaired_path = os.path.join(mp4_repaired_dir, repaired_name)
        if not os.path.exists(repaired_path):
            log.warning("[%s] MP4 réparé introuvable : %s", base, repaired_path)
            continue
        per_cam[base] = {
            "timestamps": df[col].dropna().values.tolist(),
            "repaired_name": repaired_name,
            "repaired_path": repaired_path,
            "selected": selected[repaired_name],
        }

    if not per_cam:
        log.warning("Sidecars : aucune caméra exploitable")
        return

    # start_trim et common_length : strictement la même logique que cut_sync.
    start_trim = {cam: d["selected"] - ref_frame for cam, d in per_cam.items()}
    frame_counts = {
        cam: _fast_frame_count(d["repaired_path"], fps_target)
        for cam, d in per_cam.items()
    }
    eff = [frame_counts[cam] - start_trim[cam]
           for cam in per_cam if frame_counts[cam] is not None]
    if not eff:
        log.error("Sidecars : impossible de calculer common_length (frame counts manquants)")
        return
    common_length = min(eff)

    log.info("Sidecars : start_trim=%s common_length=%d", start_trim, common_length)

    sidecar_paths = {
        cam: os.path.join(mp4_synced_dir, f"{cam}_repaired_synced.dropped.json")
        for cam in per_cam
    }
    if not overwrite and all(os.path.exists(p) for p in sidecar_paths.values()):
        log.info("Sidecars déjà présents (utiliser --overwrite ou --rerun-sidecars pour régénérer)")
        return

    os.makedirs(mp4_synced_dir, exist_ok=True)

    for cam, d in per_cam.items():
        repaired_dropped = compute_repaired_dropped_indices(d["timestamps"], fps_target)
        st = start_trim[cam]
        upper = st + common_length
        synced_dropped = sorted(
            r - st for r in repaired_dropped if st <= r < upper
        )

        synced_mp4 = os.path.join(mp4_synced_dir, f"{cam}_repaired_synced.mp4")
        actual = (
            _fast_frame_count(synced_mp4, fps_target)
            if os.path.exists(synced_mp4) else None
        )
        if actual is not None and abs(actual - common_length) > 1:
            log.warning(
                "[%s] total_frames divergent : calculé=%d, MP4 réel=%d",
                cam, common_length, actual,
            )

        payload = {
            "fps": int(round(fps_target)),
            "total_frames": common_length,
            "dropped_frame_indices": synced_dropped,
        }
        sidecar_path = sidecar_paths[cam]
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info(
            "[%s] %d frames noires listées (sur %d réparées) -> %s",
            cam, len(synced_dropped), len(repaired_dropped), sidecar_path,
        )


# =============================================================================
# ORCHESTRATEUR
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline SVO -> MP4 synchronisés (parallèle, logging)",
    )
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Dossier contenant les .svo")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Dossier de sortie (défaut = input-dir)")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None,
                        help="Forcer un FPS cible (sinon détection auto)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Nombre de workers parallèles (défaut = min(n_svo, n_cpu))")
    parser.add_argument("--rotate", default=None,
                        help="Rotation par caméra. Formats : "
                             "'180' (toutes), "
                             "'22516499=180' (une seule, match par substring), "
                             "'cam1=90,cam2=180' (plusieurs). "
                             "Angles supportés : 0, 90, 180, 270.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force la ré-exécution de chaque étape")
    parser.add_argument("--rerun-repair", action="store_true",
                        help="Refait uniquement les étapes 4 (réparation) et 6 "
                             "(découpage) sans re-lire les SVO. Pratique pour "
                             "tester de nouvelles rotations.")
    parser.add_argument("--rerun-sidecars", action="store_true",
                        help="Force la régénération des sidecars *.dropped.json "
                             "sans toucher aux MP4.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-file", default=None,
                        help="Fichier log (défaut = <output-dir>/pipeline.log)")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir if args.output_dir else INPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_file = args.log_file if args.log_file else os.path.join(OUTPUT_DIR, "pipeline.log")
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO, log_file=log_file)

    FICHIER_CSV = os.path.join(OUTPUT_DIR, "Analyse_Timestamps_Global.csv")
    FICHIER_GRAPHIQUE = os.path.join(OUTPUT_DIR, "Graphiques_Analyse.png")
    DOSSIER_SORTIE_MP4 = os.path.join(OUTPUT_DIR, "MP4_repares")
    OUT_SYNC_DIR = os.path.join(OUTPUT_DIR, "MP4_synced")

    rotation_map = parse_rotations(args.rotate)
    if rotation_map:
        log.info("Rotations configurées : %s", rotation_map)

    log.info("=" * 60)
    log.info("DÉMARRAGE pipeline — input=%s output=%s", INPUT_DIR, OUTPUT_DIR)
    log.info("=" * 60)

    # -------------------------------------------------------------------------
    # ÉTAPES 1+3 fusionnées : SVO -> MP4 brut + timestamps (parallèle)
    # -------------------------------------------------------------------------
    svo_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.svo")))
    if not svo_files:
        log.error("Aucun .svo trouvé dans %s", INPUT_DIR)
        return 1

    expected_mp4s = [
        os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(s))[0] + ".mp4")
        for s in svo_files
    ]
    all_mp4s_exist = all(os.path.exists(p) for p in expected_mp4s)

    if os.path.exists(FICHIER_CSV) and all_mp4s_exist and not args.overwrite:
        log.info("Étape 1+3 déjà faite (CSV + MP4 bruts présents)")
    else:
        all_timestamps = step_extract_and_convert_parallel(
            svo_files, OUTPUT_DIR,
            start_frame=args.start_frame, end_frame=args.end_frame,
            n_workers=args.workers, debug=args.debug,
        )
        if not all_timestamps:
            log.error("Aucune caméra exploitable — arrêt")
            return 1
        step_write_csv_and_graphs(all_timestamps, FICHIER_CSV, FICHIER_GRAPHIQUE)

    # -------------------------------------------------------------------------
    # ÉTAPE 2 : FPS
    # -------------------------------------------------------------------------
    if args.fps is not None:
        fps_detecte = args.fps
        log.info("FPS forcé par utilisateur : %s", fps_detecte)
    else:
        fps_detecte = step_detect_fps(FICHIER_CSV)

    # -------------------------------------------------------------------------
    # ÉTAPE 4 : réparation (parallèle)
    # -------------------------------------------------------------------------
    repair_needed = (
        args.overwrite
        or args.rerun_repair
        or not os.path.isdir(DOSSIER_SORTIE_MP4)
        or not glob.glob(os.path.join(DOSSIER_SORTIE_MP4, "*.mp4"))
    )
    if not repair_needed:
        log.info("Réparation déjà faite : %s", DOSSIER_SORTIE_MP4)
    else:
        step_repair_parallel(
            FICHIER_CSV, OUTPUT_DIR, DOSSIER_SORTIE_MP4, fps_detecte,
            n_workers=args.workers, rotation_map=rotation_map,
        )

    # -------------------------------------------------------------------------
    # ÉTAPE 5 : sélection GUI
    # -------------------------------------------------------------------------
    processing_dir = DOSSIER_SORTIE_MP4
    csv_sel = os.path.join(OUTPUT_DIR, "reference_frames.csv")

    if not args.overwrite and os.path.exists(csv_sel):
        log.info("Sélection références déjà faite : %s", csv_sel)
    else:
        try:
            from select_reference_gui import select_reference_for_videos
            log.info("Lancement GUI de sélection (Entrée pour valider chaque caméra)")
            # Si start/end fournis, les MP4 réparés contiennent déjà le segment :
            # on n'applique pas d'offset supplémentaire dans le GUI.
            ok = select_reference_for_videos(processing_dir, csv_sel)
            if not ok:
                log.warning("Sélection des références interrompue")
                return 1
        except Exception as e:
            log.error("GUI non disponible : %s", e)
            return 1

    # -------------------------------------------------------------------------
    # ÉTAPE 6 : découpage / alignement final
    # -------------------------------------------------------------------------
    sync_needed = (
        args.overwrite
        or args.rerun_repair
        or not os.path.isdir(OUT_SYNC_DIR)
        or not glob.glob(os.path.join(OUT_SYNC_DIR, "*_synced.mp4"))
    )
    if not sync_needed:
        log.info("Découpage final déjà fait : %s", OUT_SYNC_DIR)
    else:
        try:
            from cut_sync import cut_videos_to_align
            cut_videos_to_align(processing_dir, csv_sel, OUT_SYNC_DIR, fps=fps_detecte)
        except Exception as e:
            log.error("Découpage non exécuté : %s", e)
            return 1

    # -------------------------------------------------------------------------
    # ÉTAPE 7 : sidecars JSON des frames noires (déterministe, pour calibrator)
    # -------------------------------------------------------------------------
    sidecars_force = args.overwrite or args.rerun_repair or args.rerun_sidecars
    try:
        step_write_dropped_sidecars(
            FICHIER_CSV, csv_sel, DOSSIER_SORTIE_MP4, OUT_SYNC_DIR,
            fps_detecte, overwrite=sidecars_force,
        )
    except Exception as e:
        log.error("Sidecars non générés : %s", e)
        return 1

    log.info("=" * 60)
    log.info("PIPELINE TERMINÉ")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
