"""check_first_frame_offset.py

Compare le décalage inter-caméras déduit du **timestamp de la 1ère frame SVO**
au décalage déduit de la **sélection manuelle d'un événement** dans le MP4
réparé (reference_frames.csv).

Question posée :
    Le ZED ne synchronise pas physiquement les caméras — il s'appuie sur les
    timestamps. Si on n'utilise QUE le timestamp de la première frame de chaque
    SVO pour aligner, est-ce qu'on retrouve le même alignement que celui obtenu
    en pointant manuellement un événement visuel commun ?

Subtilité critique : la frame sélectionnée par le GUI est l'index dans le MP4
RÉPARÉ (avec frames noires insérées pour combler les drops). Pour comparer
proprement, on remonte cette frame réparée à la frame brute correspondante,
puis on lit son timestamp SVO réel — pas un instant théorique calculé via fps.

Pour chaque caméra X (vs caméra de référence = sélection la plus précoce) :
    Δ_start  = ts_X[0] - ts_ref[0]
        ↳ le décalage qu'on déduirait du seul timestamp de départ
    Δ_event  = ts_X[i_X] - ts_ref[i_ref]
        ↳ le décalage réel à l'instant de l'événement (i_X = index brut de
          la frame sélectionnée, après compensation des frames noires)
    résidu   = Δ_event - Δ_start
        ↳ dérive accumulée. Si ≈ 0 → le timestamp de départ suffit.
                            Si ≫ 1 frame → la sélection manuelle apporte info.
"""
import os
import sys
import csv
import argparse
import logging

import numpy as np
import pandas as pd

# Permettre l'import du module pipeline_sync (parent du dossier tools/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_sync import _compute_n_missing_per_gap


log = logging.getLogger("check_offset")


def repaired_to_raw(selected_repaired, timestamps, fps):
    """Convertit un index dans le MP4 réparé vers (raw_idx, ts_réel).

    Renvoie (raw_idx, ts_ms, is_black) :
        - raw_idx : index dans la liste des timestamps SVO bruts
        - ts_ms   : timestamp SVO réel (ou interpolé si frame noire)
        - is_black : True si la frame sélectionnée tombe sur une frame noire

    Reproduit la logique d'écriture de `_repair_mp4_worker` : pour chaque frame
    brute i, on écrit raw[i] (position pos), puis n_missing[i] frames noires.
    """
    n_missing_list = _compute_n_missing_per_gap(timestamps, fps)
    delta_target = 1000.0 / fps
    pos = 0
    for i, ts in enumerate(timestamps):
        if pos == selected_repaired:
            return i, float(ts), False
        pos += 1  # frame brute écrite
        if i < len(n_missing_list):
            n = n_missing_list[i]
            if pos <= selected_repaired < pos + n:
                k = selected_repaired - pos  # 0..n-1
                interpolated = float(ts) + (k + 1) * delta_target
                return i, interpolated, True
            pos += n
    raise IndexError(
        f"selected_frame={selected_repaired} hors-limites (total réparé < {pos})"
    )


def read_selected(csv_path):
    """Lit reference_frames.csv -> {basename_repaired: selected_frame}."""
    sel = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            sel[row["video"]] = int(row["selected_frame"])
    return sel


def cam_id_from_repaired(name):
    """'22516499_repaired.mp4' -> '22516499'."""
    return name.replace("_repaired.mp4", "")


def detect_fps(df):
    deltas = []
    for col in df.columns:
        ts = df[col].dropna().values
        if len(ts) >= 2:
            deltas.extend(np.diff(ts))
    median = float(np.median(deltas))
    return round(1000.0 / median)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir",
                        help="Dossier contenant Analyse_Timestamps_Global.csv "
                             "et reference_frames.csv")
    parser.add_argument("--fps", type=float, default=None,
                        help="Forcer le FPS (sinon détection auto sur les ts)")
    parser.add_argument("--csv-out", default=None,
                        help="Si fourni, écrit un CSV détaillé à ce chemin")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    csv_ts = os.path.join(args.output_dir, "Analyse_Timestamps_Global.csv")
    csv_sel = os.path.join(args.output_dir, "reference_frames.csv")
    if not os.path.isfile(csv_ts):
        log.error("Introuvable : %s", csv_ts)
        return 1
    if not os.path.isfile(csv_sel):
        log.error("Introuvable : %s", csv_sel)
        return 1

    df = pd.read_csv(csv_ts, index_col=0)
    selected = read_selected(csv_sel)

    fps = args.fps if args.fps else detect_fps(df)
    log.info("FPS utilisé : %s", fps)

    # Mapping : cam_id -> (repaired_name, selected_frame, timestamps_array)
    cams = {}
    for repaired_name, sel_frame in selected.items():
        cam = cam_id_from_repaired(repaired_name)
        col = f"{cam}.svo"
        if col not in df.columns:
            log.warning("Colonne timestamps absente pour %s, ignorée", cam)
            continue
        ts = df[col].dropna().values
        cams[cam] = (repaired_name, sel_frame, ts)

    if not cams:
        log.error("Aucune caméra exploitable")
        return 1

    # Pour chaque caméra, remonter la frame sélectionnée -> timestamp SVO réel
    rows = []
    for cam, (repaired, sel_frame, ts) in cams.items():
        try:
            raw_idx, ts_event, is_black = repaired_to_raw(sel_frame, ts, fps)
        except IndexError as e:
            log.error("[%s] %s", cam, e)
            continue
        n_black_before = sel_frame - raw_idx  # # frames noires insérées avant
        rows.append({
            "cam": cam,
            "ts0": float(ts[0]),
            "selected_repaired": sel_frame,
            "raw_idx": raw_idx,
            "n_black_before": n_black_before,
            "ts_event": ts_event,
            "is_black": is_black,
        })

    # Référence = caméra avec selected_repaired le plus précoce
    # (cohérent avec cut_sync : tous les Δ_manuel >= 0)
    ref = min(rows, key=lambda r: r["selected_repaired"])
    log.info("Référence (sélection la plus précoce) : %s "
             "(selected=%d, ts0=%.0f)",
             ref["cam"], ref["selected_repaired"], ref["ts0"])

    delta_t_frame = 1000.0 / fps
    for r in rows:
        r["delta_start_ms"] = r["ts0"] - ref["ts0"]
        r["delta_manuel_ms"] = (r["selected_repaired"] - ref["selected_repaired"]) * delta_t_frame
        r["delta_event_ms"] = r["ts_event"] - ref["ts_event"]
        # résidu : ce que la sélection manuelle apporte EN PLUS du timestamp 1ère frame
        # (= dérive entre ts[0] et l'instant de l'événement)
        r["residu_ms"] = r["delta_event_ms"] - r["delta_start_ms"]
        r["residu_frames"] = r["residu_ms"] / delta_t_frame

    # Affichage
    rows_sorted = sorted(rows, key=lambda r: r["cam"])
    headers = ["cam", "ts0", "sel", "raw_idx", "d_start(ms)", "d_manuel(ms)",
               "d_event(ms)", "residu(ms)", "residu(fr)"]
    widths = [10, 14, 6, 8, 12, 13, 12, 12, 11]
    sep = " | "

    def fmt_row(vals):
        return sep.join(str(v).rjust(w) for v, w in zip(vals, widths))

    print()
    print(fmt_row(headers))
    print(sep.join("-" * w for w in widths))
    for r in rows_sorted:
        marker = " *" if r["cam"] == ref["cam"] else ""
        print(fmt_row([
            r["cam"] + marker,
            f"{r['ts0']:.0f}",
            r["selected_repaired"],
            r["raw_idx"],
            f"{r['delta_start_ms']:+.1f}",
            f"{r['delta_manuel_ms']:+.1f}",
            f"{r['delta_event_ms']:+.1f}",
            f"{r['residu_ms']:+.1f}",
            f"{r['residu_frames']:+.2f}",
        ]))
    print()
    print("* = camera de reference (selection la plus precoce)")
    print()
    print("Lecture :")
    print("  d_start  = decalage deduit du SEUL timestamp de la 1ere frame")
    print("  d_manuel = decalage deduit de la selection manuelle (theorique, "
          "selected*1000/fps)")
    print("  d_event  = decalage reel a l'instant de l'evenement "
          "(timestamp SVO de la frame brute correspondante)")
    print("  residu   = d_event - d_start : ce que l'horloge a derive entre "
          "ts[0] et l'evenement")
    print("             ~ 0       -> le timestamp de la 1ere frame suffit")
    print("             >> 1 frame -> derive d'horloge, la selection manuelle "
          "apporte de l'info")

    residus = [r["residu_ms"] for r in rows if r["cam"] != ref["cam"]]
    if residus:
        print()
        print(f"Résidu : moyenne = {np.mean(residus):+.2f} ms, "
              f"max abs = {max(abs(x) for x in residus):.2f} ms "
              f"({max(abs(x) for x in residus) / delta_t_frame:.2f} frames)")

    if args.csv_out:
        pd.DataFrame(rows_sorted).to_csv(args.csv_out, index=False)
        log.info("Détails CSV -> %s", args.csv_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
