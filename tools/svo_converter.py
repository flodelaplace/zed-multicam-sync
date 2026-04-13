#!/usr/bin/env python3
"""
Convertisseur de fichiers SVO (ZED) vers AVI ou MP4
Utilise le SDK ZED (pyzed) pour lire les fichiers .svo
et OpenCV pour écrire la vidéo de sortie.

Prérequis:
    pip install pyzed opencv-python numpy
    (Le SDK ZED doit être installé sur le système)

Usage:
    python svo_converter.py --input fichier.svo --output fichier.mp4
    python svo_converter.py --input fichier.svo --output fichier.avi
    python svo_converter.py --input fichier.svo --output fichier.mp4 --view left
    python svo_converter.py --input fichier.svo --output fichier.mp4 --view depth
"""

import argparse
import sys
import os

try:
    import pyzed.sl as sl
except ImportError:
    print("❌ Erreur : Le SDK ZED (pyzed) n'est pas installé.")
    print("   Installez-le avec : pip install pyzed")
    print("   Ou téléchargez le SDK depuis : https://www.stereolabs.com/developers/release/")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("❌ Erreur : OpenCV n'est pas installé.")
    print("   Installez-le avec : pip install opencv-python")
    sys.exit(1)

import numpy as np


def get_video_codec(output_path: str):
    """Retourne le codec approprié selon l'extension du fichier de sortie."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID"), "AVI"
    elif ext == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v"), "MP4"
    else:
        raise ValueError(f"Extension non supportée : '{ext}'. Utilisez .avi ou .mp4")


def convert_svo(
    input_path: str,
    output_path: str,
    view: str = "left",
    verbose: bool = True
) -> bool:
    """
    Convertit un fichier SVO en AVI ou MP4.

    Args:
        input_path  : Chemin vers le fichier .svo source
        output_path : Chemin vers le fichier vidéo de sortie (.avi ou .mp4)
        view        : Vue à exporter ('left', 'right', 'depth', 'side_by_side')
        verbose     : Afficher la progression

    Returns:
        True si la conversion a réussi, False sinon
    """
    # Vérification du fichier d'entrée
    if not os.path.isfile(input_path):
        print(f"❌ Fichier introuvable : {input_path}")
        return False

    # Récupération du codec
    try:
        codec, fmt_name = get_video_codec(output_path)
    except ValueError as e:
        print(f"❌ {e}")
        return False

    # ── Initialisation de la caméra ZED en mode SVO ──────────────────────────
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(input_path)
    init_params.svo_real_time_mode = False          # Lecture aussi vite que possible
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # Activer la profondeur uniquement si nécessaire
    if view == "depth":
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    else:
        init_params.depth_mode = sl.DEPTH_MODE.NONE

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"❌ Impossible d'ouvrir le fichier SVO : {repr(err)}")
        return False

    # ── Informations sur la vidéo ────────────────────────────────────────────
    cam_info = zed.get_camera_information()
    fps      = cam_info.camera_configuration.fps
    width    = cam_info.camera_configuration.resolution.width
    height   = cam_info.camera_configuration.resolution.height
    total    = zed.get_svo_number_of_frames()

    # Ajustement des dimensions pour la vue côte à côte
    out_width  = width * 2 if view == "side_by_side" else width
    out_height = height

    if verbose:
        print(f"\n📹 Fichier SVO     : {input_path}")
        print(f"   Résolution      : {width}x{height}")
        print(f"   FPS             : {fps}")
        print(f"   Frames totales  : {total}")
        print(f"   Vue sélectionnée: {view}")
        print(f"\n💾 Fichier de sortie : {output_path} ({fmt_name})")
        print(f"   Résolution sortie : {out_width}x{out_height}\n")

    # ── Initialisation du VideoWriter ────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, codec, fps, (out_width, out_height))

    if not writer.isOpened():
        print("❌ Impossible d'ouvrir le VideoWriter. Vérifiez les codecs installés.")
        zed.close()
        return False

    # ── Buffers ZED ──────────────────────────────────────────────────────────
    image_left  = sl.Mat()
    image_right = sl.Mat()
    depth_map   = sl.Mat()

    runtime_params = sl.RuntimeParameters()
    frame_count    = 0
    dropped_frames = 0
    last_valid_frame = None  # Dernière frame valide pour combler les gaps

    # ── Boucle de conversion ─────────────────────────────────────────────────
    svo_position = 0
    while True:
        err = zed.grab(runtime_params)
        svo_position += 1

        if err == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # --- Récupération de l'image selon la vue choisie ---------------
            if view == "left":
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                frame = image_left.get_data()

            elif view == "right":
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                frame = image_right.get_data()

            elif view == "depth":
                zed.retrieve_image(depth_map, sl.VIEW.DEPTH)
                frame = depth_map.get_data()

            elif view == "side_by_side":
                zed.retrieve_image(image_left,  sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                left_data  = image_left.get_data()
                right_data = image_right.get_data()
                frame = np.hstack((left_data, right_data))

            else:
                print(f"❌ Vue inconnue : '{view}'")
                break

            # --- Conversion BGRA → BGR (OpenCV n'accepte pas l'alpha) --------
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            last_valid_frame = frame.copy()
            writer.write(frame)

            # --- Progression ------------------------------------------------
            if verbose and frame_count % 50 == 0:
                pct = (frame_count / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r   [{bar}] {pct:5.1f}%  ({frame_count}/{total} frames)", end="", flush=True)

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            if verbose:
                print(f"\r   [{'█'*20}] 100.0%  ({frame_count}/{total} frames)")
            break
        else:
            # --- Frame corrompue : on répète la dernière frame valide --------
            dropped_frames += 1
            if last_valid_frame is not None:
                writer.write(last_valid_frame)
                frame_count += 1
                if verbose:
                    print(f"\n⚠️  Frame corrompue à la position {svo_position} — frame précédente répétée")
            else:
                if verbose:
                    print(f"\n⚠️  Frame corrompue à la position {svo_position} — aucune frame valide disponible")

    # ── Nettoyage ────────────────────────────────────────────────────────────
    writer.release()
    zed.close()

    if verbose:
        print(f"\n✅ Conversion terminée ! {frame_count} frames exportées.")
        if dropped_frames > 0:
            print(f"   ⚠️  {dropped_frames} frames corrompues remplacées par la frame précédente.")
        else:
            print(f"   ✅ Aucune frame corrompue détectée.")
        print(f"   Fichier créé : {os.path.abspath(output_path)}\n")

    return True


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convertit un fichier SVO (ZED) en AVI ou MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python svo_converter.py -i enregistrement.svo -o sortie.mp4
  python svo_converter.py -i enregistrement.svo -o sortie.avi --view right
  python svo_converter.py -i enregistrement.svo -o sortie.mp4 --view depth
  python svo_converter.py -i enregistrement.svo -o sortie.mp4 --view side_by_side
        """
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Chemin vers le fichier .svo source"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Chemin vers le fichier de sortie (.avi ou .mp4)"
    )
    parser.add_argument(
        "--view",
        default="left",
        choices=["left", "right", "depth", "side_by_side"],
        help="Vue à exporter (défaut : left)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Désactiver l'affichage de la progression"
    )

    args = parser.parse_args()

    success = convert_svo(
        input_path=args.input,
        output_path=args.output,
        view=args.view,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
