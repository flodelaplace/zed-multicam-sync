"""utils_rotate_ffmpeg.py

Petit utilitaire pour appliquer une rotation de 90° (horloge ou anti-horloge) à tous
les fichiers vidéo présents dans un dossier en utilisant ffmpeg.

Usage:
    python utils_rotate_ffmpeg.py -i "C:\path\to\videos"          # rotate 90° clockwise, outputs to <input>/rotated
    python utils_rotate_ffmpeg.py -i "C:\path\to\videos" -o "C:\out" --angle 270 --overwrite

Arguments principaux:
    -i / --input-dir    : dossier contenant les vidéos (requis)
    -o / --output-dir   : dossier de sortie (par défaut <input>/rotated)
    --angle             : angle en degrés (90 ou 270). 90 = clockwise, 270 = counter-clockwise
    --pattern           : pattern glob pour sélectionner les fichiers (par défaut '*.*' et filtrés par extensions vidéo communes)
    --recursive         : parcourir récursivement les sous-dossiers
    --overwrite         : écraser les sorties existantes
    --dry-run           : afficher les commandes ffmpeg sans les exécuter

Le script vérifie que `ffmpeg` est disponible dans le PATH.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import glob
import ffmpeg

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg', '.wmv', '.flv'}


def find_videos(input_dir, pattern='*.*', recursive=False):
    search = os.path.join(input_dir, '**', pattern) if recursive else os.path.join(input_dir, pattern)
    matches = glob.glob(search, recursive=recursive)
    vids = [p for p in matches if Path(p).suffix.lower() in VIDEO_EXTS and os.path.isfile(p)]
    return sorted(vids)


def ffmpeg_rotate_cmd(input_path, output_path, angle, debug=False):
    # map angle to transpose value
    # transpose=1 rotate 90° clockwise
    # transpose=2 rotate 90° counter-clockwise
    if angle == 90:
        vf = 'transpose=1'
    elif angle == 270:
        vf = 'transpose=2'
    else:
        raise ValueError('Angle non supporté: utilisez 90 ou 270')

    # Use -movflags +faststart for mp4 streaming friendliness
    cmd = [
        'ffmpeg',
        '-hide_banner',
    ]

    if not debug:
        cmd.extend(['-loglevel', 'error'])

    cmd.extend([
        '-i', input_path,
        '-vf', vf,
        '-c:a', 'copy',  # copy audio stream to be fast
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'veryfast',
        '-movflags', '+faststart',
        output_path
    ])
    return cmd


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Rotate videos by 90° using ffmpeg (batch)')
    parser.add_argument('-i', '--input-dir', required=True, help='Dossier contenant les vidéos')
    parser.add_argument('-o', '--output-dir', default=None, help='Dossier de sortie (par défaut: <input>/rotated)')
    parser.add_argument('--angle', type=int, choices=[90, 270], default=90, help='Angle en degrés: 90 ou 270 (default 90 clockwise)')
    parser.add_argument('--pattern', default='*.*', help='Glob pattern pour sélectionner les fichiers (default *.*)')
    parser.add_argument('--recursive', action='store_true', help='Parcourir récursivement les sous-dossiers')
    parser.add_argument('--overwrite', action='store_true', help='Écraser les fichiers de sortie existants')
    parser.add_argument('--dry-run', action='store_true', help="Afficher les commandes ffmpeg sans les exécuter")
    parser.add_argument('--ffmpeg-path', default=None, help='Chemin complet vers ffmpeg.exe si non dans le PATH')
    parser.add_argument('--debug', action='store_true', help='Afficher la sortie complète de ffmpeg en cas d\'erreur')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"❌ Dossier d'entrée introuvable: {input_dir}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(input_dir, 'rotated')
    ensure_dir_exists(output_dir)

    # check ffmpeg (optionnel: chemin fourni par l'utilisateur)
    ffmpeg_bin = None
    if args.ffmpeg_path:
        ffmpeg_bin = os.path.abspath(args.ffmpeg_path)
        if not os.path.exists(ffmpeg_bin):
            print(f"❌ Chemin ffmpeg fourni introuvable: {ffmpeg_bin}")
            sys.exit(1)
    else:
        ffmpeg_bin = shutil.which('ffmpeg')
        if ffmpeg_bin is None:
            print('❌ ffmpeg introuvable dans le PATH. Installez ffmpeg ou fournissez --ffmpeg-path et réessayez.')
            sys.exit(1)

    print(f"ℹ️  ffmpeg utilisé : {ffmpeg_bin}")

    videos = find_videos(input_dir, pattern=args.pattern, recursive=args.recursive)
    if not videos:
        print('❌ Aucune vidéo trouvée avec le pattern et extensions connues dans', input_dir)
        sys.exit(0)

    print(f"Found {len(videos)} video(s) in {input_dir}")
    failures = []

    for vid in videos:
        rel = os.path.relpath(vid, input_dir)
        out_path = os.path.join(output_dir, rel)
        out_dir = os.path.dirname(out_path)
        ensure_dir_exists(out_dir)

        # if same input and output path (shouldn't happen), avoid overwrite
        if os.path.abspath(vid) == os.path.abspath(out_path):
            print(f"⚠️ Ignoré (entrée == sortie) : {vid}")
            continue

        if os.path.exists(out_path) and not args.overwrite:
            print(f"⏭️  Déjà existant, saut : {out_path}")
            continue

        # build command using explicit ffmpeg binary
        cmd = ffmpeg_rotate_cmd(vid, out_path, args.angle, debug=args.debug)
        cmd[0] = ffmpeg_bin
        
        # Utiliser la méthode officielle de Windows pour formatter la commande
        cmd_str = subprocess.list2cmdline(cmd)
        print(f"▶️  Rotating: {rel} -> {os.path.relpath(out_path, output_dir)} (angle={args.angle})")
        if args.dry_run:
            print('    [dry-run] ffmpeg command:', cmd_str)
            continue

        try:
            # capture output for debug if needed
            # shell=True force Python à utiliser le shell, imitant le comportement de votre terminal
            proc = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                print(f"❌ ffmpeg failed for {vid} (rc={proc.returncode})")
                if args.debug:
                    print('---- ffmpeg stdout ----')
                    print(proc.stdout)
                    print('---- ffmpeg stderr ----')
                    print(proc.stderr)
                failures.append((vid, proc.returncode))
            else:
                if args.debug:
                    print(proc.stdout)
                    print(proc.stderr)
                print(f"✅ OK: {out_path}")
        except Exception as e:
            print(f"❌ Exception for {vid}: {e}")
            failures.append((vid, str(e)))

    print('\nDone.')
    if failures:
        print(f"{len(failures)} erreur(s) :")
        for f in failures:
            print(' -', f)
        sys.exit(2)


if __name__ == '__main__':
    main()
