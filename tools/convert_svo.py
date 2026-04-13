"""convert_svo.py

Convert .svo files to .mp4. First attempts to run the user-provided ffmpeg
template. If ffmpeg fails to read the .svo (common on some setups), the
script falls back to using the ZED Python SDK to read frames from the SVO
and write an MP4 using OpenCV.
"""
import os
import glob
import subprocess
import time

import cv2
import pandas as pd

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except Exception:
    ZED_AVAILABLE = False


def _convert_with_zed(svo_path, output_mp4, fps=30, crop_left_half='left', rotate=0, orient='auto', debug=False, start_frame=None, end_frame=None):
    """Fallback conversion using ZED SDK: read frames and write mp4 via OpenCV.

    Args:
        svo_path (str): input .svo file
        output_mp4 (str): output mp4 path
        fps (float): output framerate to use
        crop_left_half (bool): if True, crop to left half (matches ffmpeg crop=iw/2:ih:0:0)
    """
    if not ZED_AVAILABLE:
        print("⚠️ ZED SDK (pyzed.sl) non disponible — impossible de convertir via ZED")
        return False

    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(svo_path)
    init_parameters.svo_real_time_mode = False

    zed = sl.Camera()
    if zed.open(init_parameters) != sl.ERROR_CODE.SUCCESS:
        print(f"⚠️ Impossible d'ouvrir {svo_path} via ZED SDK")
        return False

    img = sl.Mat()
    out = None
    frame_count = 0
    frame_idx = 0
    start = time.time()
    # Try to get total frames from SDK (if available) to report progress
    total_svo_frames = None
    try:
        if hasattr(zed, 'get_svo_number_of_frames'):
            try:
                total_svo_frames = zed.get_svo_number_of_frames()
            except Exception:
                total_svo_frames = None
    except Exception:
        total_svo_frames = None
    # If user supplied explicit end_frame, we can compute exact number to write
    if (start_frame is not None) and (end_frame is not None):
        total_to_write = max(0, end_frame - start_frame + 1)
    else:
        total_to_write = None

    # Progress print interval (frames)
    progress_interval = 25 if debug else 50
    while True:
        # If an end_frame was requested, stop BEFORE grabbing the next frame
        # to avoid reading the whole SVO when we've already passed the desired range.
        if (end_frame is not None) and (frame_idx > end_frame):
            if debug:
                print(f"[DEBUG] pre-grab check: frame_idx={frame_idx} > end_frame={end_frame}, breaking")
            break
        err = zed.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            break
        zed.retrieve_image(img, sl.VIEW.LEFT)
        arr = img.get_data()
        if arr is None:
            continue
        # Ensure we have 3 channels
        if arr.ndim == 3 and arr.shape[2] == 4:
            frame = arr[:, :, :3]
        else:
            frame = arr

        # Decide whether cropping is needed: if the retrieved frame is very wide
        # (e.g. side-by-side 3840x1080), then apply half-crop. If the SDK already
        # returned the left image (≈1920x1080) we must NOT crop further.
        if out is None:
            # detect width from first valid frame
            src_h, src_w = frame.shape[0], frame.shape[1]
            aspect = float(src_w) / float(src_h) if src_h > 0 else 0.0
            # Detect side-by-side if very wide: use aspect ratio (e.g. 3840x1080 -> ~3.55)
            need_half_crop = (aspect > 1.7) and (src_w > 2000)
            if debug:
                print(f"[DEBUG] first frame size: {src_w}x{src_h}, aspect={aspect:.2f}, need_half_crop={need_half_crop}")
        # apply crop only when source is side-by-side and user requested a half crop
        if 'need_half_crop' in locals() and need_half_crop and crop_left_half in ('left', 'right'):
            w = frame.shape[1]
            if crop_left_half == 'left':
                frame = frame[:, : w // 2]
            else:
                frame = frame[:, w // 2:]

        # apply rotation if requested (rotate clockwise)
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # If orient==portrait and current is landscape, rotate to portrait
        if orient == 'portrait':
            fh, fw = frame.shape[:2]
            if fw > fh:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Initialize VideoWriter on first valid frame
        if out is None:
            out_height, out_width = frame.shape[0], frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if debug:
                base_name = os.path.splitext(os.path.basename(output_mp4))[0]
                first_frame_path = os.path.join(os.path.dirname(output_mp4), f"{base_name}_firstframe.png")
                try:
                    cv2.imwrite(first_frame_path, frame)
                    print(f"[DEBUG] saved first frame to {first_frame_path}")
                except Exception as e:
                    print(f"[DEBUG] failed to save first frame: {e}")
            out = cv2.VideoWriter(output_mp4, fourcc, fps, (out_width, out_height))
            print(f"[DEBUG] VideoWriter initialized with size: width={out_width}, height={out_height}, fps={fps}")
            if not out.isOpened():
                print(f"⚠️ Impossible d'ouvrir VideoWriter pour {output_mp4}")
                zed.close()
                return False

        # write frame only if within requested start/end (frame indices counted from 0)
        write_this = True
        if (start_frame is not None) and (frame_idx < start_frame):
            write_this = False
        if (end_frame is not None) and (frame_idx > end_frame):
            write_this = False

        if write_this:
            out.write(frame)
            frame_count += 1
            # Print progress / ETA periodically
            if frame_count % progress_interval == 0 or (total_to_write is not None and frame_count == total_to_write):
                elapsed_local = time.time() - start if (time.time() - start) > 0 else 0.0001
                avg_fps = frame_count / elapsed_local
                if total_to_write is not None and total_to_write > 0:
                    pct = 100.0 * frame_count / total_to_write
                    remaining = max(0, total_to_write - frame_count)
                    eta = remaining / avg_fps if avg_fps > 0 else float('inf')
                    print(f"[PROGRESS] wrote {frame_count}/{total_to_write} frames ({pct:.1f}%), elapsed={elapsed_local:.1f}s, avg={avg_fps:.2f} fps, ETA={eta:.1f}s")
                elif total_svo_frames is not None:
                    # show read progress relative to SVO length
                    pct = 100.0 * frame_idx / float(total_svo_frames) if total_svo_frames > 0 else 0.0
                    print(f"[PROGRESS] wrote {frame_count} frames (read {frame_idx}/{total_svo_frames} [{pct:.1f}%]), elapsed={elapsed_local:.1f}s, avg={avg_fps:.2f} fps")
                else:
                    print(f"[PROGRESS] wrote {frame_count} frames, read frame_idx={frame_idx}, elapsed={elapsed_local:.1f}s, avg={avg_fps:.2f} fps")

        frame_idx += 1

        # If an end_frame was requested, stop reading the SVO once we've passed it.
        # Without this, the ZED fallback would continue to read the entire SVO file
        # even if we only wanted a small segment, which makes it much slower.
        if (end_frame is not None) and (frame_idx > end_frame):
            if debug:
                print(f"[DEBUG] reached end_frame {end_frame}, stopping read (frame_idx={frame_idx})")
            break

    if out is not None:
        out.release()
    zed.close()
    elapsed = time.time() - start
    print(f"  ✅ Conversion via ZED terminée : {frame_count} frames écrites en {elapsed:.1f}s -> {output_mp4}")
    if frame_count == 0:
        # No frames were written — remove empty output if any and signal failure
        if os.path.exists(output_mp4):
            try:
                os.remove(output_mp4)
            except Exception:
                pass
        print(f"⚠️ Aucun frame écrit pour {svo_path} dans la plage demandée ({start_frame}-{end_frame})")
        return False
    return True


def convert_all_svo(dossier_travail, ffmpeg_template=None, overwrite=False, fallback_to_zed=True, zed_fps=30, crop='left', rotate=0, orient='auto', debug=False, start_frame=None, end_frame=None, output_dir=None):
    """Convert every .svo in dossier_travail to .mp4 using ffmpeg_template.

    If ffmpeg fails for a file, optionally fall back to the ZED SDK conversion.
    Returns a dict mapping svo_path -> dict(output, rc, method)
    """
    if ffmpeg_template is None:
        # default: quieter logs, crop left half, encode with libx264 veryfast preset
        ffmpeg_template = 'ffmpeg -hide_banner -loglevel error -i "{input}" -vf "crop=iw/2:ih:0:0" -c:v libx264 -preset veryfast -crf 18 "{output}"'

    results = {}
    svo_files = glob.glob(os.path.join(dossier_travail, "*.svo"))
    for svo_path in svo_files:
        base = os.path.splitext(os.path.basename(svo_path))[0]
        target_dir = output_dir if output_dir is not None else dossier_travail
        os.makedirs(target_dir, exist_ok=True)
        output_mp4 = os.path.join(target_dir, f"{base}.mp4")
        if os.path.exists(output_mp4) and not overwrite:
            results[svo_path] = {"output": output_mp4, "skipped": True, "rc": 0, "method": "skip"}
            continue

        # If user requested a partial range (start_frame or end_frame), prefer
        # the ZED SDK conversion because it reliably reads frames by index from
        # the SVO. ffmpeg seeking inside .svo is unreliable on some systems.
        if (start_frame is not None) or (end_frame is not None):
            if fallback_to_zed:
                # ensure output directory exists for ZED fallback
                os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
                if debug:
                    print(f"[DEBUG] start/end requested -> using ZED fallback for {svo_path} ({start_frame}-{end_frame})")
                ok = _convert_with_zed(svo_path, output_mp4, fps=zed_fps, crop_left_half=crop, rotate=rotate, orient=orient, debug=debug, start_frame=start_frame, end_frame=end_frame)
                results[svo_path] = {"output": output_mp4, "rc": 0 if ok else 1, "skipped": False, "method": "zed" if ok else "zed_failed"}
                # move to next file
                continue
            # else: fall back to trying ffmpeg below

        # Try ffmpeg first. Build command and, if start/end requested, try to limit frames.
        cmd = ffmpeg_template.format(input=svo_path, output=output_mp4)

        # If start/end provided, attempt to seek by timestamp using the global CSV
        # (Analyse_Timestamps_Global.csv) which is generated by the pipeline. If we
        # find a timestamp for the requested start_frame we insert -ss before -i
        # and use -frames:v to limit the number of frames. This is usually much
        # faster than using a select filter when start>0.
        if (start_frame is not None) and (end_frame is not None):
            # search CSV in output_dir first, then in the input folder
            csv_candidates = []
            if output_dir is not None:
                csv_candidates.append(os.path.join(output_dir, 'Analyse_Timestamps_Global.csv'))
            csv_candidates.append(os.path.join(dossier_travail, 'Analyse_Timestamps_Global.csv'))
            csv_path = None
            for c in csv_candidates:
                if os.path.exists(c):
                    csv_path = c
                    break
            if csv_path is not None:
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    colname = os.path.basename(svo_path)
                    if colname in df.columns:
                        series = df[colname].dropna()
                        if len(series) > start_frame:
                            ts_ms = float(series.iloc[start_frame])
                            start_sec = ts_ms / 1000.0
                            n_frames = end_frame - start_frame + 1
                            in_quoted = f'"{svo_path}"'
                            pattern = f'-i {in_quoted}'
                            if pattern in cmd:
                                cmd = cmd.replace(pattern, f'-ss {start_sec:.6f} -i {in_quoted}')
                            else:
                                cmd = f'-ss {start_sec:.6f} ' + cmd
                            out_quoted = f'"{output_mp4}"'
                            if out_quoted in cmd:
                                cmd = cmd.replace(out_quoted, f'-frames:v {n_frames} {out_quoted}')
                            else:
                                cmd = cmd + f' -frames:v {n_frames}'
                            if debug:
                                print(f"[DEBUG] Using CSV seek: csv={csv_path} start_sec={start_sec:.6f} n_frames={n_frames}")
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] failed to read CSV for seek: {e}")

        # If CSV-based seek wasn't applied above, fall back to existing behaviour
        if (start_frame is not None) or (end_frame is not None):
            s = start_frame if start_frame is not None else 0
            e = end_frame if end_frame is not None else None
            # If start==0 and end specified, use -frames:v for fast early stop
            if s == 0 and e is not None:
                n_frames = e - s + 1
                # insert -frames:v N before the final output filename
                out_quoted = f'"{output_mp4}"'
                if out_quoted in cmd:
                    cmd = cmd.replace(out_quoted, f'-frames:v {n_frames} {out_quoted}')
                else:
                    cmd = cmd + f' -frames:v {n_frames}'
            else:
                # inject a select filter into -vf to pick frames between n=s..e
                sel_end = e if e is not None else 999999999
                select_filter = f"select='between(n,{s},{sel_end})',setpts=N/FRAME_RATE/TB"
                if '-vf' in cmd:
                    cmd = cmd.replace('-vf "', f'-vf "{select_filter},')
                else:
                    # append -vf filter before output
                    parts = cmd.rsplit('"', 1)
                    if len(parts) == 2:
                        cmd = parts[0] + f' -vf "{select_filter}" "' + parts[1]
                    else:
                        cmd = cmd + f' -vf "{select_filter}"'

        print(f"Exécution: {cmd}")
        proc = subprocess.run(cmd, shell=True)
        rc = proc.returncode
        if rc == 0:
            results[svo_path] = {"output": output_mp4, "rc": rc, "skipped": False, "method": "ffmpeg"}
            continue

        print(f"⚠️ ffmpeg a retourné {rc} pour {svo_path} — tentative de fallback via ZED SDK")
        # Fallback
        if fallback_to_zed:
            # ensure output directory exists for ZED fallback
            os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
            ok = _convert_with_zed(svo_path, output_mp4, fps=zed_fps, crop_left_half=crop, rotate=rotate, orient=orient, debug=debug, start_frame=start_frame, end_frame=end_frame)
            results[svo_path] = {"output": output_mp4, "rc": 0 if ok else rc, "skipped": False, "method": "zed" if ok else "ffmpeg_failed"}
        else:
            results[svo_path] = {"output": output_mp4, "rc": rc, "skipped": False, "method": "ffmpeg_failed"}

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dossier', help='Dossier contenant les .svo')
    parser.add_argument('--output-dir', default=None, help='(optionnel) dossier de sortie pour les mp4 (par défaut = dossier des svo)')
    parser.add_argument('--start-frame', type=int, default=None, help='(optionnel) frame de départ à extraire')
    parser.add_argument('--end-frame', type=int, default=None, help='(optionnel) frame de fin à extraire')
    parser.add_argument('--ffmpeg-template', help='Template ffmpeg avec {input} et {output}',
                        default='ffmpeg -i "{input}" -vf "crop=iw/2:ih:0:0" "{output}"')
    parser.add_argument('--overwrite', action='store_true', help='Écrase les mp4 existants')
    parser.add_argument('--no-zed-fallback', action='store_true', help="Désactive la conversion via ZED si ffmpeg échoue")
    parser.add_argument('--zed-fps', type=float, default=30.0, help='FPS utilisé par la conversion ZED fallback')
    parser.add_argument('--crop', choices=['left', 'right', 'none'], default='left', help='Couper la moitié gauche/droite ou none')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0, help='Rotation à appliquer au rendu final (degrés, clockwise)')
    parser.add_argument('--orient', choices=['portrait', 'landscape', 'auto'], default='auto', help='Forcer orientation/rotation (portrait fera rotation si utile)')
    parser.add_argument('--debug', action='store_true', help='Affiche des infos de debug (taille du 1er frame...)')
    args = parser.parse_args()
    # Prepare ffmpeg template modifications: if cropping left/right is requested and the default template is used,
    # ensure the crop filter matches the choice. For left: crop=iw/2:ih:0:0. For right: crop=iw/2:ih:iw/2:0
    ffmpeg_template = args.ffmpeg_template
    if args.ffmpeg_template == 'ffmpeg -i "{input}" -vf "crop=iw/2:ih:0:0" "{output}"':
        if args.crop == 'left':
            ffmpeg_template = 'ffmpeg -i "{input}" -vf "crop=iw/2:ih:0:0" "{output}"'
        elif args.crop == 'right':
            ffmpeg_template = 'ffmpeg -i "{input}" -vf "crop=iw/2:ih:iw/2:0" "{output}"'
        else:
            ffmpeg_template = 'ffmpeg -i "{input}" "{output}"'

    # If rotation is needed for ffmpeg approach, append transpose filters (90 CW => transpose=1)
    if args.rotate != 0:
        rotate_map = {90: 'transpose=1', 180: 'transpose=1,transpose=1', 270: 'transpose=2'}
        rotate_filter = rotate_map.get(args.rotate)
        # inject rotate into ffmpeg_template if it contains -vf, else add -vf
        if '-vf' in ffmpeg_template:
            ffmpeg_template = ffmpeg_template.replace('-vf "', f'-vf "{rotate_filter},')
        else:
            # replace the literal "{output}" substring with -vf "rotate" "{output}"
            ffmpeg_template = ffmpeg_template.replace('"{output}"', f'-vf "{rotate_filter}" "{{output}}"')

    res = convert_all_svo(args.dossier, ffmpeg_template=ffmpeg_template, overwrite=args.overwrite, fallback_to_zed=not args.no_zed_fallback, zed_fps=args.zed_fps, crop=args.crop, rotate=args.rotate, orient=args.orient, debug=args.debug, start_frame=args.start_frame, end_frame=args.end_frame, output_dir=args.output_dir)
    for k, v in res.items():
        print(k, '->', v)
