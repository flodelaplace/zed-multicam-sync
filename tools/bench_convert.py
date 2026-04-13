#!/usr/bin/env python3
"""bench_convert.py

Quick benchmark script to compare ffmpeg vs ZED-fallback conversion for a short
segment (by number of frames) of a .svo.

Usage example (PowerShell):
  python bench_convert.py "E:\2024-02-05-09-09-06\22516499.svo" --start-frame 0 --n-frames 100 --output-dir "C:\temp\bench"

This script will:
 - run ffmpeg attempt and measure time (will use -frames:v to limit frames)
 - run ZED SDK fallback (_convert_with_zed) and measure time (if pyzed available)
 - report duration, return code, and output file size for each method

Note: ffmpeg may not be able to read .svo on some setups; in that case the ffmpeg
attempt will fail quickly and the ZED fallback is the reliable method.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def run_ffmpeg(ffmpeg_path, input_svo, n_frames, output_path, crop_filter=True):
    cmd = [ffmpeg_path, '-hide_banner', '-loglevel', 'error', '-i', input_svo]
    # limit frames
    cmd += ['-frames:v', str(n_frames)]
    if crop_filter:
        cmd += ['-vf', 'crop=iw/2:ih:0:0']
    # encoding options (fast, reasonable quality)
    cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-y', output_path]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    rc = proc.returncode
    size = os.path.getsize(output_path) if (rc == 0 and os.path.exists(output_path)) else 0
    return {'method': 'ffmpeg', 'rc': rc, 'time_s': dt, 'size_bytes': size, 'cmd': ' '.join(cmd)}


def run_zed(convert_module, input_svo, start_frame, end_frame, output_path, fps=30.0, crop='left', rotate=0, orient='auto', debug=False):
    # convert_module is the imported module convert_svo
    _convert = getattr(convert_module, '_convert_with_zed', None)
    if _convert is None:
        return {'method': 'zed', 'rc': 1, 'time_s': 0.0, 'size_bytes': 0, 'error': 'no _convert_with_zed available'}

    t0 = time.perf_counter()
    ok = _convert(input_svo, output_path, fps=fps, crop_left_half=crop, rotate=rotate, orient=orient, debug=debug, start_frame=start_frame, end_frame=end_frame)
    dt = time.perf_counter() - t0
    rc = 0 if ok else 1
    size = os.path.getsize(output_path) if (rc == 0 and os.path.exists(output_path)) else 0
    return {'method': 'zed', 'rc': rc, 'time_s': dt, 'size_bytes': size}


def human_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024.0:
            return f"{b:.1f}{unit}"
        b /= 1024.0
    return f"{b:.1f}TB"


def main():
    parser = argparse.ArgumentParser(description='Benchmark ffmpeg vs ZED conversion on a .svo for N frames')
    parser.add_argument('input_svo', help='Path to .svo file')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame index (0-based)')
    parser.add_argument('--n-frames', type=int, default=100, help='Number of frames to convert')
    parser.add_argument('--output-dir', default=None, help='Directory to write outputs (default: same folder as SVO)')
    parser.add_argument('--ffmpeg-path', default=None, help='Path to ffmpeg executable (default: auto-detect from PATH or conda env)')
    parser.add_argument('--ffmpeg-encoder', default='libx264', choices=['libx264','h264_nvenc'], help='ffmpeg video encoder to use')
    parser.add_argument('--no-crop', dest='crop', action='store_false', help='Do not apply default crop filter')
    parser.add_argument('--debug', action='store_true', help='Enable debug for ZED conversion')
    parser.add_argument('--prefer-zed', action='store_true', help='Skip ffmpeg test and only run ZED fallback')
    args = parser.parse_args()

    input_svo = os.path.abspath(args.input_svo)
    if not os.path.exists(input_svo):
        print('Input SVO not found:', input_svo)
        raise SystemExit(1)

    start = args.start_frame
    n = args.n_frames
    end = start + n - 1

    out_dir = args.output_dir if args.output_dir else os.path.dirname(input_svo)
    os.makedirs(out_dir, exist_ok=True)

    ff_out = os.path.join(out_dir, f"bench_ffmpeg_{Path(input_svo).stem}_{start}_{end}.mp4")
    zed_out = os.path.join(out_dir, f"bench_zed_{Path(input_svo).stem}_{start}_{end}.mp4")

    results = []

    if not args.prefer_zed:
        # Auto-detect ffmpeg: prefer explicit path, then PATH, then common conda env location
        ffpath = None
        if args.ffmpeg_path:
            ffpath = args.ffmpeg_path
        else:
            ffpath = shutil.which('ffmpeg')
            if not ffpath:
                # common conda env path used earlier by the user
                candidate = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'bin', 'ffmpeg.exe')
                if os.path.exists(candidate):
                    ffpath = candidate
                else:
                    # fallback common location used in the conversation
                    fallback = r"C:\ProgramData\anaconda3\envs\zed_env\Library\bin\ffmpeg.exe"
                    if os.path.exists(fallback):
                        ffpath = fallback

        if not ffpath or not os.path.exists(ffpath):
            print(f"ffmpeg not found (tried: {args.ffmpeg_path or 'PATH/conda candidates'}), skipping ffmpeg test")
        else:
            print(f"Running ffmpeg test -> {ff_out} (frames {start}..{end})")
            # Remove existing output
            try:
                if os.path.exists(ff_out):
                    os.remove(ff_out)
            except Exception:
                pass
            r = run_ffmpeg(ffpath, input_svo, n, ff_out, crop_filter=args.crop)
            results.append(r)
            print(f" ffmpeg rc={r['rc']} time={r['time_s']:.2f}s size={human_size(r['size_bytes'])}")

    # Try ZED fallback if pyzed available
    try:
        import convert_svo as convmod
        if not args.prefer_zed:
            print('Running ZED fallback test ->', zed_out)
        else:
            print('Running only ZED fallback test ->', zed_out)
        # remove existing
        try:
            if os.path.exists(zed_out):
                os.remove(zed_out)
        except Exception:
            pass
        rzed = run_zed(convmod, input_svo, start, end, zed_out, fps=30.0, crop='left', rotate=0, orient='auto', debug=args.debug)
        results.append(rzed)
        if rzed.get('rc', 1) == 0:
            print(f" zed rc=0 time={rzed['time_s']:.2f}s size={human_size(rzed['size_bytes'])}")
        else:
            print(f" zed failed (rc={rzed.get('rc')}) time={rzed['time_s']:.2f}s")
    except Exception as e:
        print('ZED fallback not available (pyzed missing or import error):', e)

    print('\nSummary:')
    for r in results:
        method = r.get('method')
        rc = r.get('rc')
        t = r.get('time_s')
        size = r.get('size_bytes')
        print(f" - {method:6} rc={rc} time={t:.2f}s size={human_size(size)}")

    print('\nDone.')


if __name__ == '__main__':
    main()

