"""
select_reference_gui.py

Simple OpenCV GUI to step through a list of videos and pick a reference frame for each.
Saves results as CSV with columns: video_name, selected_frame_index, timestamp_ms (optional if provided list)
"""
import os
import csv
import cv2
import argparse


def select_reference_for_videos(dossier_videos, output_csv, initial_video=None, start_frame=None, end_frame=None):
    video_files = sorted([os.path.join(dossier_videos, f) for f in os.listdir(dossier_videos) if f.lower().endswith('.mp4')])
    if not video_files:
        print(f"Aucune MP4 dans {dossier_videos}")
        return False

    # If initial_video provided, rotate list so it starts with that
    if initial_video:
        full_initial = os.path.join(dossier_videos, initial_video) if not os.path.isabs(initial_video) else initial_video
        if full_initial in video_files:
            idx = video_files.index(full_initial)
            video_files = video_files[idx:] + video_files[:idx]

    results = []
    last_selected = None

    for vid in video_files:
        cap = cv2.VideoCapture(vid)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Apply optional start/end frame limits for testing a segment
        if start_frame is not None:
            start_idx = int(max(0, start_frame))
        else:
            start_idx = 0
        if end_frame is not None:
            end_idx = int(min(total_frames - 1, end_frame))
        else:
            end_idx = total_frames - 1
        total = end_idx - start_idx + 1 if end_idx >= start_idx else 0
        print(f"Ouverture {vid} ({total} frames)")
        if last_selected is not None:
            idx = min(max(last_selected, start_idx), end_idx)
        else:
            idx = start_idx
        selected = None
        window_name = f"Select reference - {os.path.basename(vid)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print("Lecture terminée")
                break
            display = frame.copy()
            # Display frame number relative to original video
            cv2.putText(display, f"Frame: {idx}/{end_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if selected is not None:
                cv2.putText(display, f"Selected: {selected}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyWindow(window_name)
                print("Opération annulée par l'utilisateur.")
                return False
            elif key == ord('d') or key == 83:  # right arrow
                idx = min(idx + 1, end_idx)
            elif key == ord('a') or key == 81:  # left arrow
                idx = max(idx - 1, start_idx)
            elif key == ord('w') or key == ord(' '):  # up or space to mark
                selected = idx
            elif key == ord('s') or key == ord('\r') or key == 13:  # enter to confirm and move on
                if selected is None:
                    selected = idx
                # Save result for this video
                results.append((os.path.basename(vid), selected))
                last_selected = selected
                break
            elif key == ord('z'):  # jump -10
                idx = max(idx - 10, 0)
            elif key == ord('x'):  # jump +10
                idx = min(idx + 10, total - 1)

        cap.release()
        cv2.destroyWindow(window_name)

    # Écrire CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'selected_frame'])
        for row in results:
            writer.writerow(row)

    print(f"Sélection sauvegardée dans {output_csv}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dossier', help='Dossier contenant les mp4')
    parser.add_argument('--out', help='Fichier CSV de sortie', default='reference_frames.csv')
    parser.add_argument('--start', help='Vidéo à ouvrir en premier (filename.mp4)', default=None)
    parser.add_argument('--start-frame', type=int, help='Index de frame de départ (optionnel) pour tester une portion', default=None)
    parser.add_argument('--end-frame', type=int, help='Index de frame de fin (optionnel) pour tester une portion', default=None)
    args = parser.parse_args()
    select_reference_for_videos(args.dossier, args.out, args.start, start_frame=args.start_frame, end_frame=args.end_frame)

