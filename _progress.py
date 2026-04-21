"""_progress.py

Infrastructure de suivi de progression partagée entre workers
`ProcessPoolExecutor` et une barre tqdm par caméra côté main.

Protocole (messages posés dans la queue par les workers) :
  ("total", cam_name, N)  : nombre total de frames attendues
  ("inc",   cam_name, k)  : +k frames traitées depuis le dernier envoi
  ("done",  cam_name, _)  : worker terminé

Le main lance un thread qui drain la queue et met à jour les barres.
"""
from queue import Empty
from tqdm.auto import tqdm


_PROGRESS_Q = None
REPORT_EVERY = 20  # frames entre deux reports pour limiter l'IPC


def init_worker(q):
    """Initializer du ProcessPoolExecutor : stocke la queue dans le worker."""
    global _PROGRESS_Q
    _PROGRESS_Q = q


def send(kind, cam_name, value=None):
    q = _PROGRESS_Q
    if q is None:
        return
    try:
        q.put_nowait((kind, cam_name, value))
    except Exception:
        pass


def drain(progress_q, n_workers_target, stop_event, step_label):
    """Consomme la queue et affiche une barre tqdm par caméra."""
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
