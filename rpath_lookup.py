#!/usr/bin/env python3
"""
Small utility to show which R matrix file `baseline_runs.py` would pick
for a given channel and speed using the same search logic.
"""
import os, glob, argparse

def find_rpath(channel, speed):
    repo_dir = os.path.dirname(__file__)
    search_dirs = [os.path.join(repo_dir, "R_mats_timedomain"), repo_dir, os.getcwd()]
    ch_up = channel.upper()
    is_tdl = ch_up.startswith("TDL") or ch_up.startswith("T")
    is_cdl = ch_up.startswith("CDL") or ch_up.startswith("C")
    preferred_letter = None
    if "-" in ch_up:
        parts = ch_up.split("-")
        if len(parts) > 1 and len(parts[1]) == 1:
            preferred_letter = parts[1]
    candidates = []
    speed_tag = f"{float(speed):.1f}"
    pat_speed = f"*Speed{speed_tag}*.npz"
    for d in search_dirs:
        if not d:
            continue
        patterns = []
        if is_tdl:
            patterns += [os.path.join(d, f"TDL_R_*Speed{speed_tag}*.npz"),
                         os.path.join(d, f"TDL_R_CIR_*Speed{speed_tag}*.npz"),
                         os.path.join(d, pat_speed)]
        elif is_cdl:
            patterns += [os.path.join(d, f"CDL_R_*Speed{speed_tag}*.npz"),
                         os.path.join(d, f"CDL_R_CIR_*Speed{speed_tag}*.npz"),
                         os.path.join(d, pat_speed)]
        else:
            patterns += [os.path.join(d, pat_speed)]
        for p in patterns:
            for m in glob.glob(p):
                if m not in candidates:
                    candidates.append(m)
    chosen = None
    if candidates:
        if preferred_letter:
            for c in candidates:
                name = os.path.basename(c).upper()
                if f"_{preferred_letter}" in name or f"CIR_{preferred_letter}" in name or name.endswith(f"{preferred_letter}.NPZ"):
                    chosen = c
                    break
        if not chosen:
            for d in search_dirs:
                for c in candidates:
                    if os.path.dirname(c) == d:
                        chosen = c
                        break
                if chosen:
                    break
        if not chosen:
            chosen = candidates[0]
    return chosen

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--channel', default=os.environ.get('CHANNEL','TDL-D'))
    p.add_argument('--speed', default=os.environ.get('SPEED','10'))
    args = p.parse_args()
    r = find_rpath(args.channel, args.speed)
    if r:
        print(r)
    else:
        print('No R matrix found for', args.channel, args.speed)
