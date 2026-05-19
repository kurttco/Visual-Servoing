#!/usr/bin/env python3
"""
plot_results.py
---------------
Reads the most recent CSV log produced by safe_vs_controller.py and
generates the three result figures for the technical report:

  fig_errors.png    — Feature errors e_u and e_A over time
  fig_avoidance.png — FSM state and obstacle sqrt_area during avoidance
  fig_cmds.png      — Linear and angular velocity commands over time

Usage:
  python3 plot_results.py                    # uses most recent CSV in /tmp/puzzlebot_logs/
  python3 plot_results.py path/to/run.csv   # uses a specific file
  python3 plot_results.py --out ./figs/     # saves to a specific output folder
"""

import csv
import os
import sys
import glob
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────
STATE_COLORS = {
    'ACQUIRE':            '#ddeeff',
    'SERVO':              '#ddffdd',
    'HOLD':               '#f0f0f0',
    'LOST':               '#f0f0f0',
    'PRE_AVOID_STOP':     '#fff3cc',
    'APPROACH_OBSTACLE':  '#ffe0b0',
    'SQ_TURN_1':          '#f0e0ff',
    'SQ_LEG_1':           '#e8d5ff',
    'SQ_TURN_2':          '#f0e0ff',
    'SQ_LEG_2':           '#e8d5ff',
    'REACQUIRE':          '#ddeeff',
}

AVOIDANCE_STATES = {
    'PRE_AVOID_STOP', 'APPROACH_OBSTACLE',
    'SQ_TURN_1', 'SQ_LEG_1', 'SQ_TURN_2', 'SQ_LEG_2',
}

STATE_ORDER = [
    'ACQUIRE', 'SERVO', 'PRE_AVOID_STOP', 'APPROACH_OBSTACLE',
    'SQ_TURN_1', 'SQ_LEG_1', 'SQ_TURN_2', 'SQ_LEG_2',
    'REACQUIRE', 'HOLD', 'LOST',
]
STATE_NUM = {s: i for i, s in enumerate(STATE_ORDER)}


# ── CSV loading ───────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse(rows):
    """Convert string fields to numeric types."""
    data = {
        't':            [],
        'state':        [],
        'state_num':    [],
        'target_det':   [],
        'u_c':          [],
        'sqrt_area':    [],
        'e_u':          [],
        'e_A':          [],
        'obs_det':      [],
        'obs_sqrt_area':[],
        'obs_u_c':      [],
        'danger':       [],
        'v_cmd':        [],
        'w_cmd':        [],
    }
    for r in rows:
        data['t'].append(float(r['t_rel']))
        state = r['state']
        data['state'].append(state)
        data['state_num'].append(STATE_NUM.get(state, 0))
        data['target_det'].append(int(r['target_det']))
        data['u_c'].append(float(r['u_c']))
        data['sqrt_area'].append(float(r['sqrt_area']))
        data['e_u'].append(float(r['e_u']))
        data['e_A'].append(float(r['e_A']))
        data['obs_det'].append(int(r['obs_det']))
        data['obs_sqrt_area'].append(float(r['obs_sqrt_area']))
        data['obs_u_c'].append(float(r['obs_u_c']))
        data['danger'].append(int(r['danger']))
        data['v_cmd'].append(float(r['v_cmd']))
        data['w_cmd'].append(float(r['w_cmd']))
    for k in data:
        data[k] = np.array(data[k]) if k != 'state' else data[k]
    return data


def shade_states(ax, t, states, alpha=0.18, ymin=0, ymax=1):
    """
    Draw coloured background bands for each contiguous state block.
    """
    if len(t) == 0:
        return
    i = 0
    while i < len(states):
        s = states[i]
        j = i
        while j < len(states) and states[j] == s:
            j += 1
        color = STATE_COLORS.get(s, '#ffffff')
        ax.axvspan(t[i], t[j-1], ymin=ymin, ymax=ymax,
                   color=color, alpha=alpha, zorder=0)
        i = j


def find_latest_csv(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, 'run_*.csv')))
    if not files:
        raise FileNotFoundError(
            f'No run_*.csv files found in {log_dir}.\n'
            'Make sure the controller ran with log_csv: true.'
        )
    return files[-1]


# ── Figure 1: Feature errors ──────────────────────────────────────────────

def plot_errors(data, out_dir):
    t   = data['t']
    e_u = data['e_u']
    e_A = data['e_A']

    # Only show rows where target was detected for cleaner plot
    mask = data['target_det'] == 1
    if mask.sum() == 0:
        print('  Warning: no target detections in log — plot_errors may be empty.')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    shade_states(ax1, t, data['state'])
    shade_states(ax2, t, data['state'])

    # e_u
    ax1.plot(t, e_u, color='#1155cc', lw=1.5, label='$e_u$ (px)', zorder=3)
    ax1.axhline(0,   color='#888888', lw=0.9, ls='--', zorder=2)
    ax1.axhline( 25, color='#cc4444', lw=0.8, ls=':', alpha=0.7, zorder=2)
    ax1.axhline(-25, color='#cc4444', lw=0.8, ls=':', alpha=0.7, zorder=2)
    ax1.set_ylabel('Lateral error $e_u$ (px)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(
        min(-60, e_u.min() - 10),
        max( 60, e_u.max() + 10)
    )

    # small text annotation for deadband lines
    ax1.text(t[-1]*0.98, 27, '±deadband', ha='right',
             fontsize=7, color='#cc4444', alpha=0.8)

    # e_A
    ax2.plot(t, e_A, color='#cc7700', lw=1.5, label='$e_A$ (px)', zorder=3)
    ax2.axhline(0, color='#888888', lw=0.9, ls='--', zorder=2)
    ax2.axhline( 8, color='#cc4444', lw=0.8, ls=':', alpha=0.7, zorder=2)
    ax2.axhline(-8, color='#cc4444', lw=0.8, ls=':', alpha=0.7, zorder=2)
    ax2.set_ylabel('Size error $e_A$ (px)', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # state legend
    legend_handles = [
        mpatches.Patch(color=STATE_COLORS['SERVO'],   label='SERVO',   alpha=0.6),
        mpatches.Patch(color=STATE_COLORS['APPROACH_OBSTACLE'], label='Avoidance', alpha=0.6),
        mpatches.Patch(color=STATE_COLORS['REACQUIRE'], label='REACQUIRE', alpha=0.6),
    ]
    ax1.legend(handles=legend_handles + ax1.get_legend_handles_labels()[0],
               loc='upper right', fontsize=8, ncol=2)

    fig.suptitle('Feature Errors During Target Servoing', fontsize=12, y=0.98)

    out = os.path.join(out_dir, 'fig_errors.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Figure 2: Avoidance sequence ─────────────────────────────────────────

def plot_avoidance(data, out_dir, obs_trigger=150.0, obs_approach=350.0):
    t      = data['t']
    states = data['state']
    obs_A  = data['obs_sqrt_area']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.10)

    # ── top: obs sqrt_area ───────────────────────────────────────────
    shade_states(ax1, t, states)

    obs_visible = np.where(data['obs_det'] == 1, obs_A, np.nan)
    ax1.plot(t, obs_visible, color='#2255cc', lw=1.8,
             label='$\\sqrt{A_{\\rm obs}}$ (px)', zorder=3)
    ax1.axhline(obs_trigger,  color='#cc4400', lw=1.2, ls='--',
                label=f'Trigger threshold ({obs_trigger:.0f} px)', zorder=2)
    ax1.axhline(obs_approach, color='#338833', lw=1.2, ls='--',
                label=f'Approach target ({obs_approach:.0f} px)', zorder=2)
    ax1.set_ylabel('Obstacle apparent size $\\sqrt{A_{\\rm obs}}$ (px)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # mark where avoidance starts
    avoid_idx = next(
        (i for i, s in enumerate(states) if s == 'PRE_AVOID_STOP'), None)
    if avoid_idx is not None:
        ax1.axvline(t[avoid_idx], color='#cc4400', lw=1.5, alpha=0.6)
        ax1.text(t[avoid_idx] + 0.1, ax1.get_ylim()[1]*0.9,
                 'avoidance\ntriggered', color='#cc4400', fontsize=7.5)

    # ── bottom: FSM state ─────────────────────────────────────────────
    state_nums = data['state_num']
    ax2.step(t, state_nums, where='post', color='#444444', lw=1.5, zorder=3)
    shade_states(ax2, t, states, alpha=0.35)

    ax2.set_yticks(range(len(STATE_ORDER)))
    ax2.set_yticklabels(
        [s.replace('_', '\n') for s in STATE_ORDER],
        fontsize=6.5
    )
    ax2.set_ylabel('FSM State', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_ylim(-0.5, len(STATE_ORDER) - 0.5)

    fig.suptitle('Obstacle Avoidance Sequence', fontsize=12, y=0.99)

    out = os.path.join(out_dir, 'fig_avoidance.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Figure 3: Velocity commands ───────────────────────────────────────────

def plot_cmds(data, out_dir):
    t     = data['t']
    v_cmd = data['v_cmd']
    w_cmd = data['w_cmd']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    shade_states(ax1, t, data['state'])
    shade_states(ax2, t, data['state'])

    ax1.plot(t, v_cmd, color='#1155cc', lw=1.5, label='$v$ (m/s)', zorder=3)
    ax1.axhline(0,    color='#888888', lw=0.8, ls='--', zorder=2)
    ax1.axhline(0.18, color='#cc4444', lw=0.8, ls=':', alpha=0.6, zorder=2)
    ax1.text(t[-1]*0.98, 0.185, '$v_{\\rm max}$', ha='right',
             fontsize=7.5, color='#cc4444', alpha=0.8)
    ax1.set_ylabel('Linear velocity $v$ (m/s)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 0.25)

    ax2.plot(t, w_cmd, color='#cc5500', lw=1.5, label='$\\omega$ (rad/s)', zorder=3)
    ax2.axhline(0,     color='#888888', lw=0.8, ls='--', zorder=2)
    ax2.axhline( 0.45, color='#cc4444', lw=0.8, ls=':', alpha=0.6, zorder=2)
    ax2.axhline(-0.45, color='#cc4444', lw=0.8, ls=':', alpha=0.6, zorder=2)
    ax2.text(t[-1]*0.98, 0.46, '$\\omega_{\\rm max}$', ha='right',
             fontsize=7.5, color='#cc4444', alpha=0.8)
    ax2.set_ylabel('Angular velocity $\\omega$ (rad/s)', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.60, 0.60)

    # state legend (bottom axis)
    legend_handles = [
        mpatches.Patch(color=STATE_COLORS['SERVO'],              label='SERVO',            alpha=0.6),
        mpatches.Patch(color=STATE_COLORS['APPROACH_OBSTACLE'],  label='Approach',         alpha=0.6),
        mpatches.Patch(color=STATE_COLORS['SQ_TURN_1'],          label='Turn (90°)',       alpha=0.6),
        mpatches.Patch(color=STATE_COLORS['SQ_LEG_1'],           label='Straight leg',     alpha=0.6),
        mpatches.Patch(color=STATE_COLORS['REACQUIRE'],          label='Reacquire',        alpha=0.6),
    ]
    ax2.legend(handles=legend_handles, loc='lower right',
               fontsize=7.5, ncol=2, framealpha=0.9)

    fig.suptitle('Commanded Velocities Over Full Run', fontsize=12, y=0.98)

    out = os.path.join(out_dir, 'fig_cmds.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate report figures from CSV log.')
    parser.add_argument('csv', nargs='?', default=None,
                        help='Path to CSV log file. Default: most recent in /tmp/puzzlebot_logs/')
    parser.add_argument('--log-dir', default='/tmp/puzzlebot_logs',
                        help='Directory to search for CSV files (default: /tmp/puzzlebot_logs)')
    parser.add_argument('--out', default=None,
                        help='Output directory for PNG files (default: same as CSV)')
    parser.add_argument('--obs-trigger',  type=float, default=150.0,
                        help='obstacle_min_sqrt_area threshold (default: 150)')
    parser.add_argument('--obs-approach', type=float, default=350.0,
                        help='approach_obs_sqrt_area threshold (default: 350)')
    args = parser.parse_args()

    # find CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv(args.log_dir)
    print(f'Reading: {csv_path}')

    # output directory
    out_dir = args.out if args.out else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # load and plot
    rows = load_csv(csv_path)
    print(f'  Loaded {len(rows)} rows  ({rows[0]["t_rel"]}s – {rows[-1]["t_rel"]}s)')

    data = parse(rows)

    print('Generating figures...')
    plot_errors(data, out_dir)
    plot_avoidance(data, out_dir,
                   obs_trigger=args.obs_trigger,
                   obs_approach=args.obs_approach)
    plot_cmds(data, out_dir)

    print(f'\nDone. PNG files saved to: {out_dir}')
    print('Include in LaTeX with:')
    print('  \\includegraphics[width=\\columnwidth]{fig_errors.png}')
    print('  \\includegraphics[width=\\columnwidth]{fig_avoidance.png}')
    print('  \\includegraphics[width=\\columnwidth]{fig_cmds.png}')


if __name__ == '__main__':
    main()