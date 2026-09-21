"""Play visualizations (notebook Chapters 4 and 11): field drawing, frame-by-frame play animation strips and the comparison
between ground truth and model prediction.

Field convention: the long axis of the field (x, 0-120 yards) is the vertical axis of the plots, the width (y, 0-53.3)
the horizontal one.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

from .data import get_play
from .smoothing import smooth_trajectories_with_context

OFFENSE_ROLES = ["Passer", "Other Route Runner"]


# ---------------------------------------------------------------- Field
def draw_football_field(ax=None, scrimmage=None, field_color='white', line_color='lightgrey', lw=2):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6.33))

    # Field background
    ax.set_facecolor(field_color)

    # Line of Scrimmage
    ax.plot([0,53.3], [scrimmage, scrimmage], color= "#00000066", linewidth= lw, zorder= 1, clip_on= True)

    # End zones
    ax.add_patch(plt.Rectangle((0,0), 53.3, 10, facecolor='darkblue', alpha=0.4, zorder= 0, clip_on= True))
    ax.add_patch(plt.Rectangle((0,110), 53.3, 10, facecolor='darkred', alpha=0.4, zorder= 0, clip_on= True))

    # Yard lines every 5 yards
    for x in range(0, 121, 5):
        lw_line = lw if x % 10 == 0 else lw/2
        if x != 5 and x != 115:
            ax.plot([0, 53.3], [x, x], color=line_color, linewidth=lw_line, zorder= 0, clip_on= True)

        # Yard line numbers (every 10 yards, mirrored correctly)
        if 10 < x < 110 and x % 10 == 0:
            display_num = x - 10 if x <= 60 else 110 - x  # Mirror numbers after 50
            ax.text(3, x, str(display_num), color= line_color, fontsize= 20, ha='center', va='center', rotation= 270, zorder= 0, clip_on= True)
            ax.text(53.3 - 3, x, str(display_num), color= line_color, fontsize= 20, ha='center', va='center', rotation= 90, zorder= 0, clip_on= True)

    # Sidelines and end zone borders
    ax.plot([0,0], [0, 120], color=line_color, linewidth=lw, zorder= 0, clip_on= True)
    ax.plot([53.3,53.3], [0, 120],color=line_color, linewidth=lw, zorder= 0, clip_on= True)
    ax.plot([0,53.3], [0,0], color=line_color, linewidth=lw, zorder= 0, clip_on= True)
    ax.plot([0,53.3], [120,120], color=line_color, linewidth=lw, zorder= 0, clip_on= True)

    # Hash marks
    for x in range(10, 111):
        for y in [0.4, 53.3 - 0.4]:
            ax.plot([y, y+0.2], [x, x], color=line_color, linewidth=1, zorder= 0, clip_on= True)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


# ---------------------------------------------------------------- Ground truth vs prediction
def gnn_batch_to_viz_inputs(gnn_batch):
    """Rebuild the inputs of `visualize_comparison` from a PyG Batch with B>=1 plays: gnn_batch.x concatenates the
    nodes of ALL the plays of the batch (no padding), so it is split per graph with batch.ptr before being padded to
    [B, T_max*P_max, F] for plotting.

    Everything is moved to CPU: with batch lists preloaded on the GPU these tensors would reach matplotlib/numpy
    (scrimmage, ball, ...) as CUDA tensors, which they do not accept."""
    B = gnn_batch.num_graphs
    x_cpu = gnn_batch.x.detach().cpu()
    num_frames_per_graph = gnn_batch.num_frames.detach().cpu()

    # Players per graph = nodes of graph i / frames of graph i: the number of valid players can change between plays.
    num_players_per_graph = [
        int((gnn_batch.ptr[i + 1] - gnn_batch.ptr[i]) // int(num_frames_per_graph[i]))
        for i in range(B)
    ]

    max_frames = int(num_frames_per_graph.max())
    max_players = max(num_players_per_graph)

    x_nodes = torch.zeros((B, max_frames * max_players, x_cpu.shape[1]))
    input_mask = torch.zeros((B, max_frames, max_players), dtype=torch.bool)

    for i in range(B):
        start, end = int(gnn_batch.ptr[i]), int(gnn_batch.ptr[i + 1])
        n_frames_i = int(num_frames_per_graph[i])
        n_players_i = num_players_per_graph[i]
        x_nodes[i, :end - start] = x_cpu[start:end]
        input_mask[i, :n_frames_i, :n_players_i] = True

    y = gnn_batch.y_target.detach().cpu()          # [B, T_out, N, 2]
    output_mask = gnn_batch.y_mask.detach().cpu()  # [B, T_out, N]

    return x_nodes, input_mask, y, output_mask, gnn_batch.x_global.detach().cpu()


def visualize_comparison(input_batch, input_mask, target_batch, pred_batch, output_mask, scrimmage=None, ball=None,
                         batch_idx=0, smooth=True, verbose=True, title=None):
    """Ground truth vs prediction of one play.

    Args:
        input_batch:  [B, T_in * N, F]  - past (pre-pass) node features, frame-major
        input_mask:   [B, T_in, N]      - validity mask
        target_batch: [B, T_out, N, 2]  - ground truth
        pred_batch:   [B, T_out, N, 2]  - model prediction
        output_mask:  [B, T_out, N]     - validity mask
        scrimmage, ball: line of scrimmage (yards) and (x, y) landing point of the ball, in RAW yards
        smooth: apply the Savitzky-Golay post-processing to the prediction before drawing/scoring it
        verbose: print the MSE of every player before ("PRE") and after ("POST") the smoothing
    Returns the MSE per player as a list of (player index, mse_before, mse_after).
    """
    min_y, max_y = 130, -10
    fig, ax = plt.subplots()

    defense_color = 'red'
    offense_color = 'blue'
    defense_pred_color = 'darkred'
    offense_pred_color = 'darkblue'
    alpha_prepass = 0.2
    alpha_postpass = 0.7

    def get_player_color_by_side(side, pred=False):
        if pred:
            return defense_pred_color if side[0] else offense_pred_color
        return defense_color if side[0] else offense_color

    # PRE-PASS REPRESENTATION
    prepass = input_batch[batch_idx, :, :2]
    side = input_batch[batch_idx, :, 22:24]
    mask = input_mask[batch_idx]

    num_players = int(mask[0].sum())
    frames_per_player = mask.sum(dim = 0)
    num_frames = int(frames_per_player.max())

    num_nodes = num_players * num_frames
    filtered_prepass = prepass[:num_nodes]

    complete_prepass = torch.zeros((num_players, num_frames, 2))
    for n in range(num_nodes):
        complete_prepass[n % num_players, n // num_players] = filtered_prepass[n]

    min_y = min(min_y, float(complete_prepass[:, :, 0].min()))
    max_y = max(max_y, float(complete_prepass[:, :, 0].max()))

    for n in range(complete_prepass.shape[0]):
        ax.plot(complete_prepass[n, :, 1].cpu().numpy(), complete_prepass[n, :, 0].cpu().numpy(),
                color= get_player_color_by_side(side[n % num_players]), alpha= alpha_prepass, linewidth=2)

    # POST-PASS REPRESENTATION
    y_true = target_batch[batch_idx]
    y_pred = pred_batch[batch_idx]
    mask = output_mask[batch_idx]
    frames_per_player = mask.sum(axis=0)

    ax.set_xlim(-5, 55)
    scores = []

    for n in range(y_true.shape[1]):
        valid_frames = int(frames_per_player[n])
        if valid_frames == 0:
            continue

        # Ground truth: what really happened
        ax.plot(y_true[:valid_frames, n, 1].cpu().numpy(), y_true[:valid_frames, n, 0].cpu().numpy(),
                color= get_player_color_by_side(side[n]), alpha= alpha_postpass, linewidth=2)

        # Prediction of the model
        mse_pre = F.mse_loss(y_true[:valid_frames, n], y_pred[:valid_frames, n]).item()

        # Separate variable: reassigning y_pred inside the loop would smooth player n with n+1 passes of the filter
        if smooth:
            y_shown = smooth_trajectories_with_context(y_pred[:valid_frames], complete_prepass)
        else:
            y_shown = y_pred[:valid_frames]
        mse_post = F.mse_loss(y_true[:valid_frames, n], y_shown[:valid_frames, n]).item()
        scores.append((n, mse_pre, mse_post))
        if verbose:
            print(f"Player {n}: MSE before smoothing {mse_pre:.3f} | after {mse_post:.3f}")

        ax.plot(y_shown[:valid_frames, n, 1].cpu().numpy(), y_shown[:valid_frames, n, 0].cpu().numpy(),
                color=get_player_color_by_side(side[n], True), alpha= alpha_postpass, linewidth=2)

        # End markers: the distance between them is the final displacement error (FDE)
        ax.scatter(y_true[valid_frames-1, n, 1].cpu().numpy(), y_true[valid_frames-1, n, 0].cpu().numpy(), color= get_player_color_by_side(side[n]), s=30, alpha=0.5)
        ax.scatter(y_shown[valid_frames-1, n, 1].cpu().numpy(), y_shown[valid_frames-1, n, 0].cpu().numpy(), color= get_player_color_by_side(side[n], True), s=30, alpha=0.5)

        for arr in (y_true[:valid_frames, n, 0], y_shown[:valid_frames, n, 0]):
            min_y = min(min_y, float(arr.min()))
            max_y = max(max_y, float(arr.max()))

    ax.set_ylim(min_y - 10, max_y + 10)
    ax.set_title(title or f"Ground truth vs prediction - play {batch_idx}")
    draw_football_field(ax, scrimmage)
    if ball is not None:
        ax.scatter(ball[1], ball[0], color='orange')

    legend_elements = [
        Line2D([0], [0], color=offense_color, lw=2, alpha=alpha_postpass, label='Offense - ground truth'),
        Line2D([0], [0], color=offense_pred_color, lw=2, alpha=alpha_postpass, label='Offense - prediction'),
        Line2D([0], [0], color=defense_color, lw=2, alpha=alpha_postpass, label='Defense - ground truth'),
        Line2D([0], [0], color=defense_pred_color, lw=2, alpha=alpha_postpass, label='Defense - prediction'),
        Line2D([0], [0], color=offense_color, lw=2, alpha=alpha_prepass, label='Pre-pass (observed)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Ball landing'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close(fig)  # avoid piling up open figures in the evaluation loop
    return scores


def show_predictions(model, gnn_batch_list, device, global_minmax, n_plays=3, first=0, smooth=True, verbose=True):
    """Draw `n_plays` plays of a list of batches: ground truth vs prediction of `model`.

    The global features are min-max normalized, so scrimmage and ball landing (indices 2 and 5:7 of the "all" vector)
    are mapped back to yards with `global_minmax` (`GLOBAL_FEATURE_MINMAX`) before being drawn on the field.
    """
    def to_yards(value, name):
        lo, hi = global_minmax[name]
        return value * (hi - lo) + lo

    model.eval()
    shown = 0
    seen = 0
    for gnn_batch in gnn_batch_list:
        if shown >= n_plays:
            break
        with torch.no_grad():
            preds = model(gnn_batch.to(device)).detach().cpu()
        x_nodes_v, input_mask_v, y_v, out_mask_v, x_global_v = gnn_batch_to_viz_inputs(gnn_batch)

        for i in range(y_v.shape[0]):
            if seen < first:
                seen += 1
                continue
            if shown >= n_plays:
                break
            scrimmage = to_yards(x_global_v[i, 2], "scrimmage")
            ball = (to_yards(x_global_v[i, 5], "ball_at_landing_x"), to_yards(x_global_v[i, 6], "ball_at_landing_y"))
            visualize_comparison(x_nodes_v, input_mask_v, y_v, preds, out_mask_v, scrimmage, ball,
                                 batch_idx=i, smooth=smooth, verbose=verbose,
                                 title=f"Ground truth vs prediction - test play {seen}")
            shown += 1
            seen += 1


# ---------------------------------------------------------------- Frame-by-frame play strips
def visualize_play(data, game_id, play_id, arrows="motion", frame_step=1):
    """Frame-by-frame view of a play: pre-pass frames (solid trails) followed by post-pass frames (dashed trails).

    arrows="motion":      arrow along the movement direction `dir`, length proportional to speed + 0.5 * acceleration
    arrows="orientation": arrow along the body orientation `o`, fixed length
    frame_step: draw one frame every `frame_step` (the last frame of each phase is always drawn)
    """
    example_play, output_example_play = get_play(data, game_id, play_id)
    if example_play is None or output_example_play is None:
        return
    angle_col = "dir" if arrows == "motion" else "o"

    # Ball landing position
    ball_x = example_play['ball_land_x'].iloc[0]
    ball_y = example_play['ball_land_y'].iloc[0]

    # --- Dynamic field bounds ---
    scrimmage = example_play["absolute_yardline_number"].iloc[0]
    all_x = list(example_play["x"]) + list(output_example_play["x"])
    xmin, xmax = min(all_x) - 8, max(all_x) + 8
    ymin, ymax = 0 - 5 , 53.3 + 5

    # --- Pivot for plotting ---
    players_positions_x = example_play.pivot(index="nfl_id", columns="frame_id", values="x")
    players_positions_y = example_play.pivot(index="nfl_id", columns="frame_id", values="y")
    players_speed = example_play.pivot(index="nfl_id", columns="frame_id", values="s")
    players_accel = example_play.pivot(index="nfl_id", columns="frame_id", values="a")
    players_angle = example_play.pivot(index="nfl_id", columns="frame_id", values=angle_col)

    input_frames = players_positions_x.columns

    output_players_positions_x = output_example_play.pivot(index="nfl_id", columns="frame_id", values="x")
    output_players_positions_y = output_example_play.pivot(index="nfl_id", columns="frame_id", values="y")
    output_frames = output_players_positions_x.columns

    def select(n):
        idx = list(range(0, n, frame_step))
        return idx if idx and idx[-1] == n - 1 else idx + [n - 1]

    in_sel, out_sel = select(len(input_frames)), select(len(output_frames))

    # Player roles
    player_roles = example_play.drop_duplicates("nfl_id").set_index("nfl_id")["player_role"]

    # Colors
    offense_color = 'blue'
    defense_color = 'red'
    target_color = 'darkblue'
    ball_color = 'orange'
    trail_alpha = 0.4

    def get_player_color(player):
        role = player_roles.get(player, "Unknown")
        if role == "Targeted Receiver":
            return target_color
        elif role == "Passer":
            return 'skyblue'
        elif role in OFFENSE_ROLES:
            return offense_color
        else:
            return defense_color

    # --- Global arrow scale factor ---
    arrow_lengths = (players_speed + 0.5 * players_accel).values.flatten()
    max_arrow_length = np.nanmax(arrow_lengths)
    arrow_scale = 3 / max_arrow_length if max_arrow_length > 0 else 1  # scale to ~3 units max

    total_frames = len(in_sel) + len(out_sel)
    cols = 3
    rows = math.ceil(total_frames / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    axes = np.atleast_1d(axes).flatten()

    def plot_arrow(ax, x, y, length, dir_val, color):
        if np.isnan(length) or np.isnan(dir_val):
            return
        theta = np.deg2rad(dir_val)
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)
        ax.arrow(x, y, dx, dy, color=color, width=0.15, head_length=0.7, head_width=0.5,
                 length_includes_head=True, alpha=0.8)

    # --- Input frames ---
    for pos, i in enumerate(in_sel):
        frame = input_frames[i]
        ax = axes[pos]
        draw_football_field(ax, scrimmage)
        for player in players_positions_x.index:
            x = players_positions_x.loc[player, input_frames[:i+1]].values
            y = players_positions_y.loc[player, input_frames[:i+1]].values
            color = get_player_color(player)
            ax.plot(y, x, color=color, alpha=trail_alpha, linewidth=2)
            ax.scatter(y[-1], x[-1], color=color)

            # Arrow
            if arrows == "motion":
                s_val = players_speed.loc[player, frame]
                a_val = players_accel.loc[player, frame]
                length = (s_val + 0.5*a_val) * arrow_scale if not np.isnan(s_val + 0.5*a_val) else 0
            else:
                length = 2
            plot_arrow(ax, y[-1], x[-1], length, players_angle.loc[player, frame], color)

        ax.set_ylim(xmin, xmax)
        ax.set_xlim(ymin, ymax)
        ax.set_title(f"Frame {frame}")
        ax.grid(False)
        ax.invert_xaxis()

    # --- Output frames ---
    for pos, i in enumerate(out_sel):
        frame = output_frames[i]
        ax = axes[pos + len(in_sel)]
        draw_football_field(ax, scrimmage)
        for player in set(players_positions_x.index) | set(output_players_positions_x.index):
            # Input trail
            if player in players_positions_x.index:
                ax.plot(players_positions_y.loc[player].values, players_positions_x.loc[player].values,
                        color=get_player_color(player), alpha=trail_alpha, linewidth=2)

            # Output trail
            if player in output_players_positions_x.index:
                x_output = output_players_positions_x.loc[player, output_frames[:i+1]].values
                y_output = output_players_positions_y.loc[player, output_frames[:i+1]].values
                color = get_player_color(player)
                ax.plot(y_output, x_output, color=color, linestyle='--', alpha=trail_alpha*2, linewidth=2)
                ax.scatter(y_output[-1], x_output[-1], color=color)

        if i == len(output_frames) - 1:
            ax.scatter(ball_y, ball_x, color=ball_color, s=100)

        ax.set_ylim(xmin, xmax)
        ax.set_xlim(ymin, ymax)
        ax.set_title(f"Frame {frame}")
        ax.grid(False)
        ax.invert_xaxis()

    # Hide unused axes
    for j in range(total_frames, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
