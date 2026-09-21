"""Post-processing of the predicted trajectories (used in notebook Chapter 11): Savitzky-Golay smoothing."""
import torch
from scipy.signal import savgol_filter


def smooth_pred(y_pred, window_length=7, polyorder=2):
    """Savitzky-Golay filter on the predictions of the model.

    Args:
        y_pred (torch.Tensor): model output, shape [B, T, N, 2]
        window_length (int): filter window (odd)
        polyorder (int): order of the fitting polynomial

    Returns:
        torch.Tensor: smoothed trajectories, same shape and device as the input.
    """
    device = y_pred.device
    y_pred_np = y_pred.detach().cpu().numpy()

    # The time axis of [B, T, N, 2] is axis=1
    y_smooth_np = savgol_filter(y_pred_np,
                                window_length=window_length,
                                polyorder=polyorder,
                                axis=1,
                                mode='nearest')  # 'nearest' helps at the borders of the future

    return torch.from_numpy(y_smooth_np).to(device)


def smooth_trajectories_with_context(y_pred, x_past=None, window_length=11, polyorder=2):
    """Smooth the predicted future of the players together with their observed past, so the filter sees a continuous
    trajectory across the throw instant, and return only the future part.

    Args:
        y_pred: [T_out, N, 2] prediction of one play
        x_past: [N, T_in, 2] observed pre-pass positions of the same players
    Returns:
        [T_out, N_players, 2] smoothed prediction (N_players = number of players of x_past)
    """
    device = y_pred.device
    y_pred_np = y_pred.detach().cpu()

    x_past = x_past.permute(1, 0, 2)                                       # [T_in, N, 2]
    combined = torch.cat([x_past, y_pred_np[:, :x_past.shape[1]]])         # [T_in + T_out, N, 2]

    smoothed = savgol_filter(combined,
                             window_length=window_length,
                             polyorder=polyorder,
                             axis=0,
                             mode='nearest')

    smoothed_full = torch.from_numpy(smoothed).to(device)

    # Keep only the future part (from T_in on)
    refined_pred = smoothed_full[x_past.shape[0]:, :, :]

    # Optional hard anchoring: force the first predicted frame to equal the last observed one (zero gap):
    # refined_pred[0, :, :] = x_past[-1, :, :].clone()

    return refined_pred
