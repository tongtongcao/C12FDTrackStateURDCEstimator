import numpy as np


def read_tracks_with_hits(filename):
    """
    NEW FORMAT (3 lines per event):

    Line 1: uRWell hits
      (xo, yo, xe, ye, z)

    Line 2: DC hits
      (doca, xm, xr, yr, z)

    Line 3: track state
      The corresponding track state at z = 223 (z of crosses in uRWell R1) in the tilted sector frame:
      (x, y, tx, ty, q/p)

    Returns
    -------
    ur_hits_list : list of np.ndarray [Nu, 5]
    dc_hits_list : list of np.ndarray [Nd, 5]
    states : np.ndarray [N, 5]
    """

    ur_hits_list = []
    dc_hits_list = []
    states = []

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    # MUST be multiple of 3 lines now
    if len(lines) % 3 != 0:
        raise ValueError("File must contain 3 lines per event (uRWell, DC, state).")

    # --------------------------
    # loop over events
    # --------------------------
    for i in range(0, len(lines), 3):

        # =========================
        # 1. uRWell hits
        # =========================
        ur_values = [float(x) for x in lines[i].split(",")]

        if len(ur_values) % 5 != 0:
            raise ValueError(
                f"Line {i+1}: uRWell hits must be multiple of 5"
            )

        ur_hits = np.array(ur_values, dtype=np.float32).reshape(-1, 5)
        ur_hits_list.append(ur_hits)

        # =========================
        # 2. DC hits
        # =========================
        dc_values = [float(x) for x in lines[i+1].split(",")]

        if len(dc_values) % 5 != 0:
            raise ValueError(
                f"Line {i+2}: DC hits must be multiple of 5"
            )

        dc_hits = np.array(dc_values, dtype=np.float32).reshape(-1, 5)
        dc_hits_list.append(dc_hits)

        # =========================
        # 3. track state
        # =========================
        state_values = [float(x) for x in lines[i+2].split(",")]

        if len(state_values) != 5:
            raise ValueError(
                f"Line {i+3}: state must have 5 values"
            )

        states.append(state_values)

    states = np.array(states, dtype=np.float32)

    return ur_hits_list, dc_hits_list, states


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":

    ur_hits, dc_hits, states = read_tracks_with_hits("sample.csv")

    print(f"Tracks: {len(states)}")

    print("First event:")
    print("  uRWell hits shape:", ur_hits[0].shape)
    print("  DC hits shape:", dc_hits[0].shape)
    print("  state:", states[0])