import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def phm08_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true

    # TODO: apply the asymmetric formula above, per-element
    # hint: build a boolean mask for d < 0 vs d >= 0, apply the two
    # different exp formulas to each subset, then sum everything
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    return score

def evaluate(y_true, y_pred) -> dict:
    # TODO: return both metrics together, plus maybe n_units, for a results table
    rmse_value = rmse(y_true, y_pred)
    phm08_value = phm08_score(y_true, y_pred)
    return {
        "RMSE": rmse_value,
        "PHM08_score": phm08_value,
    }


if __name__ == "__main__":
    # Sanity check against the worked example from earlier in our discussion:
    # d=-10 -> ~1.16, d=+10 -> ~1.72, d=-30 -> ~9.05, d=+30 -> ~19.09
    for d in [-30, -10, 10, 30]:
        y_true = [100]
        y_pred = [100 + d]
        print(f"d={d:+d} -> phm08 contribution={phm08_score(y_true, y_pred):.2f}")