def classify_demand(
    prediction: float,
    low_threshold: float = 33.0,
    high_threshold: float = 67.0,
) -> str:
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be lower than high_threshold")
    if prediction < low_threshold:
        return "low"
    if prediction < high_threshold:
        return "medium"
    return "high"

