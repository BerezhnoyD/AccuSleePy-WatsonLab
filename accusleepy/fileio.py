def save_labels(
    labels: np.ndarray,
    filename: str,
    spectr_data: np.ndarray | None = None,
    spectr_freq: np.ndarray | None = None,
    confidence_scores: np.ndarray | None = None,
) -> None:
    """Save brain state labels to file and optionally save spectrogram arrays.

    :param labels: brain state labels
    :param filename: filename for CSV output (e.g. 'some_labels.csv')
    :param spectr_data: optional spectrogram array to save as <base>_spectr.npy
    :param spectr_freq: optional frequency array to save as <base>_spectr_freq.npy
    :param confidence_scores: optional confidence scores
    """
    if confidence_scores is not None:
        pd.DataFrame(
            {c.BRAIN_STATE_COL: labels, c.CONFIDENCE_SCORE_COL: confidence_scores}
        ).to_csv(filename, index=False)
    else:
        pd.DataFrame({c.BRAIN_STATE_COL: labels}).to_csv(filename, index=False)

    base, _ = os.path.splitext(filename)
    if spectr_data is not None:
        np.save(f"{base}_spectr.npy", spectr_data)
    if spectr_freq is not None:
        np.save(f"{base}_spectr_freq.npy", spectr_freq)
