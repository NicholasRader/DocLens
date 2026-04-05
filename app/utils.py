def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"


def normalize_filename(filename: str) -> str:
    if not filename.lower().endswith(".pdf"):
        return filename + ".pdf"
    return filename