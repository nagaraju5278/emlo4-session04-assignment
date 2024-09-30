from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar


def get_rich_progress_bar():
    return RichProgressBar()


def get_rich_model_summary():
    return RichModelSummary(max_depth=2)  # You can adjust max_depth as needed
