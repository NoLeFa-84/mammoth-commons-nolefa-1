def default_progress_callback(progress, message):
    progress = min(1, max(progress, 0))
    progress_blocks = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    full_blocks = int(progress * 20)
    partial_block_index = int((progress * 20 - full_blocks) * len(progress_blocks))
    bar = "█" * full_blocks
    if full_blocks < 20:
        bar += progress_blocks[partial_block_index]
    bar = bar.ljust(20)
    print(bar + message, end="\r")


__mammoth_progress_callback = default_progress_callback
__mammoth_progress_end = lambda: print("\r" + 140 * " " + "\r")


def register_progress_callback(callback, end=None):
    global __mammoth_progress_callback
    global __mammoth_progress_end

    def do_nothing():
        pass

    __mammoth_progress_callback = callback
    __mammoth_progress_end = do_nothing if end is None else end


def notify_progress(progress, message):
    global __mammoth_progress_callback
    __mammoth_progress_callback(progress, message)
    if progress >= 1:
        notify_end()
        return


def notify_end():
    global __mammoth_progress_end
    __mammoth_progress_end()
