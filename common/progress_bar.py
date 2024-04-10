class ProgressBar:
    @staticmethod
    def print_progress_bar(
        iteration: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 1,
        length: int = 100,
        fill: str = "█",
        printEnd: str = "\r",
    ):
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print("\n")
