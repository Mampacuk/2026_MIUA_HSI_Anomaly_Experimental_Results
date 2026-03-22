def inspect_score_distribution(pred, method, dataset):

    import matplotlib.pyplot as plt
    import numpy as np

    values = pred.flatten()

    plt.figure(figsize=(8,5))

    # histogram
    plt.hist(values, bins=1000, log=True)

    # percentile markers
    percentiles = [95, 97, 98, 99, 99.5, 99.9]

    for p in percentiles:

        v = np.percentile(values, p)

        plt.axvline(
            v,
            linestyle="--",
            linewidth=1,
            label=f"{p}%"
        )

    plt.title(f"{method} {dataset} score distribution")
    plt.xlabel("abnormality score")
    plt.ylabel("pixel count (log scale)")

    plt.legend()

    plt.tight_layout()
    plt.show()

    # text output

    print(f"\n{method} {dataset} score percentiles:\n")

    for p in percentiles:
        print(f"{p}% : {np.percentile(values, p)}")

    print(f"\nmin : {values.min()}")
    print(f"max : {values.max()}")