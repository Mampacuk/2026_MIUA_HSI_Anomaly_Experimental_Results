import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def generate_roc_curves(eval_data, roc_folder):

    os.makedirs(roc_folder, exist_ok=True)

    datasets = set(dataset for _, dataset in eval_data)

    for dataset in datasets:

        plt.figure()

        for (method, ds), item in eval_data.items():

            if ds != dataset:
                continue

            pred = item["pred"]
            gt = item["gt"]

            fpr, tpr, _ = roc_curve(gt.flatten(), pred.flatten())
            auc_val = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{method} ({auc_val})")

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(dataset)

        plt.legend()

        plt.savefig(os.path.join(roc_folder, f"{dataset}.pdf"))
        plt.close()