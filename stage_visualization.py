import os
import matplotlib.pyplot as plt


def generate_visualizations(eval_data, jpg_folder):

    os.makedirs(jpg_folder, exist_ok=True)

    for (method, dataset), item in eval_data.items():

        pred = item["pred"]
        filename = item["file"]

        out_folder = os.path.join(jpg_folder, method)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(str(out_folder), filename.replace(".mat", ".jpg"))

        plt.figure(frameon=False)
        plt.imshow(pred, cmap="viridis")
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(out_path, dpi=300,
                    bbox_inches="tight",
                    pad_inches=0)

        plt.close()