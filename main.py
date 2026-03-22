from load_datasets import discover_detector_files, preload_evaluation_data
from stage_metrics import compute_metrics
from stage_roc import generate_roc_curves
from stage_visualization import generate_visualizations


def main():
    methods = [
        # "RX",
        # "LRX",
        # "KRX_RFF",
        # "LRASR",
        # "LSMAD",
        # "CRD",
        # "FEBPAD",
        # "KIFD_best",
        "PTA_best",
    ]

    datasets = [
        "porcine1",
        "porcine2",
        "porcine3",
        "porcine4",
        "bovine1",
        "bovine2",
        "bovine3",
        "bovine4",
        "bovine5.1",
        "bovine5.2",
        "bovine5.3",
    ]

    groups = [
        "porcine",
        "bovine"
    ]

    det_folder = "detector_outputs"
    gt_folder = "ground_truth"

    jpg_folder = "visualizations"
    metrics_folder = "metrics"
    roc_folder = "roc_curves"

    clamp_config = {
        # GRX
        ("RX", "porcine1"): 0.995,
        ("RX", "porcine2"): 0.9975,
        ("RX", "porcine3"): 0.98,
        ("RX", "porcine4"): 0.995,
        ("RX", "bovine1"): 0.995,
        ("RX", "bovine2"): 0.995,
        ("RX", "bovine3"): 0.99,
        ("RX", "bovine4"): 0.995,
        ("RX", "bovine5.1"): 0.99,
        ("RX", "bovine5.2"): 0.99,
        ("RX", "bovine5.3"): 0.995,
        # LRX
        ("LRX", "porcine1"): 0.995,
        ("LRX", "porcine2"): 0.995,
        ("LRX", "porcine3"): 0.98,
        ("LRX", "porcine4"): 0.999,
        ("LRX", "bovine1"): 0.99,
        ("LRX", "bovine2"): 0.995,
        ("LRX", "bovine3"): 0.995,
        ("LRX", "bovine4"): 0.995,
        ("LRX", "bovine5.1"): 0.995,
        ("LRX", "bovine5.2"): 0.999,
        ("LRX", "bovine5.3"): 0.995,
        # KRX
        ("KRX_RFF", "porcine1"): 0.99,
        ("KRX_RFF", "porcine2"): 0.99,
        ("KRX_RFF", "porcine3"): 0.98,
        ("KRX_RFF", "porcine4"): 0.99,
        ("KRX_RFF", "bovine1"): 0.995,
        ("KRX_RFF", "bovine2"): 0.99,
        ("KRX_RFF", "bovine3"): 0.995,
        ("KRX_RFF", "bovine4"): 0.995,
        ("KRX_RFF", "bovine5.1"): 0.995,
        ("KRX_RFF", "bovine5.2"): 0.995,
        ("KRX_RFF", "bovine5.3"): 0.995,
        # LRASR
        ("LRASR", "porcine1"): 0.995,
        ("LRASR", "porcine2"): 0.995,
        ("LRASR", "porcine3"): 0.99,
        ("LRASR", "porcine4"): 0.99,
        ("LRASR", "bovine1"): 0.99,
        ("LRASR", "bovine2"): 0.99,
        ("LRASR", "bovine3"): 0.99,
        ("LRASR", "bovine4"): 0.99,
        ("LRASR", "bovine5.1"): 0.99,
        ("LRASR", "bovine5.2"): 0.98,
        ("LRASR", "bovine5.3"): 0.98,
        # LSMAD
        ("LSMAD", "bovine1"): 0.995,
        ("LSMAD", "bovine2"): 0.995,
        ("LSMAD", "bovine3"): 0.99,
        ("LSMAD", "bovine4"): 0.995,
        ("LSMAD", "bovine5.1"): 0.999,
        ("LSMAD", "bovine5.2"): 0.999,
        ("LSMAD", "bovine5.3"): 0.995,
        ("LSMAD", "porcine1"): 0.995,
        ("LSMAD", "porcine2"): 0.999,
        ("LSMAD", "porcine3"): 0.995,
        ("LSMAD", "porcine4"): 0.999,
        # CRD
        ("CRD", "bovine1"): 0.995,
        ("CRD", "bovine2"): 0.995,
        ("CRD", "bovine3"): 0.99,
        ("CRD", "bovine4"): 0.99,
        ("CRD", "bovine5.1"): 0.995,
        ("CRD", "bovine5.2"): 0.99,
        ("CRD", "bovine5.3"): 0.99,
        ("CRD", "porcine1"): 0.995,
        ("CRD", "porcine2"): 0.995,
        ("CRD", "porcine3"): 0.985,
        ("CRD", "porcine4"): 0.995,
        # FEBPAD
        ("FEBPAD", "bovine1"): 0.999,
        ("FEBPAD", "bovine2"): 0.99,
        ("FEBPAD", "bovine3"): 0.995,
        ("FEBPAD", "bovine4"): 0.99,
        ("FEBPAD", "bovine5.1"): 0.99,
        ("FEBPAD", "bovine5.2"): 0.99,
        ("FEBPAD", "bovine5.3"): 0.99,
        ("FEBPAD", "porcine1"): 0.99,
        ("FEBPAD", "porcine2"): 0.99,
        ("FEBPAD", "porcine3"): 0.99,
        ("FEBPAD", "porcine4"): 0.98,
        # KIFD_best
        ("KIFD_best", "bovine1"): 0.995,
        ("KIFD_best", "bovine2"): 0.99,
        ("KIFD_best", "bovine3"): 0.995,
        ("KIFD_best", "bovine4"): 0.995,
        ("KIFD_best", "bovine5.1"): 0.995,
        ("KIFD_best", "bovine5.2"): 0.995,
        ("KIFD_best", "bovine5.3"): 0.99,
        ("KIFD_best", "porcine1"): 0.99,
        ("KIFD_best", "porcine2"): 0.99,
        ("KIFD_best", "porcine3"): 0.98,
        ("KIFD_best", "porcine4"): 0.99,
        # PTA_best
        ("PTA_best", "bovine1"): 0.999,
        ("PTA_best", "bovine2"): 0.999,
        ("PTA_best", "bovine3"): 0.995,
        ("PTA_best", "bovine4"): 0.995,
        ("PTA_best", "bovine5.1"): 0.999,
        ("PTA_best", "bovine5.2"): 0.999,
        ("PTA_best", "bovine5.3"): 0.999,
        ("PTA_best", "porcine1"): 0.999,
        ("PTA_best", "porcine2"): 0.999,
        ("PTA_best", "porcine3"): 0.995,
        ("PTA_best", "porcine4"): 0.999,
    }

    detector_files, dataset_to_group = discover_detector_files(
        methods,
        datasets,
        groups,
        det_folder
    )

    eval_data = preload_evaluation_data(detector_files, gt_folder, det_folder, clamp_config=clamp_config,
                                        histogram_debug=False)

    generate_visualizations(eval_data, jpg_folder)

    compute_metrics(eval_data, dataset_to_group, metrics_folder, ndigits=3)

    generate_roc_curves(eval_data, roc_folder)


if __name__ == "__main__":
    main()
