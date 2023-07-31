import argparse
import os

from ..workflow import simple_workflow


def main():
    parser = argparse.ArgumentParser(
        prog="fmrierp", description="Extracts and plots erp like data from fMRI"
    )

    parser.add_argument(
        "-f",
        "--fmri_file",
        type=str,
        dest="fmri",
        help="The fMRI file to be processed.",
    )
    parser.add_argument(
        "-m",
        "--mask_file",
        dest="mask",
        type=str,
        help="The mask file from which the data is to be extracted.",
    )
    parser.add_argument(
        "-e",
        "--events_file",
        dest="events",
        type=str,
        help="The event file which has the events.",
    )
    parser.add_argument(
        "-o", "--out_dir", dest="outdir", type=str, help="Out file for tsv and figure."
    )
    parser.add_argument(
        "-en",
        "--event_name",
        dest="event_names",
        nargs="*",
        type=str,
        help="The column values to use for the erp.",
    )
    parser.add_argument("-tr", "--tr", dest="tr", type=float, help="TR of the data.")
    parser.add_argument(
        "--mask_name",
        dest="mask_name",
        type=str,
        help="Name of the mask, if multiple extractions.",
        default="",
    )
    parser.add_argument(
        "--window",
        dest="window",
        type=float,
        nargs="+",
        help="The window shape",
        default=[0, 15],
    )

    args = parser.parse_args()

    simple_workflow(
        args.events,
        args.fmri,
        args.mask,
        args.outdir,
        args.event_names,
        args.tr,
        mask_name=args.mask_name,
        window=args.window,
    )


if __name__ == "__main__":
    main()
