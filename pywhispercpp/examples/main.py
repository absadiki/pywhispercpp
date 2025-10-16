#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple Command Line Interface to test the package
"""
import argparse
import importlib.metadata
import logging

import pywhispercpp.constants as constants

__version__ = importlib.metadata.version('pywhispercpp')

__header__ = f"""
PyWhisperCpp
A simple Command Line Interface to test the package
Version: {__version__}               
====================================================
"""

from pywhispercpp.model import Model
import pywhispercpp.utils as utils


def _get_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if constants.PARAMS_SCHEMA[arg]['type'] is bool:
                if getattr(args, arg).lower() == 'false':
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = constants.PARAMS_SCHEMA[arg]['type'](getattr(args, arg))
    return params


def run(args):
    logging.info(f"Running with model `{args.model}`")
    params = _get_params(args)
    logging.info(f"Running with params {params}")
    m = Model(model=args.model, **params)
    logging.info(f"System info: n_threads = {m.get_params()['n_threads']} | Processors = {args.processors} "
                 f"| {m.system_info()}")
    for file in args.media_file:
        segs = []
        try:
            logging.info(f"Processing file {file} ...")
            m.transcribe(file,
                         n_processors=int(args.processors) if args.processors else None,
                         new_segment_callback=lambda s: segs.append(s),
                         start_time=args.start_time,
                         end_time=args.end_time if args.end_time is not None else None,
                         ),
            m.print_timings()
        except KeyboardInterrupt:
            logging.info("Transcription manually stopped")
            break
        except Exception as e:
            logging.error(f"Error while processing file {file}: {e}")
        finally:
            # output stuff
            if segs:
                segs = [seg.shift(args.start_time * 100) for seg in segs]
                if args.output_txt:
                    logging.info(f"Saving result as a txt file ...")
                    txt_file = utils.output_txt(segs, file)
                    logging.info(f"txt file saved to {txt_file}")
                if args.output_vtt:
                    logging.info(f"Saving results as a vtt file ...")
                    vtt_file = utils.output_vtt(segs, file)
                    logging.info(f"vtt file saved to {vtt_file}")
                if args.output_srt:
                    logging.info(f"Saving results as a srt file ...")
                    srt_file = utils.output_srt(segs, file)
                    logging.info(f"srt file saved to {srt_file}")
                if args.output_csv:
                    logging.info(f"Saving results as a csv file ...")
                    csv_file = utils.output_csv(segs, file)
                    logging.info(f"csv file saved to {csv_file}")


def main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('media_file', type=str, nargs='+', help="The path of the media file or a list of files "
                                                                "separated by space")

    parser.add_argument('-m', '--model', default='tiny', help="Path to the `ggml` model, or just the model name")

    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--processors', help="number of processors to use during computation")
    parser.add_argument('-otxt', '--output-txt', action='store_true', help="output result in a text file")
    parser.add_argument('-ovtt', '--output-vtt', action='store_true', help="output result in a vtt file")
    parser.add_argument('-osrt', '--output-srt', action='store_true', help="output result in a srt file")
    parser.add_argument('-ocsv', '--output-csv', action='store_true', help="output result in a CSV file")
    parser.add_argument( '-st', '--start-time', type=float, default=0,
                        help="start time in seconds for transcription (default: 0)")
    parser.add_argument('-et','--end-time', type=float,
                        help="end time in seconds for transcription (optional, transcribes until end if not specified)")
    # add params from PARAMS_SCHEMA
    for param in constants.PARAMS_SCHEMA:
        param_fields = constants.PARAMS_SCHEMA[param]
        parser.add_argument(f'--{param}',
                            help=f'{param_fields["description"]}')

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
