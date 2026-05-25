"""
CSI data annotation tool for in-situ data collection.

Simple CLI tool to label CSI segments with activity labels
during the data collection phase at deployment site.

Usage:
    python training/annotator.py --data data/raw/session_001/
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI data annotation tool")
    parser.add_argument("--data", required=True, help="Path to raw CSI capture directory")
    parser.add_argument("--output", default="data/annotations/", help="Output directory for annotations")
    parser.add_argument("--labels", nargs="+", default=["fall", "walking", "sitting", "lying", "standing"],
                        help="Activity labels to choose from")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Annotation tool for: {args.data}")
    logger.info(f"Labels: {args.labels}")

    # TODO: Implement annotation CLI in Phase 1
    # Features:
    #   - Load CSI capture segments
    #   - Display CSI amplitude heatmap or variance plot
    #   - Prompt user for label selection
    #   - Save annotations as {timestamp_start, timestamp_end, label}
    #   - Export to data/annotations/ directory

    logger.info("Annotation tool not yet implemented — placeholder for Phase 1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
