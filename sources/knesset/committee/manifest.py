from sources.common.manifest import build_manifest as common_build_manifest, COMMON_COLUMNS


# Source-specific columns for knesset committee
COMMITTEE_COLUMNS = [
    "session_id",
    "duration",
    "session_date",
]


def build_manifest(input_folder: str) -> None:
    """Build a manifest CSV file for knesset committee sessions.

    Args:
        input_folder: Path to the folder containing metadata.json files
    """
    columns = COMMON_COLUMNS + COMMITTEE_COLUMNS
    common_build_manifest(input_folder, columns)
