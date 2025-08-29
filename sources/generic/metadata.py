from typing import Optional

from sources.common.metadata import NormalizedEntryMetadata

source_type = "generic"


class GenericMetadata(NormalizedEntryMetadata):
    """Metadata for a generic audio/transcript dataset entry."""

    language: str
    duration: float
