from typing import Optional

from sources.common.metadata import NormalizedEntryMetadata

source_type = "knesset"
plenum_source_id = "plenum"
committee_source_id = "committee"


class CommitteeMetadata(NormalizedEntryMetadata):
    """Metadata for a Knesset committee."""

    session_id: str
    duration: Optional[float] = None
    title: Optional[str] = None
    session_date: Optional[str] = None
    language: str = "he"
