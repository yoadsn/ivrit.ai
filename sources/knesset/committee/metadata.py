from typing import Optional

from pydantic import ConfigDict

from sources.common.metadata import NormalizedEntryMetadata

source_type = "knesset"
plenum_source_id = "plenum"
committee_source_id = "committee"


class CommitteeMetadata(NormalizedEntryMetadata):
    """Metadata for a Knesset committee.

    Extra fields from the metadata manifest CSV are stored verbatim when
    ``--metadata-manifest-file`` is supplied to the downloader.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    duration: Optional[float] = None
    title: Optional[str] = None
    session_date: Optional[str] = None
    language: str = "he"
