import argparse
import logging
import pathlib
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

_SERVICE_NAME = "ivritai-data-pipeline"
_SERVICE_VERSION = "v1.0.0"
_DEPLOYMENT_ENVIRONMENT = "na"


def _parse_dsn(dsn: str) -> tuple[str, str]:
    """Parse an Uptrace DSN into ``(dsn, otlp_http_base_url)``.

    The DSN format is ``https://token@host?grpc=<port>``.
    The full DSN string is passed as the ``uptrace-dsn`` header value —
    Uptrace Cloud requires the complete DSN for authentication, not just
    the token.  The HTTP base URL is derived from the scheme and host,
    stripping userinfo and the gRPC query param (HTTP transport only).

    The OTel HTTP log exporter only appends ``/v1/logs`` when using the
    *default* endpoint.  When an explicit ``endpoint`` is passed it is used
    verbatim, so we must include the full path.

    Example::

        "https://mytoken@api.uptrace.dev?grpc=4317"
        -> ("https://mytoken@api.uptrace.dev?grpc=4317", "https://api.uptrace.dev/v1/logs")
    """
    parsed = urlparse(dsn)
    base_url = f"{parsed.scheme}://{parsed.hostname}/v1/logs"
    return dsn, base_url


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Register logging-related CLI arguments onto *parser*."""
    parser.add_argument(
        "--logs-folder",
        type=str,
        help="Folder to store log files. If not specified, logging is disabled.",
    )
    parser.add_argument(
        "--log-uptrace-dsn",
        type=str,
        default=None,
        help=(
            "Uptrace DSN to enable log export to Uptrace via OTLP/HTTPS "
            "(requires the 'logging' optional dependency: pip install 'ivrit-ai-data-pipeline[logging]'). "
            "When provided, INFO-level logs are automatically forwarded to Uptrace."
        ),
    )


def configure_logging(args: argparse.Namespace) -> None:
    """Configure root logger from parsed CLI arguments.

    When ``args.logs_folder`` is provided, a rotating file handler writing to
    ``<logs_folder>/download_log`` is added and the root log level is set to
    INFO.  When it is None, all logging is suppressed (level set above CRITICAL).

    When ``args.log_uptrace_dsn`` is provided and the OTel SDK packages are
    installed, logs are additionally exported to Uptrace over OTLP/HTTP
    using a ``BatchLogRecordProcessor``.  gRPC is not used.
    """
    logging.basicConfig(level=logging.CRITICAL + 1)
    if not args.logs_folder:
        return

    logs_path = pathlib.Path(args.logs_folder)
    logs_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = RotatingFileHandler(
        logs_path / "download_log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)

    if args.log_uptrace_dsn:
        _configure_uptrace_http(args.log_uptrace_dsn, root_logger)


def _configure_uptrace_http(dsn: str, root_logger: logging.Logger) -> None:
    """Attach an OTLP/HTTP log exporter that forwards records to Uptrace.

    Uses the pure-HTTP exporter (``opentelemetry-exporter-otlp-proto-http``)
    so gRPC is never required.  The DSN is parsed to extract the token
    (sent as the ``uptrace-dsn`` header) and to derive the plain HTTP endpoint.
    """
    try:
        from opentelemetry._logs import set_logger_provider  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http import Compression  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (  # noqa: PLC0415
            OTLPLogExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler  # noqa: PLC0415
        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.semconv.resource import ResourceAttributes  # noqa: PLC0415
    except ImportError:
        root_logger.warning(
            "OpenTelemetry SDK packages are not installed; --log-uptrace-dsn has no effect. "
            "Install them with: pip install 'ivrit-ai-data-pipeline[logging]'"
        )
        return

    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: _SERVICE_NAME,
            ResourceAttributes.SERVICE_VERSION: _SERVICE_VERSION,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: _DEPLOYMENT_ENVIRONMENT,
        }
    )

    dsn_header, logs_endpoint = _parse_dsn(dsn)

    exporter = OTLPLogExporter(
        endpoint=logs_endpoint,
        headers={"uptrace-dsn": dsn_header},
        compression=Compression.Gzip,
        timeout=30,
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    set_logger_provider(logger_provider)

    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    root_logger.addHandler(otel_handler)

    root_logger.info(
        "Uptrace log export configured via OTLP/HTTPS (service=%s %s)",
        _SERVICE_NAME,
        _SERVICE_VERSION,
    )
