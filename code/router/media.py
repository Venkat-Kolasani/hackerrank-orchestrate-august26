"""Multimodal perception: MediaInterpreter with offline fallback and cache.

Returns structured ``ContentSummary`` fields only — never routing decisions.
All media-derived text is untrusted data for later safety scanning.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import prompts
from .data import Dataset
from .types import ContentSummary, MessageRecord

logger = logging.getLogger(__name__)

# Bump when prompt/schema changes so stale cache entries are ignored.
CACHE_VERSION = "media-v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "media"

# Env configuration (documented in code/README.md)
ENV_API_KEY = "OPENAI_API_KEY"
ENV_PROVIDER = "ROUTER_MEDIA_PROVIDER"  # auto | offline | openai
ENV_IMAGE_MODEL = "ROUTER_IMAGE_MODEL"
ENV_ASR_MODEL = "ROUTER_ASR_MODEL"
ENV_CACHE_DIR = "ROUTER_MEDIA_CACHE_DIR"
ENV_DISABLE_CACHE = "ROUTER_MEDIA_CACHE_DISABLE"

DEFAULT_IMAGE_MODEL = "gpt-4o-mini"
DEFAULT_ASR_MODEL = "whisper-1"


@dataclass(frozen=True)
class MediaPerceptionResult:
    """Cached perception payload (content channels only)."""

    ocr_text: str = ""
    asr_transcript: str = ""
    caption: str = ""
    interpreted: bool = False
    warning: str = ""
    source: str = "offline"  # offline | openai | cache | missing

    def apply(self, base: ContentSummary) -> ContentSummary:
        return ContentSummary(
            message_text=base.message_text,
            ocr_text=self.ocr_text or base.ocr_text,
            asr_transcript=self.asr_transcript or base.asr_transcript,
            caption=self.caption or base.caption,
            media_type=base.media_type,
            media_id=base.media_id,
        )


class MediaInterpreter(ABC):
    """Interpret image/voice media into normalized content channels."""

    name: str = "base"

    @abstractmethod
    def interpret(
        self,
        *,
        media_type: str,
        media_id: str,
        file_path: Optional[Path],
        available: bool,
        accompanying_text: str,
    ) -> MediaPerceptionResult:
        raise NotImplementedError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _warn(message: str) -> None:
    logger.warning(message)
    warnings.warn(message, stacklevel=2)


class OfflineMediaInterpreter(MediaInterpreter):
    """Text-only fallback. Never invents OCR/ASR content."""

    name = "offline"

    def __init__(self) -> None:
        self._warned: set[str] = set()

    def interpret(
        self,
        *,
        media_type: str,
        media_id: str,
        file_path: Optional[Path],
        available: bool,
        accompanying_text: str,
    ) -> MediaPerceptionResult:
        del accompanying_text  # retained for interface parity; unused offline
        if media_type not in {"image", "voice"}:
            return MediaPerceptionResult(interpreted=True, source="offline")
        if not available or file_path is None:
            msg = (
                f"media {media_id} ({media_type}) unavailable; "
                "offline interpreter leaving OCR/ASR empty"
            )
            if media_id not in self._warned:
                self._warned.add(media_id)
                _warn(msg)
            return MediaPerceptionResult(
                interpreted=False,
                warning=msg,
                source="missing",
            )
        msg = (
            f"offline media interpreter: skipping {media_type} analysis for "
            f"{media_id} (no provider credentials / forced offline)"
        )
        if media_id not in self._warned:
            self._warned.add(media_id)
            _warn(msg)
        return MediaPerceptionResult(
            interpreted=False,
            warning=msg,
            source="offline",
        )


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


class ApiMediaInterpreter(MediaInterpreter):
    """OpenAI vision + Whisper path. Requires ``OPENAI_API_KEY``."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        image_model: Optional[str] = None,
        asr_model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(ENV_API_KEY, "")
        self.image_model = image_model or os.environ.get(
            ENV_IMAGE_MODEL, DEFAULT_IMAGE_MODEL
        )
        self.asr_model = asr_model or os.environ.get(ENV_ASR_MODEL, DEFAULT_ASR_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(f"{ENV_API_KEY} is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai package is required for ApiMediaInterpreter; "
                "pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def interpret(
        self,
        *,
        media_type: str,
        media_id: str,
        file_path: Optional[Path],
        available: bool,
        accompanying_text: str,
    ) -> MediaPerceptionResult:
        if media_type not in {"image", "voice"}:
            return MediaPerceptionResult(interpreted=True, source="openai")
        if not available or file_path is None or not file_path.is_file():
            msg = (
                f"media {media_id} ({media_type}) file missing; "
                "cannot run provider perception"
            )
            _warn(msg)
            return MediaPerceptionResult(
                interpreted=False,
                warning=msg,
                source="missing",
            )
        try:
            if media_type == "image":
                return self._interpret_image(
                    media_id=media_id,
                    file_path=file_path,
                    accompanying_text=accompanying_text,
                )
            return self._interpret_voice(
                media_id=media_id,
                file_path=file_path,
                accompanying_text=accompanying_text,
            )
        except Exception as exc:  # noqa: BLE001 - keep router runnable
            msg = (
                f"provider media interpretation failed for {media_id}: "
                f"{type(exc).__name__}; leaving OCR/ASR empty"
            )
            _warn(msg)
            return MediaPerceptionResult(
                interpreted=False,
                warning=msg,
                source="openai",
            )

    def _interpret_image(
        self,
        *,
        media_id: str,
        file_path: Path,
        accompanying_text: str,
    ) -> MediaPerceptionResult:
        client = self._get_client()
        mime, _ = mimetypes.guess_type(str(file_path))
        mime = mime or "image/jpeg"
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        response = client.chat.completions.create(
            model=self.image_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompts.IMAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts.image_user_prompt(
                                media_id=media_id,
                                accompanying_text=accompanying_text,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        )
        payload = _extract_json_object(response.choices[0].message.content or "")
        ocr = str(payload.get("ocr_text") or "").strip()
        caption = str(payload.get("caption") or "").strip()
        return MediaPerceptionResult(
            ocr_text=ocr,
            caption=caption,
            interpreted=bool(ocr or caption),
            source="openai",
        )

    def _interpret_voice(
        self,
        *,
        media_id: str,
        file_path: Path,
        accompanying_text: str,
    ) -> MediaPerceptionResult:
        client = self._get_client()
        with file_path.open("rb") as audio:
            transcription = client.audio.transcriptions.create(
                model=self.asr_model,
                file=audio,
                temperature=0,
            )
        transcript = (getattr(transcription, "text", None) or str(transcription)).strip()
        # Optional structuring pass; if it fails, keep raw transcript only.
        structured_transcript = transcript
        try:
            response = client.chat.completions.create(
                model=self.image_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompts.VOICE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompts.voice_user_prompt(
                            media_id=media_id,
                            accompanying_text=accompanying_text,
                            transcript=transcript,
                        ),
                    },
                ],
            )
            payload = _extract_json_object(response.choices[0].message.content or "")
            structured = str(payload.get("asr_transcript") or "").strip()
            if structured:
                structured_transcript = structured
        except Exception:  # noqa: BLE001
            structured_transcript = transcript

        return MediaPerceptionResult(
            asr_transcript=structured_transcript,
            interpreted=bool(structured_transcript),
            source="openai",
        )


class CachedMediaInterpreter(MediaInterpreter):
    """Disk cache keyed by media content hash + interpreter identity."""

    def __init__(
        self,
        inner: MediaInterpreter,
        *,
        cache_dir: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.enabled = enabled
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(ENV_CACHE_DIR)
            or DEFAULT_CACHE_DIR
        )
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        *,
        media_type: str,
        file_path: Path,
    ) -> str:
        digest = _file_sha256(file_path)
        blob = f"{CACHE_VERSION}|{self.inner.name}|{media_type}|{digest}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def interpret(
        self,
        *,
        media_type: str,
        media_id: str,
        file_path: Optional[Path],
        available: bool,
        accompanying_text: str,
    ) -> MediaPerceptionResult:
        if (
            self.enabled
            and available
            and file_path is not None
            and file_path.is_file()
            and media_type in {"image", "voice"}
        ):
            key = self._cache_key(media_type=media_type, file_path=file_path)
            path = self._cache_path(key)
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    result = MediaPerceptionResult(
                        ocr_text=str(data.get("ocr_text") or ""),
                        asr_transcript=str(data.get("asr_transcript") or ""),
                        caption=str(data.get("caption") or ""),
                        interpreted=bool(data.get("interpreted")),
                        warning=str(data.get("warning") or ""),
                        source="cache",
                    )
                    return result
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass

            result = self.inner.interpret(
                media_type=media_type,
                media_id=media_id,
                file_path=file_path,
                available=available,
                accompanying_text=accompanying_text,
            )
            try:
                path.write_text(
                    json.dumps(asdict(result), ensure_ascii=True, sort_keys=True),
                    encoding="utf-8",
                )
            except OSError:
                pass
            return result

        return self.inner.interpret(
            media_type=media_type,
            media_id=media_id,
            file_path=file_path,
            available=available,
            accompanying_text=accompanying_text,
        )


def provider_from_env() -> str:
    """Return configured provider: auto | offline | openai."""
    raw = (os.environ.get(ENV_PROVIDER) or "auto").strip().lower()
    if raw in {"auto", "offline", "openai"}:
        return raw
    return "auto"


def build_media_interpreter(
    *,
    provider: Optional[str] = None,
    cache: Optional[bool] = None,
    cache_dir: Optional[Path] = None,
) -> MediaInterpreter:
    """Factory used by the production router.

    Default ``auto`` selects OpenAI when ``OPENAI_API_KEY`` is set, otherwise
    the offline interpreter. Cache is on unless ``ROUTER_MEDIA_CACHE_DISABLE=1``.
    """
    mode = (provider or provider_from_env()).strip().lower()
    if mode == "auto":
        mode = "openai" if os.environ.get(ENV_API_KEY) else "offline"

    if mode == "openai":
        if not os.environ.get(ENV_API_KEY):
            _warn(
                f"{ENV_API_KEY} missing; falling back to OfflineMediaInterpreter"
            )
            inner: MediaInterpreter = OfflineMediaInterpreter()
        else:
            inner = ApiMediaInterpreter()
    else:
        inner = OfflineMediaInterpreter()

    cache_disabled = (os.environ.get(ENV_DISABLE_CACHE) or "").strip() in {
        "1",
        "true",
        "yes",
    }
    use_cache = (not cache_disabled) if cache is None else cache
    if use_cache:
        return CachedMediaInterpreter(inner, cache_dir=cache_dir, enabled=True)
    return inner


def interpret_message(
    dataset: Dataset,
    message: MessageRecord,
    interpreter: MediaInterpreter,
) -> tuple[ContentSummary, MediaPerceptionResult]:
    """Build a ContentSummary for ``message`` using ``interpreter``."""
    base = ContentSummary(
        message_text=message.message_text or "",
        media_type=message.media_type,
        media_id=message.media_id,
    )
    if not message.media_type or not message.media_id:
        empty = MediaPerceptionResult(interpreted=True, source="offline")
        return base, empty

    media_ref = dataset.resolve_media(message.media_type, message.media_id)
    file_path = (
        Path(media_ref.absolute_path)
        if media_ref and media_ref.absolute_path
        else None
    )
    available = bool(media_ref and media_ref.available and file_path)
    result = interpreter.interpret(
        media_type=message.media_type,
        media_id=message.media_id,
        file_path=file_path,
        available=available,
        accompanying_text=message.message_text or "",
    )
    return result.apply(base), result
