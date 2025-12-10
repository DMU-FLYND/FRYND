"""Gemini 2.5 Flash TTS 헬퍼 - 프리뷰 음성 출력 우선 사용"""

from __future__ import annotations

import base64
import io
import os
import wave
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - 런타임 의존성 안내용
    genai = None
    types = None
from gtts import gTTS

GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")


def text_to_speech(text: str, language_code: str = "ko") -> tuple[bytes, str]:
    """텍스트를 음성으로 변환합니다. Gemini 2.5 Flash TTS(프리뷰)를 우선 사용하고 실패 시 gTTS로 폴백합니다.

    Args:
        text: 변환할 텍스트
        language_code: 언어 코드 (기본값: ko)

    Returns:
        (오디오 바이트, MIME 타입)
    """
    try:
        return _generate_audio_with_gemini(text)
    except Exception:
        return _generate_audio_with_gtts(text, language_code)


def _generate_audio_with_gemini(text: str) -> tuple[bytes, str]:
    """Gemini 2.5 Flash 프리뷰 음성 기능을 사용해 오디오를 생성합니다."""
    if not genai or not types:
        raise RuntimeError("google-genai 패키지가 필요합니다. `pip install google-genai`로 설치하세요.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_TTS_VOICE)
                )
            ),
        ),
        request_options={"timeout": 30},
    )

    audio_bytes, mime_type = _extract_audio_bytes(response)
    if not audio_bytes:
        raise RuntimeError("Gemini가 오디오 데이터를 반환하지 않았습니다.")
    return audio_bytes, mime_type


def _extract_audio_bytes(response: Any) -> tuple[bytes, str]:
    """Gemini 응답에서 오디오 바이트를 추출합니다."""
    candidates = getattr(response, "candidates", None) or []

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue

        for part in getattr(content, "parts", []):
            inline = getattr(part, "inline_data", None)
            if not inline:
                continue

            data = getattr(inline, "data", None)
            mime_type = getattr(inline, "mime_type", "") or ""
            if not data:
                continue

            if isinstance(data, bytes):
                return _normalize_audio(data, mime_type)
            if isinstance(data, str):
                try:
                    decoded = base64.b64decode(data)
                    return _normalize_audio(decoded, mime_type)
                except Exception:
                    continue

    return b"", ""


def _normalize_audio(raw_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """PCM 데이터를 WAV로 감싸거나 알려진 MIME 타입을 그대로 반환합니다."""
    normalized_mime = mime_type.lower()

    if normalized_mime in ("audio/mp3", "audio/mpeg"):
        return raw_bytes, "audio/mp3"

    if normalized_mime in ("audio/wav", "audio/wave", "audio/x-wav", "audio/x-wave"):
        return raw_bytes, "audio/wav"

    # Gemini 프리뷰는 PCM일 수 있으므로 WAV로 감싼다.
    wav_bytes = _pcm_to_wav(raw_bytes)
    return wav_bytes, "audio/wav"


def _pcm_to_wav(pcm_bytes: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
    """PCM 바이트를 WAV 컨테이너로 감싸서 반환합니다."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    buffer.seek(0)
    return buffer.read()


def _generate_audio_with_gtts(text: str, language_code: str) -> tuple[bytes, str]:
    """gTTS를 사용하여 음성을 생성합니다 (폴백용)."""
    try:
        tts = gTTS(text=text, lang=language_code, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read(), "audio/mp3"
    except Exception as e:
        raise RuntimeError(f"TTS 변환 중 오류가 발생했습니다: {e}")


__all__ = ["text_to_speech"]
