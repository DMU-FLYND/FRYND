"""OpenAI Realtime API를 사용한 양방향 음성 대화"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Callable, Optional

import streamlit as st

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None
    WebSocketClientProtocol = None


class RealtimeVoiceChat:
    """OpenAI Realtime API를 사용한 실시간 음성 대화 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API 키. None이면 환경변수에서 가져옴
        """
        if not websockets:
            raise RuntimeError("websockets 패키지가 필요합니다. `pip install websockets`로 설치하세요.")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.ws: Optional[WebSocketClientProtocol] = None
        self.model = "gpt-4o-realtime-preview-2024-12-17"
        self.voice = os.getenv("OPENAI_REALTIME_VOICE", "shimmer")
        self.ws_url = "wss://api.openai.com/v1/realtime"
        
    async def connect(self) -> None:
        """Realtime API 웹소켓에 연결"""
        url = f"{self.ws_url}?model={self.model}"
        
        # websockets.connect는 additional_headers 파라미터 사용
        self.ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        )
        
        # 세션 설정
        await self._configure_session()
    
    async def _configure_session(self) -> None:
        """세션 설정 업데이트"""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "당신은 FRYND 항공 상담 챗봇입니다. "
                    "항공권 검색, 기내식 정보, FAQ에 대해 친절하게 답변하세요. "
                    "한국어로 자연스럽게 대화하세요."
                ),
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.8,
                "max_response_output_tokens": 4096
            }
        }
        
        await self.ws.send(json.dumps(config))
    
    async def send_text(self, text: str) -> None:
        """텍스트 메시지 전송"""
        if not self.ws:
            raise RuntimeError("웹소켓이 연결되지 않았습니다.")
        
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }
        
        await self.ws.send(json.dumps(message))
        
        # 응답 생성 요청
        await self.ws.send(json.dumps({"type": "response.create"}))
    
    async def send_audio(self, audio_bytes: bytes) -> None:
        """오디오 데이터 전송 (PCM16 형식)"""
        if not self.ws:
            raise RuntimeError("웹소켓이 연결되지 않았습니다.")
        
        # base64 인코딩
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }
        
        await self.ws.send(json.dumps(message))
    
    async def commit_audio(self) -> None:
        """오디오 입력 완료 및 응답 생성 요청"""
        if not self.ws:
            raise RuntimeError("웹소켓이 연결되지 않았습니다.")
        
        # 오디오 입력 커밋
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        
        # 응답 생성 요청
        await self.ws.send(json.dumps({"type": "response.create"}))
    
    async def listen(
        self,
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        웹소켓 메시지 수신 및 처리
        
        Args:
            on_audio: 오디오 데이터 수신 시 호출될 콜백
            on_text: 텍스트 응답 수신 시 호출될 콜백
            on_transcript: 음성 인식 결과 수신 시 호출될 콜백
            on_error: 에러 발생 시 호출될 콜백
        """
        if not self.ws:
            raise RuntimeError("웹소켓이 연결되지 않았습니다.")
        
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    event_type = data.get("type", "")
                    
                    # 오디오 응답
                    if event_type == "response.audio.delta":
                        audio_base64 = data.get("delta", "")
                        if audio_base64 and on_audio:
                            audio_bytes = base64.b64decode(audio_base64)
                            on_audio(audio_bytes)
                    
                    # 텍스트 응답
                    elif event_type == "response.text.delta":
                        text = data.get("delta", "")
                        if text and on_text:
                            on_text(text)
                    
                    # 음성 인식 결과
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = data.get("transcript", "")
                        if transcript and on_transcript:
                            on_transcript(transcript)
                    
                    # 에러 처리
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get("message", "Unknown error")
                        if on_error:
                            on_error(error_msg)
                
                except json.JSONDecodeError:
                    if on_error:
                        on_error("Invalid JSON received")
                except Exception as e:
                    if on_error:
                        on_error(str(e))
        
        except websockets.exceptions.ConnectionClosed:
            if on_error:
                on_error("Connection closed")
    
    async def close(self) -> None:
        """웹소켓 연결 종료"""
        if self.ws:
            await self.ws.close()
            self.ws = None


def render_realtime_voice_ui() -> None:
    """Streamlit UI에서 실시간 음성 대화 인터페이스 렌더링"""
    st.header("🎙️ 실시간 음성 대화")
    st.caption("텍스트로 입력하거나 오디오 파일을 업로드하세요. AI가 음성으로 답변합니다.")
    
    # 세션 상태 초기화
    if "realtime_messages" not in st.session_state:
        st.session_state.realtime_messages = []
    
    # 대화 내역 표시
    for msg in st.session_state.realtime_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")
    
    # 입력 방식 선택
    col1, col2 = st.columns([1, 5])
    
    with col1:
        input_mode = st.selectbox(
            "입력 방식",
            ["텍스트", "오디오 파일"],
            label_visibility="collapsed"
        )
    
    with col2:
        if input_mode == "텍스트":
            user_input = st.chat_input("메시지를 입력하세요...")
            if user_input:
                _handle_text_input(user_input)
        else:
            audio_file = st.file_uploader(
                "음성 파일을 업로드하세요 (mp3, wav, m4a 등)",
                type=["mp3", "wav", "m4a", "ogg", "webm"],
                key="audio_upload"
            )
            if audio_file:
                if st.button("🎤 음성 처리", type="primary"):
                    _handle_audio_file(audio_file)


def _handle_text_input(text: str) -> None:
    """텍스트 입력 처리"""
    # 사용자 메시지 추가
    st.session_state.realtime_messages.append({
        "role": "user",
        "content": text
    })
    
    # TODO: Realtime API 호출
    with st.spinner("응답 생성 중..."):
        try:
            # 비동기 함수를 동기적으로 실행
            asyncio.run(_send_and_receive(text))
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")


def _handle_voice_input() -> None:
    """음성 입력 처리"""
    st.info("음성 입력 기능은 현재 개발 중입니다.")
    # TODO: 마이크에서 오디오 녹음 및 Realtime API 전송


def _handle_audio_file(audio_file) -> None:
    """업로드된 오디오 파일 처리"""
    try:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")
            st.caption("🎤 음성 메시지")
        
        # 오디오 파일 읽기
        audio_bytes = audio_file.read()
        
        # TODO: 오디오를 텍스트로 변환 (Whisper API 사용 가능)
        # 현재는 텍스트 입력과 동일하게 처리
        with st.spinner("음성을 인식하고 응답을 생성하는 중..."):
            asyncio.run(_send_audio_and_receive(audio_bytes))
        
        st.rerun()
    
    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")


async def _send_audio_and_receive(audio_bytes: bytes) -> None:
    """오디오 전송 및 응답 수신"""
    chat = RealtimeVoiceChat()
    
    try:
        await chat.connect()
        
        # 응답 수집용
        response_text = ""
        audio_chunks = []
        transcript = ""
        
        def on_text(delta: str):
            nonlocal response_text
            response_text += delta
        
        def on_audio(audio_data: bytes):
            audio_chunks.append(audio_data)
        
        def on_transcript(text: str):
            nonlocal transcript
            transcript = text
        
        def on_error(error_msg: str):
            st.error(f"오류: {error_msg}")
        
        # TODO: 오디오를 적절한 형식(PCM16)으로 변환
        # 현재는 간단하게 Whisper API를 사용하여 텍스트로 변환 후 전송
        
        # OpenAI Whisper로 음성 인식
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )
            
            user_text = transcription.text
            
            # 사용자 메시지 추가
            st.session_state.realtime_messages.append({
                "role": "user",
                "content": f"🎤 {user_text}"
            })
            
            # 텍스트로 응답 생성
            await chat.send_text(user_text)
            
            # 응답 수신 (타임아웃 설정)
            try:
                await asyncio.wait_for(
                    chat.listen(
                        on_text=on_text,
                        on_audio=on_audio,
                        on_transcript=on_transcript,
                        on_error=on_error
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                pass
            
            # 응답을 세션 상태에 저장
            if response_text:
                audio_data = b"".join(audio_chunks) if audio_chunks else None
                st.session_state.realtime_messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "audio": audio_data
                })
        
        finally:
            # 임시 파일 삭제
            import os as os_module
            if os_module.path.exists(tmp_path):
                os_module.remove(tmp_path)
    
    finally:
        await chat.close()


async def _send_and_receive(text: str) -> None:
    """메시지 전송 및 응답 수신"""
    chat = RealtimeVoiceChat()
    
    try:
        await chat.connect()
        
        # 응답 수집용
        response_text = ""
        audio_chunks = []
        
        def on_text(delta: str):
            nonlocal response_text
            response_text += delta
        
        def on_audio(audio_bytes: bytes):
            audio_chunks.append(audio_bytes)
        
        def on_error(error_msg: str):
            st.error(f"오류: {error_msg}")
        
        # 텍스트 전송
        await chat.send_text(text)
        
        # 응답 수신 (타임아웃 설정)
        try:
            await asyncio.wait_for(
                chat.listen(
                    on_text=on_text,
                    on_audio=on_audio,
                    on_error=on_error
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            pass
        
        # 응답을 세션 상태에 저장
        if response_text:
            audio_data = b"".join(audio_chunks) if audio_chunks else None
            st.session_state.realtime_messages.append({
                "role": "assistant",
                "content": response_text,
                "audio": audio_data
            })
    
    finally:
        await chat.close()


__all__ = ["RealtimeVoiceChat", "render_realtime_voice_ui"]
