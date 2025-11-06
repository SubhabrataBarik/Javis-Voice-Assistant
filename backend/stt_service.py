import logging
import logging.config
import logging.handlers
import asyncio
import time
import os
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
from dotenv import load_dotenv

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingError,
    StreamingEvents,
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
)

# For inserting the final utterance into the VoiceState
from langchain_core.messages import HumanMessage
from VoiceState import VoiceState

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
DEFAULT_STT_TIMEOUT = float(os.getenv("STT_TIMEOUT", "30"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))

if not API_KEY:
    raise ValueError("❌ Missing AssemblyAI API key. Check your .env file.")

# -------------------------------
# ✅ File-Only Structured Logging Configuration
# -------------------------------
def setup_file_only_logging():
    """Enhanced logging configuration with environment variable control"""
    
    # Get configuration from environment variables
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE', 'stt_service.log')
    max_file_size = int(os.getenv('LOG_MAX_SIZE_MB', '10')) * 1024 * 1024
    
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            base_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            
            if hasattr(record, '__dict__') and record.levelno <= logging.INFO:
                extras = []
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                 'pathname', 'filename', 'module', 'lineno', 'funcName',
                                 'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                                 'processName', 'process', 'getMessage', 'asctime']:
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:97] + "..."
                        extras.append(f"{key}={str_value}")
                
                if extras:
                    base_format += f" | {' | '.join(extras)}"
            
            formatter = logging.Formatter(base_format)
            return formatter.format(record)
    
    from logging.handlers import RotatingFileHandler
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_file_size, 
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(StructuredFormatter())
    
    root_logger = logging.getLogger()
    selected_level = level_mapping.get(log_level, logging.INFO)
    root_logger.setLevel(selected_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    
    if selected_level >= logging.WARNING:
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)
        logging.getLogger('assemblyai').setLevel(logging.ERROR)
        logging.getLogger('asyncio').setLevel(logging.ERROR)
    elif selected_level >= logging.INFO:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('assemblyai').setLevel(logging.WARNING)
    
    print(f"📝 STT Service Logging Configuration:")
    print(f"   🎚️  Level: {log_level} ({logging.getLevelName(selected_level)})")
    print(f"   📁  File: {log_file}")
    print(f"   💾  Max size: {max_file_size // (1024*1024)} MB per file")
    print(f"   🔄  Backup files: 5")
    print(f"   📊  Performance mode: {'High' if selected_level >= logging.WARNING else 'Balanced'}")
    print(f"   🔗  Connection: Fresh per conversation (Fixed)")
    print(f"   📊  Standard streaming: Reliable")
    
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system configured with fixed connection management",
        extra={
            "event_type": "logging_configured",
            "log_level": log_level,
            "log_file": log_file,
            "max_file_size_mb": max_file_size // (1024*1024),
            "backup_count": 5,
            "performance_optimized": selected_level >= logging.WARNING,
            "connection_pooling": False,  # ✅ CHANGED: Disabled to fix rate issues
            "streaming_mode": "standard_reliable"
        }
    )

setup_file_only_logging()
logger = logging.getLogger(__name__)

# -------------------------------
# ✅ Clean Terminal Output Functions
# -------------------------------
def terminal_info(message: str):
    print(message)

def terminal_listening():
    print("🎙️ Listening... Speak into your mic")

def terminal_processing(text: str):
    status = "🗣️" if len(text.split()) > 2 else "💬"
    print(f"{status} {text}")

def terminal_final_capture(text: str):
    print(f"\n✅ You said: {text}")

# -------------------------------
# Helpers: attribute access + safe utter construction
# -------------------------------
def _get_attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        try:
            return obj.get(name, default)
        except Exception:
            return default

def _construct_utterance_from_words(event: TurnEvent) -> str | None:
    words = _get_attr(event, "words", None)
    if not words:
        return None
    final_words = []
    for w in words:
        word_is_final = _get_attr(w, "word_is_final", None)
        if word_is_final is None and isinstance(w, dict):
            word_is_final = w.get("word_is_final", False)
        text = _get_attr(w, "text", None)
        if text is None and isinstance(w, dict):
            text = w.get("text")
        if word_is_final and text:
            final_words.append(text)
    if not final_words:
        return None
    return " ".join(final_words)

def _get_best_utter(event: TurnEvent) -> str | None:
    utter = _get_attr(event, "utterance", None)
    if utter:
        return utter
    utter = _construct_utterance_from_words(event)
    if utter:
        return utter
    return _get_attr(event, "transcript", None)

# -------------------------------
# ✅ Production Telemetry and Metrics
# -------------------------------
class STTTelemetry:
    def __init__(self):
        self.session_start_time: Optional[float] = None
        self.turn_count: int = 0
        self.latency_history: List[float] = []
        self.error_count: int = 0
        self.total_sessions: int = 0
        self.successful_sessions: int = 0
        self.total_processing_time: float = 0.0
        self.word_count_history: List[int] = []
        self.confidence_history: List[float] = []
        
    def start_session(self, session_id: str):
        self.session_start_time = time.time()
        self.turn_count = 0
        self.total_sessions += 1
        
        logger.info(
            "STT session telemetry started",
            extra={
                "event_type": "telemetry_session_start",
                "session_id": session_id,
                "total_sessions": self.total_sessions,
                "timestamp": datetime.utcnow().isoformat(),
                "component": "stt_telemetry"
            }
        )
    
    def track_turn(self, utterance: str, is_final: bool, confidence: float = None):
        self.turn_count += 1
        turn_latency = time.time() - self.session_start_time if self.session_start_time else 0
        word_count = len(utterance.split()) if utterance else 0
        
        if word_count > 0:
            self.word_count_history.append(word_count)
        if confidence is not None:
            self.confidence_history.append(confidence)
        
        logger.info(
            "Turn metrics tracked",
            extra={
                "event_type": "telemetry_turn",
                "turn_number": self.turn_count,
                "utterance_length": len(utterance),
                "word_count": word_count,
                "is_final": is_final,
                "turn_latency_ms": round(turn_latency * 1000, 2),
                "confidence_score": confidence
            }
        )
        
        if is_final:
            self.latency_history.append(turn_latency)
    
    def track_completion(self, final_result: str = None, success: bool = True):
        if success:
            self.successful_sessions += 1
        
        total_time = time.time() - self.session_start_time if self.session_start_time else 0
        self.total_processing_time += total_time
        
        success_rate = self.successful_sessions / self.total_sessions if self.total_sessions > 0 else 0
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        
        logger.info(
            "STT session telemetry completed",
            extra={
                "event_type": "telemetry_session_complete",
                "session_duration_ms": round(total_time * 1000, 2),
                "total_turns": self.turn_count,
                "success": success,
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency * 1000, 2),
                "final_result_length": len(final_result) if final_result else 0
            }
        )
    
    def track_error(self, error_type: str, error_message: str, session_id: str = None):
        self.error_count += 1
        
        logger.error(
            "STT error tracked in telemetry",
            extra={
                "event_type": "telemetry_error",
                "session_id": session_id,
                "error_type": error_type,
                "error_message": error_message,
                "total_errors": self.error_count,
                "error_rate": round(self.error_count / self.total_sessions, 3) if self.total_sessions > 0 else 0
            }
        )

# -------------------------------
# ✅ Fixed Callback Handler (Safe Event Loop Handling)
# -------------------------------
class STTCallbackHandler:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.result = None
        self.completion_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.error_occurred = False
        self.session_id = None
        self.partial_queue = asyncio.Queue()
        self.final_queue = asyncio.Queue()
        self.telemetry = STTTelemetry()
        self.is_loop_closed = False  # ✅ ADDED: Track loop state to prevent errors
        
        self.on_partial_callback: Optional[Callable[[str], None]] = None
        self.on_final_callback: Optional[Callable[[str], None]] = None
        self.on_session_start_callback: Optional[Callable[[str], None]] = None
    
    def _safe_loop_call(self, func):
        """✅ ADDED: Safely call loop functions, handling closed loop"""
        try:
            if not self.is_loop_closed and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(func)
            else:
                logger.debug("Skipping loop call - event loop is closed")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                self.is_loop_closed = True
                logger.debug("Event loop closed during callback")
            else:
                raise
    
    def on_begin(self, client, event: BeginEvent):
        session_id = _get_attr(event, "id", "unknown")
        self.session_id = session_id
        
        logger.info(
            "STT session started",
            extra={
                "event_type": "session_start", 
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        self.telemetry.start_session(session_id)
        
        if self.on_session_start_callback:
            try:
                self.on_session_start_callback(session_id)
            except Exception as e:
                logger.error(f"Session start callback error: {e}")
    
    def on_turn(self, client, event: TurnEvent):
        utter = _get_best_utter(event)
        end_of_turn = _get_attr(event, "end_of_turn", False)
        confidence = _get_attr(event, "confidence", None)

        if not utter:
            return

        logger.info(
            "Turn processed",
            extra={
                "event_type": "turn_processed",
                "session_id": self.session_id,
                "utterance_length": len(utter),
                "word_count": len(utter.split()),
                "is_final": end_of_turn,
                "confidence": confidence
            }
        )

        terminal_processing(utter)
        self.telemetry.track_turn(utter, end_of_turn, confidence)

        if not end_of_turn:
            # ✅ CHANGED: Use safe loop call
            self._safe_loop_call(lambda: asyncio.create_task(self.partial_queue.put(utter)))
            if self.on_partial_callback:
                try:
                    self.on_partial_callback(utter)
                except Exception as e:
                    logger.error(f"Partial callback error: {e}")
            return

        if end_of_turn:
            terminal_final_capture(utter)
            # ✅ CHANGED: Use safe loop calls
            self._safe_loop_call(lambda: asyncio.create_task(self.final_queue.put(utter)))
            self._safe_loop_call(self.completion_event.set)
    
    def on_terminated(self, client, event: TerminationEvent):
        logger.info(
            "STT session terminated",
            extra={
                "event_type": "session_terminated",
                "session_id": self.session_id
            }
        )
        # ✅ CHANGED: Use safe loop call
        self._safe_loop_call(self.completion_event.set)
    
    def on_error(self, client, error: StreamingError):
        """✅ FIXED: Safe error handling prevents RuntimeError during shutdown"""
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.error(
            "STT streaming error occurred",
            extra={
                "event_type": "streaming_error",
                "session_id": self.session_id,
                "error_type": error_type,
                "error_message": error_message
            }
        )
        
        self.telemetry.track_error(error_type, error_message, self.session_id)
        self.error_occurred = True
        
        # ✅ FIXED: Safe loop calls prevent RuntimeError
        self._safe_loop_call(self.error_event.set)
        self._safe_loop_call(self.completion_event.set)

# -------------------------------
# ✅ Fixed Streaming STT Service (Fresh Connections Only)
# -------------------------------
class StreamingSTTService:
    """
    ✅ FIXED: Fresh connections per conversation prevent audio rate limiting
    No connection pooling to avoid "Audio Transmission Rate Exceeded" errors
    """
    
    def __init__(self, api_key: str, default_timeout: float = 30.0, vad_threshold: float = 0.5):
        self.api_key = api_key
        self.default_timeout = default_timeout
        self.vad_threshold = vad_threshold
        
        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None
        self.on_session_start: Optional[Callable[[str], None]] = None
        
        self.sample_rate = 16000
        self.service_telemetry = STTTelemetry()
        
        logger.info(
            "Streaming STT service initialized with fresh connections",
            extra={
                "event_type": "service_init",
                "service_type": "streaming_fresh_connections",
                "connection_pooling": False,  # ✅ CHANGED: Disabled to prevent rate issues
                "streaming_mode": "standard_reliable",
                "default_timeout": default_timeout,
                "vad_threshold": vad_threshold,
                "sample_rate": self.sample_rate,
                "fix_applied": "audio_rate_limit_prevention"
            }
        )
        
    async def capture_speech(self, timeout: float = None) -> str | None:
        """
        ✅ FIXED: Fresh connection per conversation prevents rate limiting and RuntimeError
        """
        if timeout is None:
            timeout = self.default_timeout
            
        capture_start_time = time.time()
        terminal_listening()
        
        loop = asyncio.get_running_loop()
        handler = STTCallbackHandler(loop)
        
        handler.on_partial_callback = self.on_partial
        handler.on_final_callback = self.on_final
        handler.on_session_start_callback = self.on_session_start
        
        logger.info(
            "Starting speech capture with fresh connection",
            extra={
                "event_type": "capture_start",
                "timeout_seconds": timeout,
                "capture_id": f"capture_{int(capture_start_time)}",
                "connection_type": "fresh",
                "rate_limit_prevention": True
            }
        )
        
        # ✅ CHANGED: Fresh client creation every time (prevents rate accumulation)
        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
                open_timeout=30
            )
        )
        
        connection_params = StreamingParameters(
            sample_rate=self.sample_rate,
            format_turns=True,
            speech_model="universal-streaming-english",
            vad_threshold=self.vad_threshold
        )
        
        final_result = None
        
        try:
            # ✅ CHANGED: Fresh connection establishment every conversation
            logger.info(
                "Establishing fresh connection",
                extra={
                    "event_type": "fresh_connection_establish",
                    "prevents_rate_limit": True
                }
            )
            
            client.connect(connection_params)
            
            # Register callbacks
            client.on(StreamingEvents.Begin, handler.on_begin)
            client.on(StreamingEvents.Turn, handler.on_turn)
            client.on(StreamingEvents.Termination, handler.on_terminated) 
            client.on(StreamingEvents.Error, handler.on_error)

            stream_task = None
            partial_task = None
            
            try:
                async def run_standard_stream():
                    try:
                        logger.info("Starting standard streaming with fresh connection")
                        mic_stream = aai.extras.MicrophoneStream(sample_rate=self.sample_rate)
                        await asyncio.to_thread(client.stream, mic_stream)
                        logger.info("Standard streaming completed successfully")
                    except Exception as e:
                        logger.error(f"Standard streaming error: {e}")
                        handler._safe_loop_call(handler.error_event.set)
                        handler._safe_loop_call(handler.completion_event.set)
                
                stream_task = asyncio.create_task(run_standard_stream())
                partial_task = asyncio.create_task(self._process_partial_results(handler.partial_queue, handler.session_id))
                
                try:
                    await asyncio.wait_for(handler.completion_event.wait(), timeout=timeout)
                    
                    if not handler.final_queue.empty():
                        final_result = await handler.final_queue.get()
                        
                    logger.info(
                        "Final result captured with fresh connection",
                        extra={
                            "event_type": "final_result_captured",
                            "session_id": handler.session_id,
                            "result_length": len(final_result) if final_result else 0,
                            "word_count": len(final_result.split()) if final_result else 0,
                            "capture_duration_ms": round((time.time() - capture_start_time) * 1000, 2),
                            "success": True,
                            "connection_type": "fresh",
                            "rate_limit_prevented": True
                        }
                    )
                        
                    if final_result and self.on_final:
                        try:
                            self.on_final(final_result)
                        except Exception as e:
                            logger.error(f"Final callback error: {e}")
                            
                except asyncio.TimeoutError:
                    logger.warning(
                        "STT capture timed out with fresh connection",
                        extra={
                            "event_type": "capture_timeout",
                            "session_id": handler.session_id,
                            "timeout_seconds": timeout,
                            "connection_type": "fresh"
                        }
                    )
                    
            except Exception as e:
                logger.error(
                    "STT capture failed with fresh connection",
                    extra={
                        "event_type": "capture_failed",
                        "session_id": handler.session_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "connection_type": "fresh"
                    }
                )
            finally:
                # Cancel tasks safely
                for task_name, task in [("stream", stream_task), ("partial", partial_task)]:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            logger.info(f"{task_name} task cancelled successfully")
                        except Exception as e:
                            logger.error(f"Error cancelling {task_name} task: {e}")
                
        except Exception as e:
            logger.error(f"Fresh connection error: {e}")
            raise
            
        finally:
            # ✅ CRITICAL: Always disconnect to prevent rate accumulation
            try:
                client.disconnect(terminate=True)
                logger.info(
                    "Fresh connection properly disconnected",
                    extra={
                        "event_type": "fresh_connection_disconnected",
                        "prevents_rate_accumulation": True
                    }
                )
            except Exception as e:
                logger.error(f"Error disconnecting fresh connection: {e}")
            
            # Track telemetry
            handler.telemetry.track_completion(
                final_result=final_result,
                success=final_result is not None and not handler.error_occurred
            )

        return final_result
    
    async def _process_partial_results(self, partial_queue: asyncio.Queue, session_id: str = None):
        partial_count = 0
        try:
            while True:
                try:
                    partial = await asyncio.wait_for(partial_queue.get(), timeout=0.1)
                    partial_count += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"Partial processor cancelled after {partial_count} partials")

# -------------------------------
# ✅ Service Instances
# -------------------------------
streaming_stt_service = StreamingSTTService(
    api_key=API_KEY,
    default_timeout=DEFAULT_STT_TIMEOUT,
    vad_threshold=VAD_THRESHOLD
)

def on_partial_speech(text: str):
    logger.debug(f"Partial speech: {text[:30]}...")

def on_final_speech(text: str):
    logger.info(f"Final speech: {text}")

def on_session_start(session_id: str):
    logger.info(f"Session started: {session_id}")

streaming_stt_service.on_partial = on_partial_speech
streaming_stt_service.on_final = on_final_speech
streaming_stt_service.on_session_start = on_session_start

# -------------------------------
# ✅ LangGraph Integration
# -------------------------------
async def stt_service(state: VoiceState) -> dict:
    request_id = f"request_{int(time.time())}"
    
    logger.info(
        "STT service request started with fresh connection",
        extra={
            "event_type": "stt_request_start",
            "request_id": request_id,
            "connection_type": "fresh",
            "rate_limit_prevention": True
        }
    )
    
    try:
        utter_final = await streaming_stt_service.capture_speech()
        
        if not utter_final:
            logger.warning(
                "No speech captured - timeout occurred",
                extra={
                    "event_type": "stt_timeout",
                    "request_id": request_id
                }
            )
            return {
                "status": "error",
                "error": {
                    "type": "timeout",
                    "message": "No speech detected. Please try again.",
                    "recoverable": True,
                    "request_id": request_id
                }
            }

        current_messages = getattr(state, "messages", []) or []
        updated_messages = current_messages + [HumanMessage(content=utter_final)]
        
        logger.info(
            "STT service request completed successfully",
            extra={
                "event_type": "stt_request_success",
                "request_id": request_id,
                "final_utterance": utter_final,
                "utterance_length": len(utter_final),
                "word_count": len(utter_final.split()),
                "total_messages": len(updated_messages),
                "connection_type": "fresh"
            }
        )
        
        return {
            "messages": updated_messages,
            "status": "success",
            "retry_count": 0,
            "request_id": request_id
        }
        
    except ConnectionError as e:
        logger.error(
            "STT connection error occurred",
            extra={
                "event_type": "stt_connection_error",
                "request_id": request_id,
                "error_message": str(e)
            }
        )
        return {
            "status": "error",
            "error": {
                "type": "connection",
                "message": "Internet connection lost. Please try again.",
                "recoverable": True,
                "original_error": str(e),
                "request_id": request_id
            }
        }
        
    except PermissionError as e:
        logger.error(
            "STT permission error occurred",
            extra={
                "event_type": "stt_permission_error",
                "request_id": request_id,
                "error_message": str(e)
            }
        )
        return {
            "status": "error", 
            "error": {
                "type": "permission",
                "message": "Microphone access denied. Please check your settings.",
                "recoverable": False,
                "original_error": str(e),
                "request_id": request_id
            }
        }
        
    except Exception as e:
        logger.error(
            "STT service unexpected error",
            extra={
                "event_type": "stt_unexpected_error",
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        return {
            "status": "error",
            "error": {
                "type": "unknown",
                "message": f"Unexpected error: {str(e)}",
                "recoverable": False,
                "original_error": str(e),
                "request_id": request_id
            }
        }

# ✅ Log initialization completion
logger.info(
    "STT service module initialized with critical fixes",
    extra={
        "event_type": "module_init_complete",
        "streaming_service": "ready",
        "structured_logging": "file_only",
        "telemetry": "enabled",
        "log_file": "stt_service.log",
        "critical_fixes": {
            "connection_pooling": False,  # Disabled to prevent rate issues
            "fresh_connections": True,   # Prevents audio accumulation
            "safe_event_loop": True,     # Prevents RuntimeError
            "proper_cleanup": True,      # Always disconnect
            "rate_limit_prevention": True
        }
    }
)