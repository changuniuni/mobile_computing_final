package com.example.arvocab

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.ImageReader
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.speech.RecognizerIntent
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var cameraPreview: TextureView
    private lateinit var objectLabelOverlay: TextView
    private lateinit var chatContainer: CardView

    // ─── Conformer TFLite ASR ─────────────────────────────────
    private var asrInterpreter: Interpreter? = null
    private var isRecording = AtomicBoolean(false)
    private var audioRecord: AudioRecord? = null
    private var audioBuffer: ShortArray? = null
    private var recordingThread: Thread? = null
    private val sampleRate = 16000
    private var isAsrAvailable = false

    // 채팅 관련 UI 요소
    private lateinit var messageInput: EditText
    private lateinit var sendButton: ImageButton
    private lateinit var voiceButton: ImageButton
    private lateinit var ttsButton: ImageButton
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var llmStatusTextView: TextView

    // 새로고침 버튼
    private lateinit var refreshButton: ImageButton

    // 메모리 최적화를 위한 모델 관리
    private var objectRecognizer: ObjectRecognizer? = null
    private lateinit var translator: TranslationHelper
    private var llm: ConversationLLM? = null
    private var isLLMLoading = AtomicBoolean(false)

    private var currentLabel: String? = null
    private val isProcessing = AtomicBoolean(false)

    // 인식 일시 중지 플래그
    private val isPaused = AtomicBoolean(false)

    private val minProcessIntervalMs = 1000L
    private var lastProcessAt = 0L
    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private lateinit var previewSize: Size
    private lateinit var imageReader: ImageReader
    private lateinit var previewRequestBuilder: CaptureRequest.Builder

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private val cameraOpenCloseLock = Semaphore(1)

    // Speech Recognizer Launcher
    private val speechRecognizerLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val spokenText = result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            if (!spokenText.isNullOrEmpty()) {
                val recognizedText = spokenText[0]
                Log.d(TAG, "Speech recognition result: '$recognizedText'")
                messageInput.setText(recognizedText)
                sendMessage(recognizedText)
                messageInput.text.clear()
            }
        } else {
            Log.d(TAG, "Speech recognition failed or cancelled")
        }
    }
    
    // Permission Launcher for Audio
    private val requestAudioPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
        if (isGranted) {
            startSpeechToText()
        } else {
            Toast.makeText(this, "Microphone permission is required.", Toast.LENGTH_SHORT).show()
        }
    }

    // TTS 관련
    private var audioTrack: AudioTrack? = null
    private var lastBotMessage: String = ""

    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        private const val RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1002
        private const val PREVIEW_WIDTH = 640  // 원하는 프리뷰 너비
        private const val PREVIEW_HEIGHT = 480 // 원하는 프리뷰 높이
    }

    /**
     * Assets 폴더에서 .tflite 파일을 MappedByteBuffer로 읽어오는 헬퍼
     */
    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fd = assets.openFd(filename)
        FileInputStream(fd.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // ────────── 1) Conformer TFLite Interpreter 초기화 ──────────
        try {
            asrInterpreter = Interpreter(loadModelFile("conformer_librispeech.tflite"),
                Interpreter.Options().apply {
                    setNumThreads(4)
                    // GPU Delegate 사용 시 아래 주석 해제 후 import
                    // addDelegate(GpuDelegate())
                })
            isAsrAvailable = true
            Log.i(TAG, "ASR model initialized successfully")
        } catch (e: Exception) {
            Log.w(TAG, "ASR model not available: ${e.message}")
            isAsrAvailable = false
            // ASR 없이도 앱은 정상 작동
        }

        // ────────── 2) UI 바인딩 ──────────
        cameraPreview = findViewById(R.id.cameraPreview)
        objectLabelOverlay = findViewById(R.id.objectLabelOverlay) // AR 라벨 오버레이
        chatContainer = findViewById(R.id.chatContainer)

        messageInput = findViewById(R.id.messageInput)
        sendButton = findViewById(R.id.sendButton)
        voiceButton = findViewById(R.id.voiceButton)
        ttsButton = findViewById(R.id.ttsButton)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        refreshButton = findViewById(R.id.refreshButton)
        llmStatusTextView = findViewById(R.id.llmStatusTextView)

        chatAdapter = ChatAdapter()
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
        chatRecyclerView.adapter = chatAdapter

        // 메모리 최적화: ObjectRecognizer만 먼저 초기화
        objectRecognizer = ObjectRecognizer(this)
        translator = TranslationHelper(this, "en", "ko")
        // ConversationLLM은 지연 로딩으로 변경
        Log.i(TAG, "ConversationLLM will be loaded after object recognition")

        // ────────── 3) sendButton 클릭 리스너 ──────────
        sendButton.setOnClickListener {
            val message = messageInput.text.toString().trim()
            if (message.isNotEmpty()) {
                sendMessage(message)
                messageInput.text.clear()
            }
        }

        // ────────── 4) voiceButton(마이크) 클릭 리스너 ──────────
        voiceButton.setOnClickListener {
            when {
                ContextCompat.checkSelfPermission(
                    this, Manifest.permission.RECORD_AUDIO
                ) == PackageManager.PERMISSION_GRANTED -> {
                    startSpeechToText()
                }
                else -> {
                    requestAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
            }
        }

        // ────────── 5) ttsButton(음성 출력) 클릭 리스너 ──────────
        ttsButton.setOnClickListener {
            if (lastBotMessage.isNotEmpty()) {
                playTextToSpeech(lastBotMessage)
            } else {
                Toast.makeText(this, "No message to play", Toast.LENGTH_SHORT).show()
            }
        }

        // ────────── 6) 새로고침 버튼 클릭 리스너 ──────────
        refreshButton.setOnClickListener {
            // 메모리 최적화를 위해 ObjectRecognizer 재초기화
            releaseObjectRecognizer()
            objectRecognizer = ObjectRecognizer(this)
            
            isPaused.set(false)
            currentLabel = null
            objectLabelOverlay.visibility = View.GONE // AR 라벨 숨김
            refreshButton.visibility = View.GONE // 새로고침 버튼도 숨김

            // 카메라 프리뷰 재시작
            resumeCameraPreview()

            chatAdapter.addMessage("OK, I'm ready for a new object! Point the camera at something.", false)
            chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)

            Toast.makeText(this, "Recognition Resumed", Toast.LENGTH_SHORT).show()
        }
    }

    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (!isPaused.get()) {
                openCamera(width, height)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
    }

    private val imageAvailableListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener

        if (isPaused.get() || objectRecognizer == null) {
            image.close()
            return@OnImageAvailableListener
        }

        val now = SystemClock.uptimeMillis()
        if (now - lastProcessAt < minProcessIntervalMs || !isProcessing.compareAndSet(false, true)) {
            image.close()
            return@OnImageAvailableListener
        }
        lastProcessAt = now

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val bitmap = objectRecognizer?.yuvToBitmap(image)
                image.close()

                if (bitmap == null) {
                    Log.w(TAG, "ObjectRecognizer is null, skipping recognition")
                    return@launch
                }

                val (recognizedLabel, score) = objectRecognizer!!.recognize(bitmap)
                bitmap.recycle()

                if (recognizedLabel.isNotBlank() && recognizedLabel != currentLabel && score > 0.1) {
                    currentLabel = recognizedLabel
                    isPaused.set(true)

                    // 메모리 최적화: 객체 인식 성공 후 ObjectRecognizer 해제하고 LLM 로드
                    releaseObjectRecognizer()
                    loadLLMIfNeeded()

                    launch(Dispatchers.Main) {
                        // 카메라 프리뷰 정지
                        pauseCameraPreview()
                        
                        chatContainer.visibility = View.VISIBLE
                        refreshButton.visibility = View.VISIBLE

                        if (recognizedLabel != "Unknown") {
                            lifecycleScope.launch(Dispatchers.IO) {
                                try {
                                    val translated = translator.translate(recognizedLabel)
                                    launch(Dispatchers.Main) {
                                        // AR 라벨 오버레이에 영어 원본 단어 표시
                                        objectLabelOverlay.text = recognizedLabel
                                        objectLabelOverlay.visibility = View.VISIBLE
                                        
                                        chatAdapter.addMessage("It looks like a $recognizedLabel! In Korean, that's '$translated'.", false)
                                        chatAdapter.addMessage("What would you like to know about it?", false)
                                        chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                                    }
                                } catch (e: Exception) {
                                    Log.e(TAG, "Translation error: ${e.message}", e)
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during recognition: ${e.message}", e)
            } finally {
                isProcessing.set(false)
            }
        }
    }

    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraOpenCloseLock.release()
            cameraDevice = camera
            createCameraPreviewSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            cameraOpenCloseLock.release()
            camera.close()
            cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
            cameraOpenCloseLock.release()
            camera.close()
            cameraDevice = null
            Log.e(TAG, "CameraDevice Error: $error")
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
        if (!isPaused.get()) {
            if (cameraPreview.isAvailable) {
                openCamera(cameraPreview.width, cameraPreview.height)
            } else {
                cameraPreview.surfaceTextureListener = surfaceTextureListener
            }
        }
    }

    override fun onPause() {
        if (!isPaused.get()) {
            closeCamera()
            stopBackgroundThread()
        }
        super.onPause()
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background thread", e)
        }
    }

    private fun checkCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
            // 사용자에게 권한이 필요한 이유를 설명 (필요 시)
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun openCamera(width: Int, height: Int) {
        if (!checkCameraPermission()) {
            requestCameraPermission()
            return
        }
        val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            val cameraId = cameraManager.cameraIdList.firstOrNull {
                cameraManager.getCameraCharacteristics(it)
                    .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: throw IllegalStateException("No back-facing camera found")

            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

            previewSize = map.getOutputSizes(ImageFormat.YUV_420_888)
                .firstOrNull { it.width == PREVIEW_WIDTH && it.height == PREVIEW_HEIGHT }
                ?: map.getOutputSizes(ImageFormat.YUV_420_888).maxByOrNull { it.width * it.height }!!

            imageReader = ImageReader.newInstance(previewSize.width, previewSize.height, ImageFormat.YUV_420_888, 5)
            imageReader.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Time out waiting to lock camera opening.")
            }
            cameraManager.openCamera(cameraId, cameraStateCallback, backgroundHandler)

        } catch (e: CameraAccessException) {
            Log.e(TAG, "Cannot access the camera.", e)
        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while trying to lock camera opening.", e)
        }
    }

    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            imageReader.close()
        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while trying to lock camera closing.", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val texture = cameraPreview.surfaceTexture!!
            texture.setDefaultBufferSize(previewSize.width, previewSize.height)
            val surface = Surface(texture)

            previewRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            previewRequestBuilder.addTarget(surface)
            previewRequestBuilder.addTarget(imageReader.surface)

            cameraDevice?.createCaptureSession(
                listOf(surface, imageReader.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        captureSession = session
                        try {
                            previewRequestBuilder.set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                            )
                            previewRequestBuilder.set(
                                CaptureRequest.CONTROL_AE_MODE,
                                CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
                            )
                            captureSession?.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler)
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, "Failed to start camera preview", e)
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(this@MainActivity, "Failed to configure camera.", Toast.LENGTH_SHORT).show()
                    }
                },
                null
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Error creating camera preview session", e)
        }
    }

    /**
     * 카메라 프리뷰를 일시 정지합니다
     */
    private fun pauseCameraPreview() {
        try {
            captureSession?.stopRepeating()
            Log.d(TAG, "Camera preview paused")
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to pause camera preview", e)
        }
    }

    /**
     * 카메라 프리뷰를 재시작합니다
     */
    private fun resumeCameraPreview() {
        try {
            if (captureSession != null && cameraDevice != null) {
                captureSession?.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler)
                Log.d(TAG, "Camera preview resumed")
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to resume camera preview", e)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RECORD_AUDIO_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startSpeechToText()
            } else {
                Toast.makeText(this, "Microphone permission is required to use voice input.", Toast.LENGTH_SHORT).show()
            }
        }
        // CAMERA 권한 처리도 마찬가지
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (cameraPreview.isAvailable) {
                    openCamera(cameraPreview.width, cameraPreview.height)
                } else {
                    cameraPreview.surfaceTextureListener = surfaceTextureListener
                }
            } else {
                Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
            }
        }
    }


    // ─── 녹음 토글 함수 ───────────────────────────────────────────
    private fun toggleRecording() {
        if (isRecording.get()) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    // ─── AudioRecord 초기화 ─────────────────────────────────────────
    private fun initAudioRecorder() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            throw SecurityException("RECORD_AUDIO permission not granted")
        }
        
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        audioBuffer = ShortArray(minBufferSize)
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBufferSize
        )
    }

    // ─── 녹음 시작 ─────────────────────────────────────────────
    private fun startRecording() {
        try {
            initAudioRecorder()
            audioRecord?.startRecording()
            isRecording.set(true)
            voiceButton.setImageResource(R.drawable.ic_mic)

            val thread = Thread {
                val pcmList = mutableListOf<Short>()
                val buffer = audioBuffer
                val recorder = audioRecord
                
                while (isRecording.get() && buffer != null && recorder != null) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        for (i in 0 until read) {
                            pcmList.add(buffer[i])
                        }
                    }
                }
                val pcmData = pcmList.toShortArray()
                runConformerInferenceRaw(pcmData)
            }
            recordingThread = thread
            thread.start()
        } catch (e: SecurityException) {
            Log.e(TAG, "RECORD_AUDIO permission not granted", e)
            Toast.makeText(this, "Microphone permission is required.", Toast.LENGTH_SHORT).show()
        }
    }

    // ─── 녹음 중지 ─────────────────────────────────────────────
    private fun stopRecording() {
        isRecording.set(false)
        audioRecord?.stop()
        audioRecord?.release()
        voiceButton.setImageResource(R.drawable.ic_mic)
    }

    // ─── Conformer TFLite 추론 (Raw PCM 기반 ASR) ───────────────────
    private fun runConformerInferenceRaw(pcmData: ShortArray) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // 1) PCM(short[]) → FloatArray 정규화
                val floatPCM = FloatArray(pcmData.size) { i ->
                    pcmData[i].toFloat() / Short.MAX_VALUE
                }

                // 2) 입력 텐서 크기 재조정 및 allocateTensors()
                asrInterpreter?.resizeInput(0, intArrayOf(floatPCM.size))
                asrInterpreter?.allocateTensors()

                // 3) 입력 텐서 준비
                // 3-1) raw PCM float 데이터
                val buf0 = ByteBuffer.allocateDirect(4 * floatPCM.size)
                    .order(ByteOrder.nativeOrder())
                for (f in floatPCM) buf0.putFloat(f)
                buf0.rewind()

                // 3-2) int32 scalar 0
                val buf1 = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
                buf1.putInt(0).rewind()

                // 3-3) zeros([1,2,1,320], float32)
                val zeroCount = 1 * 2 * 1 * 320
                val buf2 = ByteBuffer.allocateDirect(4 * zeroCount)
                    .order(ByteOrder.nativeOrder())
                repeat(zeroCount) { buf2.putFloat(0f) }
                buf2.rewind()

                // 4) 입력 배열 구성
                val inputs = arrayOf(buf0, buf1, buf2)

                // 5) 출력 텐서 준비
                val outputTensor = asrInterpreter?.getOutputTensor(0)
                val outShape = outputTensor?.shape()
                val outCount = outShape?.reduce { a, b -> a * b } ?: 0
                val outputBuffer = ByteBuffer.allocateDirect(4 * outCount)
                    .order(ByteOrder.nativeOrder())
                val outputs = mapOf(0 to outputBuffer)

                // 6) 추론 실행
                asrInterpreter?.runForMultipleInputsOutputs(inputs, outputs)

                // 7) 출력 처리
                outputBuffer.rewind()
                val tokenInts = IntArray(outCount)
                for (i in 0 until outCount) {
                    tokenInts[i] = outputBuffer.int
                }

                // 5-1) Greedy CTC 디코딩 (연속 중복 제거 & blank(0) 제거)
                val sb = StringBuilder()
                var prev = -1
                for (u in tokenInts) {
                    if (u != prev && u != 0) {
                        sb.append(Char(u))
                    }
                    prev = u
                }
                val recognizedText = sb.toString()

                // 6) UI 업데이트 및 기존 sendMessage() 호출
                withContext(Dispatchers.Main) {
                    if (recognizedText.isNotBlank()) {
                        messageInput.setText(recognizedText)
                        sendMessage(recognizedText)
                        messageInput.text.clear()
                    } else {
                        Toast.makeText(this@MainActivity, "Could not recognize speech.", Toast.LENGTH_SHORT).show()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Conformer ASR Error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Failed to recognize speech.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // ─── CLIP 기반 객체 인식 & 카메라 설정 (기존 코드 그대로) ───────────────────
    // surfaceTextureListener, imageAvailableListener, openCamera(), createCameraPreviewSession(), closeCamera(), etc.

    // ─── 채팅 메시지 전송 및 LLM 응답 처리 ─────────────────────────────────────
    private fun sendMessage(userMessage: String) {
        // 아직 일시정지되지 않은 경우에만 카메라를 정지시킴
        if (!isPaused.get()) {
            isPaused.set(true)
            pauseCameraPreview()
        }
        refreshButton.visibility = View.VISIBLE // 새로고침 버튼 표시

        // 사용자 메시지를 채팅창에 추가
        chatAdapter.addMessage(userMessage, true)
        chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)

        // LLM이 초기화되지 않았다면 초기화 후 채팅 시작
        if (llm == null) {
            initLlmAndChat(userMessage)
            return
        }

        // LLM이 이미 초기화되었다면 바로 채팅 시작
        lifecycleScope.launch {
            val response = llm?.chat(currentLabel ?: "Unknown", "", userMessage) ?: "Error: LLM not available."
            lastBotMessage = response // TTS를 위해 마지막 봇 메시지 저장
            chatAdapter.addMessage(response, false)
            chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
            
            // 번역 요청인지 확인하고 화면 중앙에 번역된 단어 표시
            checkAndDisplayTranslation(userMessage, response)
        }
    }

    /**
     * 사용자가 처음 채팅을 시도할 때 LLM을 초기화하고 첫 메시지를 전송
     */
    private fun initLlmAndChat(initialMessage: String) {
        if (isLLMLoading.get()) {
            Toast.makeText(this, "Assistant is still starting up...", Toast.LENGTH_SHORT).show()
            return
        }

        isLLMLoading.set(true)
        llmStatusTextView.text = "⏳ Starting assistant..."
        llmStatusTextView.visibility = View.VISIBLE

        lifecycleScope.launch(Dispatchers.IO) {
            llm = ConversationLLM(this@MainActivity)
            
            // UI 업데이트는 Main 스레드에서
            withContext(Dispatchers.Main) {
                isLLMLoading.set(false)
                Toast.makeText(this@MainActivity, "Assistant is ready!", Toast.LENGTH_SHORT).show()
                // 상태 업데이트 핸들러 시작
                handler.post(updateLlmStatusRunnable)

                // 첫 메시지 전송
                lifecycleScope.launch {
                    val response = llm?.chat(currentLabel ?: "Unknown", "", initialMessage) ?: "Error: LLM not available."
                    lastBotMessage = response // TTS를 위해 마지막 봇 메시지 저장
                    chatAdapter.addMessage(response, false)
                    chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                    
                    // 번역 요청인지 확인하고 화면 중앙에 번역된 단어 표시
                    checkAndDisplayTranslation(initialMessage, response)
                }
            }
        }
    }

    private val handler = Handler()
    private val updateLlmStatusRunnable = object : Runnable {
        override fun run() {
            updateLlmStatus()
            handler.postDelayed(this, 3000) // 3초마다 상태 업데이트
        }
    }

    /**
     * LLM의 현재 상태를 UI에 표시
     */
    private fun updateLlmStatus() {
        if (llm != null) {
            llmStatusTextView.text = llm?.getStatus()
            llmStatusTextView.visibility = View.VISIBLE
        } else {
            llmStatusTextView.visibility = View.GONE
        }
    }

    // ObjectRecognizer 메모리 해제
    private fun releaseObjectRecognizer() {
        objectRecognizer = null
        System.gc() // 가비지 컬렉션 강제 실행
        Log.i(TAG, "ObjectRecognizer released for memory optimization")
    }

    // ConversationLLM 지연 로딩 (조용히)
    private fun loadLLMIfNeeded() {
        if (llm == null && !isLLMLoading.get()) {
            isLLMLoading.set(true)
            
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    Log.i(TAG, "Loading ConversationLLM after object recognition...")
                    llm = ConversationLLM(this@MainActivity)
                    
                    // 로딩 완료 후 상태 확인 (로그만)
                    kotlinx.coroutines.delay(1000) // 초기화 시간 여유
                    
                    val status = llm?.getStatus() ?: "Load failed"
                    Log.i(TAG, "ConversationLLM loaded. Status: $status")
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load ConversationLLM: ${e.message}", e)
                } finally {
                    isLLMLoading.set(false)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // 메모리 정리
        releaseObjectRecognizer()
        llm = null
        asrInterpreter?.close()
        audioTrack?.release()
        audioTrack = null
    }

    private fun startSpeechToText() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US)
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak now...")
        }
        try {
            speechRecognizerLauncher.launch(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "Speech recognition is not available.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun checkAndDisplayTranslation(userMessage: String, geminiResponse: String) {
        Log.d(TAG, "Checking translation - User: '$userMessage', Response: '$geminiResponse'")
        
        // 번역 요청 키워드 감지 (더 포괄적으로)
        val translationKeywords = listOf(
            // 영어 패턴
            "in chinese", "in japanese", "in spanish", "in french", "in german", 
            "in italian", "in russian", "in arabic", "in hindi", "in portuguese",
            "chinese", "japanese", "spanish", "french", "german", "italian", "russian",
            "translate", "translation", "how do you say", "what is", "say in",
            "how to say", "what's", "whats", "how do i say",
            
            // 한국어 패턴
            "중국어로", "일본어로", "스페인어로", "프랑스어로", "독일어로", "이탈리아어로",
            "중국어", "일본어", "스페인어", "프랑스어", "독일어", "이탈리아어",
            "번역", "뭐야", "어떻게", "말해", "어떻게 말해", "뭐라고", "뭐라고 해",
            "어떻게 해", "어떻게 말하지", "뭐라고 하지"
        )
        
        val isTranslationRequest = translationKeywords.any { keyword ->
            userMessage.lowercase().contains(keyword.lowercase())
        }
        
        Log.d(TAG, "Is translation request: $isTranslationRequest")
        
        if (isTranslationRequest) {
            // Gemini 응답에서 번역된 단어 추출 시도
            val extractedWord = extractTranslatedWord(geminiResponse)
            Log.d(TAG, "Extracted word: '$extractedWord'")
            
            if (extractedWord.isNotEmpty()) {
                // 화면 중앙에 번역된 단어 표시 (그대로 유지)
                Log.d(TAG, "Displaying translated word: '$extractedWord'")
                objectLabelOverlay.text = extractedWord
                objectLabelOverlay.visibility = View.VISIBLE
            } else {
                Log.d(TAG, "No word extracted, keeping original display")
            }
        }
    }
    
    private fun extractTranslatedWord(response: String): String {
        Log.d(TAG, "Extracting word from response: '$response'")
        
        // 다양한 패턴으로 번역된 단어 추출 시도
        val patterns = listOf(
            "\"([^\"]{1,20})\"",  // 따옴표로 둘러싸인 단어
            "'([^']{1,20})'",     // 작은따옴표로 둘러싸인 단어
            "is\\s+([\\p{L}\\p{M}\\p{N}]{1,20})(?:\\s|\\.|,|!|\\?|$)", // "is 단어" 패턴
            "called\\s+([\\p{L}\\p{M}\\p{N}]{1,20})(?:\\s|\\.|,|!|\\?|$)", // "called 단어" 패턴
            "([\\p{L}\\p{M}\\p{N}]{2,15})(?:\\s|\\.|,|!|\\?|$)", // 일반 단어 패턴
            "：\\s*([\\p{L}\\p{M}\\p{N}]{1,20})", // 콜론 뒤 단어
            ":\\s*([\\p{L}\\p{M}\\p{N}]{1,20})", // 영어 콜론 뒤 단어
            "([\\p{L}\\p{M}\\p{N}]{1,20})\\s*\\(", // 괄호 앞 단어
            "\\*\\*([\\p{L}\\p{M}\\p{N}]{1,20})\\*\\*" // 볼드체 단어
        )
        
        for ((index, pattern) in patterns.withIndex()) {
            Log.d(TAG, "Trying pattern $index: $pattern")
            val regex = Regex(pattern)
            val matches = regex.findAll(response)
            for (match in matches) {
                if (match.groupValues.size > 1) {
                    val word = match.groupValues[1].trim()
                    Log.d(TAG, "Found candidate word: '$word'")
                    
                    // 영어가 아닌 문자가 포함된 경우 반환
                    if (word.length in 1..20 && 
                        !word.matches(Regex("[a-zA-Z\\s]+")) && 
                        !word.contains(" ") &&
                        word != "Unknown") {
                        Log.d(TAG, "Selected word: '$word'")
                        return word
                    }
                }
            }
        }
        
        // 마지막 시도: 응답에서 첫 번째 비영어 단어 찾기
        Log.d(TAG, "Trying fallback method")
        val words = response.split(Regex("[\\s.,!?;:()\\[\\]\"'`*]+"))
        for (word in words) {
            val cleanWord = word.trim()
            Log.d(TAG, "Checking fallback word: '$cleanWord'")
            if (cleanWord.length in 1..20 && 
                !cleanWord.matches(Regex("[a-zA-Z0-9]+")) &&
                cleanWord.isNotBlank() &&
                cleanWord != "Unknown") {
                Log.d(TAG, "Selected fallback word: '$cleanWord'")
                return cleanWord
            }
        }
        
        Log.d(TAG, "No word found")
        return ""
    }

    // 테스트용 함수 - 번역 요청 감지 테스트
    private fun testTranslationDetection(testMessage: String) {
        Log.d(TAG, "=== Testing translation detection ===")
        Log.d(TAG, "Test message: '$testMessage'")
        
        val translationKeywords = listOf(
            "in chinese", "in japanese", "chinese", "japanese", "translate", 
            "중국어로", "일본어로", "중국어", "일본어", "번역", "뭐야", "어떻게"
        )
        
        val isTranslationRequest = translationKeywords.any { keyword ->
            testMessage.lowercase().contains(keyword.lowercase())
        }
        
        Log.d(TAG, "Is translation request: $isTranslationRequest")
        Log.d(TAG, "=== End test ===")
    }

    private fun playTextToSpeech(text: String) {
        if (llm == null) {
            Toast.makeText(this, "Assistant not ready", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            try {
                Toast.makeText(this@MainActivity, "Generating speech...", Toast.LENGTH_SHORT).show()
                
                val audioData = llm?.generateSpeech(text)
                if (audioData != null) {
                    playAudio(audioData)
                } else {
                    Toast.makeText(this@MainActivity, "Failed to generate speech", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "TTS Error: ${e.message}", e)
                Toast.makeText(this@MainActivity, "Speech generation error", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun playAudio(audioData: ByteArray) {
        try {
            // AudioTrack 설정 (24kHz, 16-bit, Mono)
            val sampleRate = 24000
            val channelConfig = AudioFormat.CHANNEL_OUT_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val bufferSize = AudioTrack.getMinBufferSize(sampleRate, channelConfig, audioFormat)

            audioTrack?.release() // 기존 AudioTrack 해제

            audioTrack = AudioTrack(
                AudioManager.STREAM_MUSIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize,
                AudioTrack.MODE_STREAM
            )

            audioTrack?.play()
            
            // 백그라운드에서 오디오 재생
            Thread {
                try {
                    audioTrack?.write(audioData, 0, audioData.size)
                    audioTrack?.stop()
                } catch (e: Exception) {
                    Log.e(TAG, "Audio playback error: ${e.message}", e)
                }
            }.start()

            Toast.makeText(this, "Playing speech...", Toast.LENGTH_SHORT).show()
            
        } catch (e: Exception) {
            Log.e(TAG, "Audio setup error: ${e.message}", e)
            Toast.makeText(this, "Audio playback failed", Toast.LENGTH_SHORT).show()
        }
    }
}
