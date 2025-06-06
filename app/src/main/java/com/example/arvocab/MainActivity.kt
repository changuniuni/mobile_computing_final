package com.example.arvocab

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
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
    private lateinit var objectLabel: TextView
    private lateinit var translationView: TextView
    private lateinit var chatResponse: TextView
    // private lateinit var boundingBox: View // AR 기능 제거로 일단 사용 안 함
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
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var chatAdapter: ChatAdapter

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
        objectLabel = findViewById(R.id.objectLabel)
        translationView = findViewById(R.id.translationView)
        chatResponse = findViewById(R.id.chatResponse)
        // boundingBox = findViewById(R.id.boundingBox) // AR 기능 제거
        chatContainer = findViewById(R.id.chatContainer)

        messageInput = findViewById(R.id.messageInput)
        sendButton = findViewById(R.id.sendButton)
        voiceButton = findViewById(R.id.voiceButton)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        refreshButton = findViewById(R.id.refreshButton)

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
        // 예: 마이크 버튼 클릭 리스너 내부
        voiceButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    RECORD_AUDIO_PERMISSION_REQUEST_CODE
                )
            } else {
                toggleRecording()
            }
        }


        // ────────── 5) 새로고침 버튼 클릭 리스너 ──────────
        refreshButton.setOnClickListener {
            // 메모리 최적화를 위해 ObjectRecognizer 재초기화
            releaseObjectRecognizer()
            objectRecognizer = ObjectRecognizer(this)
            
            isPaused.set(false)
            currentLabel = null
            objectLabel.text = ""
            translationView.text = ""
            chatResponse.text = ""

            chatAdapter.addMessage("인식이 재개되었습니다. 새로운 물체를 인식해보세요.", false)
            chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)

            Toast.makeText(this, "인식이 재개되었습니다", Toast.LENGTH_SHORT).show()
        }
    }

    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            openCamera(width, height)
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
                        val currentTime = timeFormat.format(Date())

                        objectLabel.text = "[$currentTime]It is $recognizedLabel."
                        translationView.text = ""
                        chatResponse.text = ""
                        chatContainer.visibility = if (recognizedLabel != "Unknown") View.VISIBLE else View.GONE
                        refreshButton.visibility = View.VISIBLE

                        if (recognizedLabel != "Unknown") {
                            lifecycleScope.launch(Dispatchers.IO) {
                                try {
                                    val translated = translator.translate(recognizedLabel)
                                    launch(Dispatchers.Main) {
                                        translationView.text = translated
                                        chatAdapter.addMessage("인식된 물체: $recognizedLabel", false)
                                        chatAdapter.addMessage("한국어 번역: $translated", false)
                                        val infoMessage = "이 물체에 대해 더 알고 싶은 것이 있으면 질문해주세요."
                                        chatAdapter.addMessage(infoMessage, false)
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
        if (cameraPreview.isAvailable) {
            openCamera(cameraPreview.width, cameraPreview.height)
        } else {
            cameraPreview.surfaceTextureListener = surfaceTextureListener
        }
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
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
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
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

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RECORD_AUDIO_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                toggleRecording()
            } else {
                Toast.makeText(this, "마이크 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
            }
        }
        // CAMERA 권한 처리도 마찬가지
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera(cameraPreview.width, cameraPreview.height)
            } else {
                Toast.makeText(this, "카메라 권한이 필요합니다.", Toast.LENGTH_LONG).show()
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
            Toast.makeText(this, "마이크 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
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
                    } else {
                        Toast.makeText(this@MainActivity, "음성 인식 결과가 없습니다.", Toast.LENGTH_SHORT).show()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Conformer ASR 오류: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "음성 인식에 실패했습니다.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // ─── CLIP 기반 객체 인식 & 카메라 설정 (기존 코드 그대로) ───────────────────
    // surfaceTextureListener, imageAvailableListener, openCamera(), createCameraPreviewSession(), closeCamera(), etc.

    // ─── 채팅 메시지 전송 및 LLM 응답 처리 ─────────────────────────────────────
    private fun sendMessage(message: String) {
        chatAdapter.addMessage(message, true)
        chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)

        val original = currentLabel ?: ""
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val translated = if (original.isNotEmpty() && original != "Unknown") {
                    translator.translate(original)
                } else {
                    ""
                }
                
                // LLM이 로드될 때까지 대기
                var attempts = 0
                while (llm == null && attempts < 100) { // 최대 10초 대기
                    kotlinx.coroutines.delay(100)
                    attempts++
                    
                    // 진행 상황 업데이트
                    if (attempts % 20 == 0) { // 2초마다
                        withContext(Dispatchers.Main) {
                            chatAdapter.addMessage("⏳ AI 모델 로딩 중... (${attempts/10}초 경과)", false)
                            chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                        }
                    }
                }
                
                val response = if (llm != null) {
                    // 실제 모델 사용 상태 확인
                    val isUsingRealModel = llm!!.isReady()
                    Log.i(TAG, "Using real T5 model: $isUsingRealModel")
                    
                    llm!!.chat(original, translated, message)
                } else {
                    "죄송합니다. T5 AI 모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도해주세요."
                }
                
                withContext(Dispatchers.Main) {
                    chatAdapter.addMessage(response, false)
                    chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                    
                    // 실제 모델 사용 여부 표시
                    val modelStatus = llm?.let { 
                        if (it.isReady()) "🟢 실제 T5 모델 사용" else "🟡 스마트 폴백 모드"
                    } ?: "🔴 모델 미로드"
                    
                    Log.i(TAG, "Model status: $modelStatus")
                }
            } catch (e: Exception) {
                Log.e(TAG, "sendMessage 에러: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    chatAdapter.addMessage("Sorry, I couldn't process your message: ${e.message}", false)
                    chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                }
            }
        }
    }

    // ObjectRecognizer 메모리 해제
    private fun releaseObjectRecognizer() {
        objectRecognizer = null
        System.gc() // 가비지 컬렉션 강제 실행
        Log.i(TAG, "ObjectRecognizer released for memory optimization")
    }

    // ConversationLLM 지연 로딩
    private fun loadLLMIfNeeded() {
        if (llm == null && !isLLMLoading.get()) {
            isLLMLoading.set(true)
            
            // 로딩 시작 알림
            lifecycleScope.launch(Dispatchers.Main) {
                chatAdapter.addMessage("🤖 Loagind model...", false)
                chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
            }
            
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    Log.i(TAG, "Loading ConversationLLM after object recognition...")
                    llm = ConversationLLM(this@MainActivity)
                    
                    // 로딩 완료 후 상태 확인
                    kotlinx.coroutines.delay(1000) // 초기화 시간 여유
                    
                    withContext(Dispatchers.Main) {
                        val status = llm?.getDetailedStatus() ?: "❌ 로딩 실패"
                        val isRealModel = llm?.isReady() ?: false
                        
                        if (isRealModel) {
                            chatAdapter.addMessage("✅ Model is successfully loaded!", false)
                        } else {
                            chatAdapter.addMessage("⚠️ T5 모델 로딩에 실패했지만, 스마트 폴백 모드로 작동합니다.\n$status", false)
                        }
                        chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                    }
                    
                    Log.i(TAG, "ConversationLLM loaded successfully")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load ConversationLLM: ${e.message}", e)
                    withContext(Dispatchers.Main) {
                        chatAdapter.addMessage("❌ T5 모델 로딩에 실패했습니다: ${e.message}", false)
                        chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                    }
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
    }
}
