package com.example.arvocab

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class MainActivity : AppCompatActivity() {

    private lateinit var cameraPreview: TextureView
    private lateinit var objectLabel: TextView
    private lateinit var translationView: TextView
    private lateinit var chatResponse: TextView
    // private lateinit var boundingBox: View // AR 기능 제거로 일단 사용 안 함
    private lateinit var chatContainer: CardView
    
    // 채팅 관련 UI 요소 추가
    private lateinit var messageInput: EditText
    private lateinit var sendButton: ImageButton
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var chatAdapter: ChatAdapter

    // 새로고침 버튼 추가
    private lateinit var refreshButton: ImageButton

    private lateinit var objectRecognizer: ObjectRecognizer
    private lateinit var translator: TranslationHelper
    private lateinit var llm: ConversationLLM

    private var currentLabel: String? = null
    private val isProcessing = AtomicBoolean(false)

    // 인식 일시 중지 플래그 추가
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
        private const val PREVIEW_WIDTH = 640 // 원하는 프리뷰 너비
        private const val PREVIEW_HEIGHT = 480 // 원하는 프리뷰 높이
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraPreview = findViewById(R.id.cameraPreview)
        objectLabel = findViewById(R.id.objectLabel)
        translationView = findViewById(R.id.translationView)
        chatResponse = findViewById(R.id.chatResponse)
        // boundingBox = findViewById(R.id.boundingBox) // AR 기능 제거
        chatContainer = findViewById(R.id.chatContainer)
        
        // 채팅 UI 요소 초기화
        messageInput = findViewById(R.id.messageInput)
        sendButton = findViewById(R.id.sendButton)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        
        // 새로고침 버튼 초기화
        refreshButton = findViewById(R.id.refreshButton)
        
        // RecyclerView 및 어댑터 설정
        chatAdapter = ChatAdapter()
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
        chatRecyclerView.adapter = chatAdapter

        objectRecognizer = ObjectRecognizer(this)
        translator = TranslationHelper(this, "en", "ko")
        llm = ConversationLLM(this)
        
        // 메시지 전송 버튼 클릭 리스너 설정
        sendButton.setOnClickListener {
            val message = messageInput.text.toString().trim()
            if (message.isNotEmpty()) {
                sendMessage(message)
                messageInput.text.clear()
            }
        }
        
        // 새로고침 버튼 클릭 리스너 설정
        refreshButton.setOnClickListener {
            // 인식 일시 중지 해제
            isPaused.set(false)
            // 현재 라벨 초기화
            currentLabel = null
            // UI 초기화
            objectLabel.text = ""
            translationView.text = ""
            chatResponse.text = ""
            
            // 채팅 어댑터에 시스템 메시지 추가
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

        // 인식이 일시 중지된 상태이면 이미지만 닫고 처리하지 않음
        if (isPaused.get()) {
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
                val bitmap = objectRecognizer.yuvToBitmap(image)
                image.close() // 비트맵 변환 후 즉시 이미지 해제
                
                val (recognizedLabel, score) = objectRecognizer.recognize(bitmap)
                bitmap.recycle() // 비트맵 사용 후 메모리 해제

                if (recognizedLabel.isNotBlank() && recognizedLabel != currentLabel && score > 0.1) {
                    currentLabel = recognizedLabel
                    
                    // 물체가 인식되면 인식 일시 중지
                    isPaused.set(true)
                    
                    launch(Dispatchers.Main) {
                        val currentTime = timeFormat.format(Date())
                        
                        // 기존 UI 업데이트 (필요시 숨김 처리 가능)
                        objectLabel.text = "[$currentTime]It is $recognizedLabel."
                        translationView.text = ""
                        chatResponse.text = ""
                        chatContainer.visibility = if (recognizedLabel != "Unknown") View.VISIBLE else View.GONE
                        
                        // 새로고침 버튼 표시
                        refreshButton.visibility = View.VISIBLE

                        // 번역 처리 및 채팅 메시지 추가
                        if (recognizedLabel != "Unknown") {
                            lifecycleScope.launch(Dispatchers.IO) {
                                try {
                                    val translated = translator.translate(recognizedLabel)
                                    
                                    launch(Dispatchers.Main) {
                                        // 기존 UI 업데이트
                                        translationView.text = translated
                                        
                                        // 채팅 메시지로 추가
                                        chatAdapter.addMessage("인식된 물체: $recognizedLabel", false)
                                        chatAdapter.addMessage("한국어 번역: $translated", false)
                                        
                                        // 추가 정보 메시지
                                        val infoMessage = "이 물체에 대해 더 알고 싶은 것이 있으면 질문해주세요."
                                        chatAdapter.addMessage(infoMessage, false)
                                        
                                        // 스크롤 최신 메시지로 이동
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
            
            // YUV_420_888 포맷을 지원하는 가장 적절한 크기 선택 (또는 고정 크기 사용)
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
            imageReader?.close()
            // imageReader = null // imageReader는 lateinit이므로 null 할당 불가, close()만 호출
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
            previewRequestBuilder.addTarget(imageReader.surface) // ImageReader의 surface도 추가

            cameraDevice?.createCaptureSession(
                listOf(surface, imageReader.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        captureSession = session
                        try {
                            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                            previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH)
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

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (cameraPreview.isAvailable) {
                    openCamera(cameraPreview.width, cameraPreview.height)
                } else {
                    cameraPreview.surfaceTextureListener = surfaceTextureListener
                }
            } else {
                Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
                // finish() // 권한 없으면 앱 종료 또는 다른 처리
            }
        }
    }

    // 메시지 전송 및 LLM 응답 처리 함수
    private fun sendMessage(message: String) {
        // 사용자 메시지 추가
        chatAdapter.addMessage(message, true)
        
        // 스크롤을 최신 메시지로 이동
        chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
        
        // 현재 인식된 객체가 있는 경우
        val original = currentLabel ?: ""
        
        // 코루틴으로 번역 및 LLM 응답 처리
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // 인식된 객체가 있으면 번역
                val translated = if (original.isNotEmpty() && original != "Unknown") {
                    translator.translate(original)
                } else {
                    ""
                }
                
                // LLM에 메시지 전송하고 응답 받기
                val response = llm.chat(original, translated, message)
                
                // UI 스레드에서 응답 표시
                launch(Dispatchers.Main) {
                    chatAdapter.addMessage(response, false)
                    chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing message: ${e.message}", e)
                
                // 오류 발생 시 UI 스레드에서 오류 메시지 표시
                launch(Dispatchers.Main) {
                    chatAdapter.addMessage("Sorry, I couldn't process your message.", false)
                    chatRecyclerView.scrollToPosition(chatAdapter.itemCount - 1)
                }
            }
        }
    }
}