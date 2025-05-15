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

class MainActivity : AppCompatActivity() {

    private lateinit var cameraPreview: TextureView
    private lateinit var objectLabel: TextView
    private lateinit var translationView: TextView
    private lateinit var chatResponse: TextView
    // private lateinit var boundingBox: View // AR 기능 제거로 일단 사용 안 함
    private lateinit var chatContainer: CardView

    private lateinit var objectRecognizer: ObjectRecognizer
    private lateinit var translator: TranslationHelper
    private lateinit var llm: ConversationLLM

    private var currentLabel: String? = null
    private val isProcessing = AtomicBoolean(false)

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

        objectRecognizer = ObjectRecognizer(this)
        translator = TranslationHelper(this, "en", "ko") // 필요에 따라 언어 코드 수정
        llm = ConversationLLM(this)
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
                    launch(Dispatchers.Main) {
                        val currentTime = timeFormat.format(Date())
                        objectLabel.text = "[$currentTime] $recognizedLabel (${String.format("%.2f", score)})"
                        translationView.text = ""
                        chatResponse.text = ""
                        chatContainer.visibility = if (recognizedLabel != "Unknown") View.VISIBLE else View.GONE

                        // 번역 및 LLM 호출 (기존 로직 유지 또는 수정)
                        // translator.translate(recognizedLabel) { translated ->
                        //    translationView.text = translated
                        //    llm.getResponse(translated ?: recognizedLabel) { llmResponse ->
                        //        chatResponse.text = llmResponse
                        //    }
                        // }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during recognition: ${e.message}", e)
            } finally {
                isProcessing.set(false)
                // image.close()는 이미 위에서 호출했으므로 여기서는 제거
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
}