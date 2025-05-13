package com.example.arvocab

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Point
import android.media.Image
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.cardview.widget.CardView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.exceptions.DeadlineExceededException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.ResourceExhaustedException
import com.google.ar.sceneform.math.Vector3
import com.google.ar.sceneform.ux.ArFragment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import com.google.ar.core.Config
import java.util.concurrent.atomic.AtomicBoolean


class MainActivity : AppCompatActivity() {

    private lateinit var arFragment: ArFragment
    private lateinit var objectLabel: TextView
    private lateinit var translationView: TextView
    private lateinit var chatResponse: TextView
    private lateinit var boundingBox: View
    private lateinit var infoCard: CardView
    private lateinit var objectRecognizer: ObjectRecognizer
    private lateinit var translator: TranslationHelper
    private lateinit var llm: ConversationLLM

    private var currentLabel: String? = null
    private var lastShownAt: Long = 0L
    private val minShowMillis = 2000L    // 최소 2초 간격
    private val isProcessing = AtomicBoolean(false)   // java.util.concurrent

    private val minProcessIntervalMs = 1000L          // 1초보다 자주 돌리지 않음
    private var lastProcessAt = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        arFragment = supportFragmentManager.findFragmentById(R.id.arFragment) as ArFragment
        arFragment.arSceneView.session?.let { session ->
            val config = session.config.apply {
                // CPU 이미지를 Frame.acquireCameraImage()로 꺼내려면 반드시 LATEST_CAMERA_IMAGE 모드로 설정
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            }
            session.configure(config)
        }

        objectLabel       = findViewById(R.id.objectLabel)
        translationView   = findViewById(R.id.translationView)
        chatResponse      = findViewById(R.id.chatResponse)
        boundingBox       = findViewById(R.id.boundingBox)
        infoCard          = findViewById(R.id.infoCard)

        objectRecognizer  = ObjectRecognizer(this)
        translator        = TranslationHelper(this, "en", "ko")
        llm               = ConversationLLM(this)

        if (hasCameraPermission()) {
            startFrameProcessing()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                1001
            )
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startFrameProcessing() {
        // startFrameProcessing() 안에서 onUpdateListener 교체
        arFragment.arSceneView.scene.addOnUpdateListener { _ ->
            val frame = arFragment.arSceneView.arFrame ?: return@addOnUpdateListener

            // 0.5초 쿨타임 & 이미 처리‑중이면 스킵
            val now = SystemClock.uptimeMillis()
            if (now - lastProcessAt < minProcessIntervalMs) return@addOnUpdateListener
            if (!isProcessing.compareAndSet(false, true))  return@addOnUpdateListener
            lastProcessAt = now

            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val bitmap = objectRecognizer.frameToBitmap(frame) ?: return@launch
                    val label  = objectRecognizer.recognize(bitmap)
                    if (label.isBlank() || label == currentLabel) return@launch

                    val translated = translator.translate(label)
                    val response   = llm.chat(label, translated)

                    launch(Dispatchers.Main) {
                        translationView.text = "$label\n$translated\n\n$response"
                        currentLabel = label
                        lastShownAt  = now
                    }
                } finally {
                    isProcessing.set(false)    // 반드시 해제
                }
            }
        }

    }


    private fun updateBoundingBox(frame: Frame) {
        // 화면 중앙 지점 계산
        val centerX = arFragment.arSceneView.width / 2f
        val centerY = arFragment.arSceneView.height / 2f

        // 화면 중앙에서 레이캐스팅하여 실제 물체 위치 추정
        val hits = frame.hitTest(centerX, centerY)
        val filteredHits = hits.filter { hit ->
            (hit.trackable is Plane && (hit.trackable as Plane).isPoseInPolygon(hit.hitPose))
        }

        if (filteredHits.isNotEmpty()) {
            // 바운딩 박스 크기 조정 (물체 크기에 맞게)
            val boxSize = 200 // 기본 크기
            
            // 바운딩 박스 위치 설정
            boundingBox.x = centerX - (boxSize / 2)
            boundingBox.y = centerY - (boxSize / 2)
            
            // 바운딩 박스 크기 설정
            boundingBox.layoutParams.width = boxSize
            boundingBox.layoutParams.height = boxSize
            boundingBox.requestLayout()
            
            // 바운딩 박스 표시
            boundingBox.visibility = View.VISIBLE
        } else {
            boundingBox.visibility = View.GONE
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && grantResults.isNotEmpty() && 
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startFrameProcessing()
        }
    }
}
