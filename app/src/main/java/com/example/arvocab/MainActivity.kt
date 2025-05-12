package com.example.arvocab

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.ar.sceneform.ux.ArFragment
import com.google.ar.core.Frame
import com.google.ar.sceneform.FrameTime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var arFragment: ArFragment
    private lateinit var translationView: TextView
    private lateinit var objectRecognizer: ObjectRecognizer
    private lateinit var translator: TranslationHelper
    private lateinit var llm: ConversationLLM

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        arFragment = supportFragmentManager.findFragmentById(R.id.arFragment) as ArFragment
        translationView = findViewById(R.id.translationView)

        objectRecognizer = ObjectRecognizer(this)
        translator = TranslationHelper(this, "en", "ko") // default English -> Korean
        llm = ConversationLLM(this)

        // permission & frame processing
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
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun startFrameProcessing() {
        arFragment.arSceneView.scene.addOnUpdateListener { frameTime ->
            val frame: Frame = arFragment.arSceneView.arFrame ?: return@addOnUpdateListener
            if (frame.timestamp % 15L != 0L) return@addOnUpdateListener

            lifecycleScope.launch(Dispatchers.Default) {
                val bitmap = objectRecognizer.frameToBitmap(frame) ?: return@launch
                val label = objectRecognizer.recognize(bitmap)
                if (label.isNullOrBlank()) return@launch
                val translated = translator.translate(label)
                val response = llm.chat(label, translated)

                launch(Dispatchers.Main) {
                    translationView.text = "\$label\n\$translated\n\nChat:\n\$response"
                }
            }
        }
    }
}
