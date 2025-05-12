package com.example.arvocab

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.graphics.ImageFormat
import com.google.ar.core.Frame
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ObjectRecognizer(context: Context) {

    private val interpreter: Interpreter
    private val labels = listOf(
        "apple", "banana", "book", "keyboard", "mouse", "cup", "chair", "table"
    )

    init {
        // ① Interpreter 설정: 스레드 수 지정
        val options = Interpreter.Options().apply { setNumThreads(4) }
        // ② .tflite 모델을 MappedByteBuffer 로 메모리 매핑
        interpreter = Interpreter(loadModelFile(context, "clip.tflite"), options)
    }

    /** assets/clip.tflite 을 메모리 매핑하여 ByteBuffer 로 반환 */
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        FileInputStream(fd.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }
    }

    /**
     * ARCore Frame → Bitmap
     * 1) CPU 이미지 획득 (YUV_420_888)
     * 2) NV21 배열로 변환
     * 3) YuvImage → JPEG → BitmapFactory.decode
     */
    fun frameToBitmap(frame: Frame): Bitmap? {
        // Frame.acquireCameraImage() 사용 전, 반드시 LATEST_CAMERA_IMAGE 모드로 세션 구성 필요
        val image: Image = try {
            frame.acquireCameraImage()
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }

        // use{} 블록 끝에서 image.close() 자동 호출
        return image.use {
            yuvToBitmap(it)
        }
    }

    /** YUV_420_888 Image → NV21 바이트 배열 → JPEG 압축 → Bitmap 리턴 */
    private fun yuvToBitmap(image: Image): Bitmap {
        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        // 품질 90%, 속도가 중요하면 낮춰도 무방
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    /** YUV_420_888 → NV21 포맷(Byte 배열) 변환 */
    private fun yuv420888ToNv21(image: Image): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val ySize = yPlane.buffer.remaining()
        val uSize = uPlane.buffer.remaining()
        val vSize = vPlane.buffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Y 채널 복사
        yPlane.buffer.get(nv21, 0, ySize)
        // V 채널 복사 (NV21 순서: Y + V + U)
        vPlane.buffer.get(nv21, ySize, vSize)
        // U 채널 복사
        uPlane.buffer.get(nv21, ySize + vSize, uSize)

        return nv21
    }

    /** Bitmap → 모델 입력 포맷(ByteBuffer) → 라벨 예측 */
    fun recognize(bitmap: Bitmap): String {
        val input = preprocess(bitmap)
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        val idx = output[0].indices.maxByOrNull { output[0][it] } ?: -1
        return if (idx >= 0) labels[idx] else ""
    }

    /** 224×224 float32 RGB → ByteBuffer ([-1,1] 정규화) */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val size = 224
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val buffer = ByteBuffer.allocateDirect(4 * size * size * 3).apply {
            order(ByteOrder.nativeOrder())
        }

        for (y in 0 until size) {
            for (x in 0 until size) {
                val p = scaled.getPixel(x, y)
                buffer.putFloat(((p shr 16 and 0xFF) - 127.5f) / 127.5f)
                buffer.putFloat(((p shr 8  and 0xFF) - 127.5f) / 127.5f)
                buffer.putFloat(((p       and 0xFF) - 127.5f) / 127.5f)
            }
        }
        buffer.rewind()
        return buffer
    }
}