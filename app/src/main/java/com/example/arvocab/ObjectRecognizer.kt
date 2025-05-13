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

    private val interpreter: Interpreter = Interpreter(
            loadModelFile(context, "clip.tflite"),
            Interpreter.Options().apply { setNumThreads(4) }
    )
    private val tfliteLock = Any()


    // ─── Prompt Ensemble으로 평균화된 텍스트 임베딩 로드 ───
    private val LABELS: List<String> by lazy {
        context.assets.open("words.txt").bufferedReader().useLines { it.toList() }
    }

    private val textEmbeddings: Array<FloatArray> by lazy {
        // 1) assets/text_embed.npy 파일 열기
        val input = context.assets.open("text_embed.npy")

        // 2) magic + version + header length (총 10바이트) 읽기
        val preamble = ByteArray(10)
        input.read(preamble)
        // bytes 8~9 에 little-endian short 형태로 header 길이 저장
        val headerLen = ByteBuffer
            .wrap(preamble, 8, 2)
            .order(ByteOrder.LITTLE_ENDIAN)
            .short
            .toInt()

        // 3) 남은 헤더(headerLen 바이트) 건너뛰기
        input.skip(headerLen.toLong())

        // 4) 실제 float32 데이터 읽어서 ByteArray로
        val raw = input.readBytes()
        val bb  = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)

        // 5) 차원 계산: 총 float 개수 = raw.size/4
        val nLabels     = LABELS.size
        val totalFloats = raw.size / 4
        val dim         = totalFloats / nLabels

        // 6) [nLabels][dim] 형태로 배열에 채우기
        Array(nLabels) { i ->
            FloatArray(dim) { j ->
                bb.getFloat((i*dim + j)*4)
            }
        }
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
            // 1) 이미지 임베딩 추출
            interpreter.allocateTensors()

            // 1) 이미지 임베딩 추출
            val input = preprocess(bitmap)
            // output 채널 개수(=텍스트 임베딩 차원)
            val dims = interpreter.getOutputTensor(0).shape()[1]

            val output = Array(1) { FloatArray(dims) }
            synchronized(tfliteLock) {
                interpreter.run(input, output)
            }

            val imgEmb = output[0]
        
            // 2) L2 정규화
            val norm = kotlin.math.sqrt(imgEmb.fold(0f) { s, v -> s + v*v })
            if (norm > 0f) for (i in imgEmb.indices) imgEmb[i] /= norm
        
            // 3) Cosine Similarity로 최고 유사도 라벨 찾기
            var bestIdx = 0
            var bestSim = -1f
            textEmbeddings.forEachIndexed { idx, txtEmb ->
                var dot = 0f
                for (j in imgEmb.indices) dot += imgEmb[j] * txtEmb[j]
                if (dot > bestSim) {
                    bestSim = dot
                    bestIdx = idx
                }
            }
        
            // 4) 온도 스케일링 (선택)  
            val T = 0.8f  
            bestSim /= T  
        
            return LABELS.getOrElse(bestIdx) { "Unknown" }
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