package com.example.arvocab

import android.content.Context
import android.graphics.Bitmap
// import android.graphics.BitmapFactory // YuvUtils 사용으로 직접 필요 없어짐
// import android.graphics.ImageFormat // YuvUtils 사용으로 직접 필요 없어짐
// import android.graphics.Rect // YuvUtils 사용으로 직접 필요 없어짐
import android.media.Image
import android.util.Log // 로그 사용
// import com.google.ar.core.Frame // ARCore Frame은 더 이상 직접 사용되지 않음
import org.tensorflow.lite.Interpreter
// import java.io.ByteArrayOutputStream // YuvUtils 사용으로 직접 필요 없어짐
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
// import kotlin.math.min // 현재 코드에서 사용 안 함
import kotlin.math.sqrt

class ObjectRecognizer(private val context: Context) {

    // ─── 1) CLIP Image Encoder 로드 ─────────────────────────
    private val imageInterpreter: Interpreter = Interpreter(
        loadModelFile(context, "CLIPImageEncoder.tflite"),
        Interpreter.Options().apply { setNumThreads(4) }
    )
    private val tfliteLock = Any()

    // ─── 2) Prompt Ensemble 텍스트 임베딩 미리 로드 ───────────
    private val LABELS: List<String> by lazy {
        try {
            context.assets.open("words.txt")
                .bufferedReader()
                .useLines { it.filter { line -> line.isNotBlank() }.toList() }
        } catch (e: Exception) {
            Log.e("ObjectRecognizer", "Error loading labels from words.txt", e)
            emptyList<String>()
        }
    }

    private val textEmbeddings: Array<FloatArray> by lazy {
        if (LABELS.isEmpty()) {
            Log.e("ObjectRecognizer", "LABELS list is empty. Cannot load text embeddings.")
            return@lazy Array(0) { FloatArray(0) }
        }
        try {
            context.assets.open("text_embed.npy").use { input ->
                // magic + version + headerlen(총 10바이트) 읽고
                val preamble = ByteArray(10).also { input.read(it) }
                if (preamble[0] != 0x93.toByte() || String(preamble, 1, 5, Charsets.US_ASCII) != "NUMPY") {
                    Log.e("ObjectRecognizer", "Invalid NPY file magic string.")
                    return@lazy Array(0) { FloatArray(0) }
                }
                val headerLen = ByteBuffer.wrap(preamble, 8, 2)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .short.toInt()
                // .npy 헤더의 나머지 부분을 읽습니다. (파싱은 생략)
                val headerData = ByteArray(headerLen)
                input.read(headerData) // Consume the rest of the header
                // Log.d("ObjectRecognizer", "NPY Header: ${String(headerData, Charsets.UTF_8)}") // 헤더 내용 확인 (필요시)

                // 남은 바이트(float32) 읽기
                val raw = input.readBytes()
                val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)

                // (nLabels × dim) 채우기
                val nLabels = LABELS.size
                val totalFloats = raw.size / 4
                if (totalFloats % nLabels != 0) {
                    Log.e("ObjectRecognizer", "Total floats ($totalFloats) is not divisible by number of labels ($nLabels). Embedding data might be corrupt or mismatched.")
                    return@lazy Array(nLabels) { FloatArray(0) } // 또는 예외 발생
                }
                val dim = totalFloats / nLabels
                if (dim <= 0) {
                    Log.e("ObjectRecognizer", "Invalid embedding dimension: $dim")
                    return@lazy Array(nLabels) { FloatArray(0) }
                }

                Log.i("ObjectRecognizer", "Loading text embeddings: $nLabels labels, $dim dimensions.")

                Array(nLabels) { i ->
                    FloatArray(dim) { j ->
                        bb.getFloat((i * dim + j) * 4)
                    }
                }.also { embs ->
                    // 각 벡터 L2 정규화
                    embs.forEach { v ->
                        val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x })
                        if (norm > 0f) for (k in v.indices) v[k] /= norm
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("ObjectRecognizer", "Error loading text_embed.npy", e)
            Array(0) { FloatArray(0) }
        }
    }

    // assets/.tflite 파일을 메모리 매핑해서 ByteBuffer 로 반환
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

    // YUV_420_888 → Bitmap (YuvUtils 사용)
    // MainActivity에서 직접 호출할 수 있도록 public으로 변경
    fun yuvToBitmap(image: Image): Bitmap {
        return YuvUtils.yuvToBitmap(image) // YuvUtils.kt의 함수 호출
    }

    // ObjectRecognizer 내의 yuv420888ToNv21 메서드는 YuvUtils를 사용하므로 제거됨

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val size = 224
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)

        val buffer = ByteBuffer.allocateDirect(4 * size * size * 3)
            .order(ByteOrder.nativeOrder())

        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std  = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        for (y in 0 until size) {
            for (x in 0 until size) {
                val p = scaled.getPixel(x, y)
                val r = ((p shr 16 and 0xFF) / 255f - mean[0]) / std[0]
                val g = ((p shr  8 and 0xFF) / 255f - mean[1]) / std[1]
                val b = ((p       and 0xFF) / 255f - mean[2]) / std[2]
                buffer.putFloat(r).putFloat(g).putFloat(b)
            }
        }
        buffer.rewind()
        scaled.recycle() // 스케일된 비트맵 메모리 해제
        return buffer
    }

    // ─── 5) Bitmap → CLIPImageEncoder → 이미지 임베딩 → L2 정규화 → Zero-Shot 예측 ───
    fun recognize(bitmap: Bitmap): Pair<String, Float> {
        val inputBuffer = preprocess(bitmap)
        val outputShape = imageInterpreter.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer.allocateDirect(4 * outputShape[1]) // D dimension
            .order(ByteOrder.nativeOrder())

        synchronized(tfliteLock) {
            imageInterpreter.run(inputBuffer, outputBuffer)
        }
        outputBuffer.rewind()

        val imgEmb = FloatArray(outputShape[1]) { outputBuffer.getFloat() }
        val norm = sqrt(imgEmb.fold(0f) { acc, x -> acc + x * x })
        if (norm > 0f) for (k in imgEmb.indices) imgEmb[k] /= norm

        var bestSim = -Float.MAX_VALUE
        var bestIdx = -1

        if (textEmbeddings.isEmpty() || LABELS.isEmpty()) {
            Log.w("ObjectRecognizer", "Text embeddings or labels are empty. Cannot recognize.")
            return Pair("Unknown", 0f)
        }
        if (textEmbeddings[0].isEmpty()){
            Log.w("ObjectRecognizer", "Text embeddings dimension is zero. Cannot recognize.")
            return Pair("Unknown", 0f)
        }


        textEmbeddings.forEachIndexed { idx, txtEmb ->
            if (txtEmb.size != imgEmb.size) {
                Log.w("ObjectRecognizer", "Skipping text embedding at index $idx due to mismatched dimension. Img: ${imgEmb.size}, Txt: ${txtEmb.size}")
                return@forEachIndexed // 다음 텍스트 임베딩으로 넘어감
            }
            var currentDot = 0f
            for (k in imgEmb.indices) currentDot += imgEmb[k] * txtEmb[k]
            
            if (currentDot > bestSim) {
                bestSim = currentDot
                bestIdx = idx
            }
        }
        
        val bestLabel = if (bestIdx != -1 && bestIdx < LABELS.size) LABELS[bestIdx] else "Unknown"
        return Pair(bestLabel, if(bestSim == -Float.MAX_VALUE) 0f else bestSim)
    }
}