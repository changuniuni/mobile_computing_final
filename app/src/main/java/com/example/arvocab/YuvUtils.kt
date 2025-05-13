package com.example.arvocab

import android.graphics.*
import android.media.Image
import java.io.ByteArrayOutputStream

object YuvUtils {

    /** YUV_420_888 → Bitmap (row‑stride/pixel‑stride 안전) */
    fun yuvToBitmap(image: Image): Bitmap {
        val nv21 = yuv420888ToNv21(image)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    /** 안전한 row‑stride 변환 */
    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width  = image.width
        val height = image.height
        val ySize  = width * height
        val nv21   = ByteArray(ySize + 2 * (width / 2) * (height / 2))

        // ----- Y -----
        val yPlane      = image.planes[0]
        val yRowStride  = yPlane.rowStride
        val yBuffer     = yPlane.buffer
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }

        // ----- U & V (interleave to NV21: VU VU ...) -----
        val uPlane          = image.planes[1]
        val vPlane          = image.planes[2]
        val uvRowStride     = uPlane.rowStride
        val uvPixelStride   = uPlane.pixelStride   // == vPlane.pixelStride
        val vBuffer         = vPlane.buffer
        val uBuffer         = uPlane.buffer

        for (row in 0 until height / 2) {
            var col = 0
            while (col < width / 2) {
                val vuIndex = row * uvRowStride + col * uvPixelStride
                nv21[pos++] = vBuffer.get(vuIndex)        // V
                nv21[pos++] = uBuffer.get(vuIndex)        // U
                col++
            }
        }
        return nv21
    }
}