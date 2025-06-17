package com.example.arvocab

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import com.example.arvocab.BuildConfig

/**
 * Google Gemini API based conversational assistant
 * Provides contextual language learning assistance
 */
class ConversationLLM(private val context: Context) {
    
    private val geminiApiService: GeminiApiService
    
    companion object {
        private const val TAG = "ConversationLLM"
        private const val GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/"
        // The API key is now securely accessed from BuildConfig.
        private val GEMINI_API_KEY = BuildConfig.GEMINI_API_KEY
    }
    
    init {
        Log.i(TAG, "ConversationLLM constructor called - initializing Gemini API Service")
        
        val retrofit = Retrofit.Builder()
            .baseUrl(GEMINI_API_BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
                    .build()
            
        geminiApiService = retrofit.create(GeminiApiService::class.java)
    }
    
    suspend fun chat(original: String, translated: String, userMessage: String): String {
        return withContext(Dispatchers.IO) {
            Log.d(TAG, "Gemini chat request - original: '$original', message: '$userMessage'")

            // Construct a prompt for the Gemini model
            val prompt = buildString {
                append("You are AR Vocab, a concise language learning assistant.\n")
                append("Give short, direct answers. No asterisks (*) or bullet points.\n")
                append("For translations, just provide the word and brief explanation.\n\n")

                // Add AR context if available
                if (original.isNotEmpty() && original != "Unknown") {
                    append("Object: $original")
                    if (translated.isNotEmpty()) {
                        append(" (Korean: $translated)")
                    }
                    append("\n")
                }

                append("Question: $userMessage\n\n")
                append("Answer briefly and clearly. Keep it under 2 sentences.")
            }

            val request = GeminiRequest(contents = listOf(Content(parts = listOf(Part(text = prompt)))))

            try {
                val response = geminiApiService.generateContent(GEMINI_API_KEY, request)
                if (response.isSuccessful) {
                    val geminiResponse = response.body()
                    val responseText = geminiResponse?.candidates?.firstOrNull()?.content?.parts?.firstOrNull()?.text
                    
                    if (responseText != null) {
                        Log.i(TAG, "Gemini API response successful.")
                        return@withContext responseText
                    }
                }
                Log.e(TAG, "Gemini API call failed with response code: ${response.code()}, message: ${response.message()}")
                return@withContext "Sorry, I encountered an error while processing your request. (Code: ${response.code()})"
            } catch (e: Exception) {
                Log.e(TAG, "Failed to call Gemini API: ${e.message}", e)
                return@withContext "Sorry, I'm having trouble connecting to my brain right now. Please check your internet connection and try again."
            }
        }
    }

    suspend fun generateSpeech(text: String): ByteArray? {
        return withContext(Dispatchers.IO) {
            Log.d(TAG, "Generating speech for text: '$text'")

            val request = GeminiRequest(
                contents = listOf(Content(parts = listOf(Part(text = text)))),
                generationConfig = GenerationConfig(
                    responseModalities = listOf("AUDIO"),
                    speechConfig = SpeechConfig(
                        voiceConfig = VoiceConfig(
                            prebuiltVoiceConfig = PrebuiltVoiceConfig(
                                voiceName = "Kore" // 한국어 지원 음성
                            )
                        )
                    )
                )
            )

            try {
                val response = geminiApiService.generateSpeech(GEMINI_API_KEY, request)
                if (response.isSuccessful) {
                    val geminiResponse = response.body()
                    val audioData = geminiResponse?.candidates?.firstOrNull()?.content?.parts?.firstOrNull()?.inlineData?.data
                    
                    if (audioData != null) {
                        Log.i(TAG, "Speech generation successful")
                        return@withContext android.util.Base64.decode(audioData, android.util.Base64.DEFAULT)
                    }
                }
                Log.e(TAG, "Speech generation failed with response code: ${response.code()}")
                return@withContext null
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate speech: ${e.message}", e)
                return@withContext null
            }
        }
    }

    fun getStatus(): String {
        return if (GEMINI_API_KEY != "YOUR_GEMINI_API_KEY") {
            "✅ Gemini API Ready"
        } else {
            "❌ Gemini API Key not set"
        }
    }
}
