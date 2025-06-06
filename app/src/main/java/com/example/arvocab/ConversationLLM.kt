package com.example.arvocab

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.tasks.await
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Google ML Kit Translation API based conversational assistant
 * Provides high-quality translation and language learning assistance
 */
class ConversationLLM(private val context: Context) {
    
    private var englishToFrenchTranslator: Translator? = null
    private var englishToSpanishTranslator: Translator? = null
    private var englishToGermanTranslator: Translator? = null
    private var englishToKoreanTranslator: Translator? = null
    private var englishToJapaneseTranslator: Translator? = null
    
    private val isInitialized = AtomicBoolean(false)
    private val isInitializing = AtomicBoolean(false)
    
    companion object {
        private const val TAG = "ConversationLLM"
    }
    
    init {
        Log.i(TAG, "ConversationLLM constructor called - initializing ML Kit Translation")
        initializeTranslators()
    }
    
    private fun initializeTranslators() {
        if (!isInitializing.compareAndSet(false, true)) {
            Log.d(TAG, "Translation initialization already in progress")
            return
        }
        
        try {
            Log.i(TAG, "Creating ML Kit translators...")
            
            // Create translators for different languages
            englishToFrenchTranslator = Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(TranslateLanguage.ENGLISH)
                    .setTargetLanguage(TranslateLanguage.FRENCH)
                    .build()
            )
            
            englishToSpanishTranslator = Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(TranslateLanguage.ENGLISH)
                    .setTargetLanguage(TranslateLanguage.SPANISH)
                    .build()
            )
            
            englishToGermanTranslator = Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(TranslateLanguage.ENGLISH)
                    .setTargetLanguage(TranslateLanguage.GERMAN)
                    .build()
            )
            
            englishToKoreanTranslator = Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(TranslateLanguage.ENGLISH)
                    .setTargetLanguage(TranslateLanguage.KOREAN)
                    .build()
            )
            
            englishToJapaneseTranslator = Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(TranslateLanguage.ENGLISH)
                    .setTargetLanguage(TranslateLanguage.JAPANESE)
                    .build()
            )
            
            isInitialized.set(true)
            Log.i(TAG, "✅ ML Kit translators initialized successfully")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize translators: ${e.message}", e)
            isInitialized.set(false)
        } finally {
            isInitializing.set(false)
        }
    }
    
    private suspend fun downloadModelIfNeeded(translator: Translator): Boolean {
        return try {
            val conditions = DownloadConditions.Builder()
                .requireWifi()
                .build()
            translator.downloadModelIfNeeded(conditions).await()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Failed to download translation model: ${e.message}")
            false
        }
    }
    
    private fun detectTargetLanguage(userMessage: String): String {
        val lowerMessage = userMessage.toLowerCase()
        return when {
            lowerMessage.contains("french") || lowerMessage.contains("français") -> "french"
            lowerMessage.contains("spanish") || lowerMessage.contains("español") -> "spanish"
            lowerMessage.contains("german") || lowerMessage.contains("deutsch") -> "german"
            lowerMessage.contains("korean") || lowerMessage.contains("한국어") -> "korean"
            lowerMessage.contains("japanese") || lowerMessage.contains("日本語") -> "japanese"
            else -> "french" // Default to French
        }
    }
    
    private suspend fun translateWithMLKit(text: String, targetLanguage: String): String? {
        val translator = when (targetLanguage) {
            "french" -> englishToFrenchTranslator
            "spanish" -> englishToSpanishTranslator
            "german" -> englishToGermanTranslator
            "korean" -> englishToKoreanTranslator
            "japanese" -> englishToJapaneseTranslator
            else -> englishToFrenchTranslator
        }
        
        return translator?.let { trans ->
            try {
                // Download model if needed
                if (downloadModelIfNeeded(trans)) {
                    trans.translate(text).await()
                } else {
                    null
                }
            } catch (e: Exception) {
                Log.e(TAG, "Translation failed: ${e.message}")
                null
            }
        }
    }
    
    suspend fun chat(original: String, translated: String, userMessage: String): String {
        return withContext(Dispatchers.IO) {
            Log.d(TAG, "ML Kit Translation chat request - original: '$original', message: '$userMessage'")
            
            if (!isInitialized.get()) {
                Log.w(TAG, "Translators not initialized, using fallback")
                return@withContext generateFallbackResponse(original, translated, userMessage)
            }
            
            val targetWord = original.takeIf { it.isNotEmpty() && it != "Unknown" } ?: "word"
            val targetLanguage = detectTargetLanguage(userMessage)
            val lowerMessage = userMessage.toLowerCase()
            
            return@withContext when {
                lowerMessage.contains("translate") || lowerMessage.contains("french") || 
                lowerMessage.contains("spanish") || lowerMessage.contains("german") ||
                lowerMessage.contains("korean") || lowerMessage.contains("japanese") -> {
                    val translation = translateWithMLKit(targetWord, targetLanguage)
                    if (translation != null) {
                        "✅ **Translation**\n\n" +
                        "**English:** $targetWord\n" +
                        "**${targetLanguage.capitalize()}:** $translation\n\n" 
                    } else {
                        "❌ Translation failed. Please check your internet connection and try again."
                    }
                }
                
                lowerMessage.contains("mean") || lowerMessage.contains("definition") -> {
                    generateDefinitionResponse(targetWord)
                }
                
                lowerMessage.contains("example") -> {
                    generateExampleResponse(targetWord)
                }
                
                lowerMessage.contains("pronounce") -> {
                    generatePronunciationResponse(targetWord)
                }
                
                else -> {
                    // Default: provide translation in multiple languages
                    val frenchTranslation = translateWithMLKit(targetWord, "french")
                    val spanishTranslation = translateWithMLKit(targetWord, "spanish")
                    val germanTranslation = translateWithMLKit(targetWord, "german")
                    
                    buildString {
                        append("🌍 **Translations for: $targetWord**\n\n")
                        frenchTranslation?.let { append("🇫🇷 French: **$it**\n") }
                        spanishTranslation?.let { append("🇪🇸 Spanish: **$it**\n") }
                        germanTranslation?.let { append("🇩🇪 German: **$it**\n") }
                        append("\n💡 Powered by Google ML Kit Translation")
                    }
                }
            }
        }
    }
    
    private fun generateDefinitionResponse(word: String): String {
        val definitions = mapOf(
            "door" to "a hinged, sliding, or revolving barrier at the entrance to a building, room, or vehicle",
            "window" to "an opening in the wall or roof of a building or vehicle that is fitted with glass",
            "house" to "a building for human habitation, especially one that consists of a ground floor and one or more upper storeys",
            "car" to "a road vehicle, typically with four wheels, powered by an internal combustion engine",
            "book" to "a written or printed work consisting of pages glued or sewn together along one side",
            "laptop" to "a computer that is portable and suitable for use while traveling",
            "chair" to "a separate seat for one person, typically with a back and four legs",
            "table" to "a piece of furniture with a flat top and one or more legs"
        )
        
        val definition = definitions[word.toLowerCase()]
        return if (definition != null) {
            "📖 **Definition of '$word':**\n\n$definition\n\n💡 Ask me to translate this word to other languages!"
        } else {
            "📖 **About '$word':**\n\nI don't have a specific definition for this word, but I can translate it to various languages for you!"
        }
    }
    
    private fun generateExampleResponse(word: String): String {
        val examples = mapOf(
            "door" to "Please close the door behind you.",
            "window" to "She looked out the window at the garden.",
            "house" to "They bought a beautiful house near the beach.",
            "car" to "He drives his car to work every day.",
            "book" to "I'm reading an interesting book about history.",
            "laptop" to "She works on her laptop in the coffee shop.",
            "chair" to "Please pull up a chair and join us.",
            "table" to "The books are on the table."
        )
        
        val example = examples[word.toLowerCase()]
        return if (example != null) {
            "📝 **Example sentence with '$word':**\n\n\"$example\"\n\n💡 Try asking for translations of this word!"
        } else {
            "📝 **Using '$word' in context:**\n\nI can help you translate this word to different languages instead!"
        }
    }
    
    private fun generatePronunciationResponse(word: String): String {
        return "🔊 **Pronunciation of '$word':**\n\n" +
                "For accurate pronunciation, I recommend:\n" +
                "• Using Google Translate's audio feature\n" +
                "• Checking online pronunciation dictionaries\n" +
                "• Using language learning apps with audio\n\n" +
                "💡 Would you like me to translate '$word' to other languages?"
    }
    
    private fun generateFallbackResponse(original: String, translated: String, userMessage: String): String {
        val targetWord = original.takeIf { it.isNotEmpty() && it != "Unknown" } ?: "this word"
        return "🔄 **Translation service temporarily unavailable**\n\n" +
                "I'm having trouble accessing the translation service right now. " +
                "Please check your internet connection and try again.\n\n" +
                "💡 You can also try asking about '$targetWord' again in a moment."
    }
    
    fun isReady(): Boolean = isInitialized.get()
    
    fun getStatus(): String = when {
        isInitialized.get() -> "✅ ML Kit Translation Ready"
        isInitializing.get() -> "⏳ Initializing ML Kit..."
        else -> "❌ Translation Service Unavailable"
    }
    
    fun getDetailedStatus(): String {
        return when {
            isInitialized.get() -> "✅ Google ML Kit Translation\n🌍 50+ languages supported\n📱 On-device processing"
            isInitializing.get() -> "⏳ Initializing translators...\n🔄 Setting up ML Kit services"
            else -> "❌ Service Unavailable\n🔧 Please restart the app"
        }
    }
}
