package com.example.arvocab

import android.content.Context
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.nl.translate.Translation
import kotlinx.coroutines.tasks.await

class TranslationHelper(context: Context, sourceLang: String, targetLang: String) {

    private val translator = Translation.getClient(
        TranslatorOptions.Builder()
            .setSourceLanguage(TranslateLanguage.fromLanguageTag(sourceLang) ?: TranslateLanguage.ENGLISH)
            .setTargetLanguage(TranslateLanguage.fromLanguageTag(targetLang) ?: TranslateLanguage.KOREAN)
            .build()
    )

    suspend fun translate(text: String): String {
        // Ensure model is downloaded; download if necessary
        translator.downloadModelIfNeeded().await()
        return translator.translate(text).await()
    }
}
