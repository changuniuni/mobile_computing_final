package com.example.arvocab

import android.content.Context

/**
 * Placeholder for on-device multimodal LLM (e.g., Qualcomm's Large Language and Vision Assistant).
 * Replace stubbed responses with real model inference once integrated.
 */
class ConversationLLM(context: Context) {

    fun chat(original: String, translated: String, userMessage: String): String {
        // 객체가 인식된 경우 사용자의 질문과 객체 정보를 함께 고려
        if (original.isNotEmpty() && original != "Unknown") {
            return "Regarding '$original' (which is '$translated' in Korean), you asked: \"$userMessage\". " +
                   "This is a placeholder response. A real LLM would provide more details here."
        }
        // 객체가 인식되지 않은 경우 사용자의 질문만 고려
        return "You asked: \"$userMessage\". I don't have a specific object in context right now. " +
               "This is a placeholder response from the LLM stub. " +
               "Point your camera at an object if you want to ask about something specific."
    }
}
