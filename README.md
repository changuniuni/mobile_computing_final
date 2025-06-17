# VocabVision: An Interactive AI-Powered Language Learning Assistant

This project is a mobile application for Android, designed to provide an immersive and interactive language learning experience. By leveraging real-time object recognition, advanced speech recognition, machine translation, and Google's Gemini AI, VocabVision transforms the user's surroundings into a dynamic vocabulary-building environment with natural conversation capabilities.

## 1. Development Environment

- **Platform**: Android (Min API 24)
- **Language**: Kotlin
- **IDE**: Android Studio
- **Build System**: Gradle
- **Key Libraries & Technologies**:
  - **UI**: Android UI Toolkit, RecyclerView, CardView, TextureView
  - **Camera**: Android `Camera2` API for real-time camera feed processing
  - **Asynchronous Programming**: Kotlin Coroutines for managing background tasks like ML model inference and network requests without blocking the main thread
  - **Machine Learning & AI**:
    - **Object Recognition**: TensorFlow Lite with a zero-shot CLIP-based model for on-device, real-time object identification.
    - **Conversational AI**: Google Gemini API (`gemini-1.5-flash-latest`) for natural language understanding and generation.
    - **Text-to-Speech**: Google Gemini API (`gemini-2.5-flash-preview-tts`) with 30 voice options and 24 language support.
    - **Speech Recognition**:
      - On-device TFLite Conformer model (`conformer_librispeech.tflite`) for high-accuracy, offline voice-to-text.
      - Android's native `RecognizerIntent` for standard speech recognition fallback.
    - **Machine Translation**: Google ML Kit Translation API for on-demand initial translation.

## 2. Motivation

Traditional language learning often involves rote memorization of vocabulary lists, which can be disengaging and lacks real-world context. This project was motivated by the desire to bridge the gap between abstract vocabulary learning and tangible, everyday objects.

The core idea is to create a contextual learning tool with natural conversation capabilities. When users see an object, they can simply point their phone at it to learn its name in a new language, hear how it's pronounced, ask questions naturally, and receive intelligent responses. This "see it, learn it, speak it" approach enhances memory retention and makes learning more intuitive and fun. By integrating camera overlays, conversational AI, and voice interaction, we aimed to create an active learning experience that goes beyond simple flashcard apps.

## 3. Service/Application Design

The application is designed as a multi-module system orchestrated by the `MainActivity`. The architecture prioritizes performance and memory efficiency, which is crucial for running multiple ML models on a mobile device while maintaining real-time AI conversation capabilities.

### 3.1. User Flow

1.  **Launch & Discover**: The user opens the app, which immediately activates the camera.
2.  **Object Recognition**: The app analyzes the camera feed in real-time. An on-device TensorFlow Lite model continuously scans for recognizable objects.
3.  **Recognition & Pause**: When an object is identified with sufficient confidence, the recognition process pauses to focus on the identified object. This conserves battery and processing power.
4.  **Overlay Display**: The English name of the object is displayed as an overlay (`objectLabelOverlay`) on the camera view.
5.  **Interactive Chat**: A chat interface slides into view, presenting the user with the recognized object's name and an invitation to ask questions.
6.  **Natural Conversation**: The user can interact using voice or text to:
    -   Ask for translations into other supported languages (e.g., "How do you say this in French?" or "중국어로 뭐야?")
    -   Request example sentences and explanations
    -   Ask follow-up questions naturally
7.  **AI Response**: The Google Gemini API processes the query and provides intelligent, contextual responses in the chat window.
8.  **Dynamic Translation Display**: When users ask for translations, the translated word appears in the center overlay, replacing the English word.
9.  **Voice Output**: Users can tap the speaker button to hear the AI's response spoken aloud using Gemini's TTS capabilities.
10. **New Object**: The user can tap a 'Refresh' button to un-pause recognition and start the process over with a new object.

### 3.2. System Architecture

-   **`MainActivity.kt`**: The main controller that manages the UI, camera lifecycle, voice interaction, and coordinates communication between all modules.
-   **`ObjectRecognizer.kt`**: A dedicated class for zero-shot object recognition. It processes camera frames, runs inference using a CLIP-based TFLite model, and identifies objects by comparing image and text embeddings.
-   **`ConversationLLM.kt`**: The AI conversation engine powered by the Google Gemini API. It handles natural language understanding, generates contextual responses, and integrates Text-to-Speech capabilities.
-   **`GeminiApiService.kt`**: Retrofit-based service interface for communicating with Google Gemini API endpoints.
-   **`TranslationHelper.kt`**: A helper class that uses Google's ML Kit to provide an initial translation of the recognized object.
-   **Advanced Speech Recognition**: Integrates both an on-device Conformer ASR model and Android's native speech recognition for robust voice input.
-   **Memory Management**: Implements sequential loading of ML models: the `ObjectRecognizer` is active first, then released when `ConversationLLM` is loaded to prevent memory pressure.

### 3.3. Technical Workflow & State Management

The application operates in two distinct phases, with enhanced AI conversation capabilities:

**Phase 1: Recognition Loop**
1.  **Initialization**: The `ObjectRecognizer` model is loaded and begins processing camera frames.
2.  **Continuous Inference**: Real-time object detection is throttled to prevent CPU overload.
3.  **State Transition**: When an object is recognized:
    -   Recognition pauses (`isPaused` flag set to `true`).
    -   `ObjectRecognizer` is released to free memory.
    -   `ConversationLLM` (Gemini API client) is loaded asynchronously.

**Phase 2: AI Interaction Loop**
1.  **UI Update**: The camera overlay shows the English object name, and the chat interface becomes active.
2.  **Voice/Text Input**: Users can interact via:
    -   **Voice Input**: On-device Conformer ASR or Android Speech Recognition.
    -   **Text Input**: Direct typing in the chat interface.
3.  **AI Processing**: User queries are sent to the Google Gemini API for intelligent processing.
4.  **Dynamic Response**:
    -   Text responses appear in the chat.
    -   Translation requests trigger dynamic overlay updates (e.g., English "cup" → Chinese "杯子").
    -   Voice output is available via Gemini TTS.
5.  **Continuous Learning**: Users can ask follow-up questions, request different languages, or explore related concepts.

#### Enhanced Workflow Diagram

```mermaid
sequenceDiagram
    participant User
    participant MainActivity (UI)
    participant ObjectRecognizer
    participant ConversationLLM
    participant GeminiAPI
    participant MLKit_Translate

    rect rgb(240, 240, 240)
        note over MainActivity, ObjectRecognizer: Phase 1: Object Recognition
        User->>MainActivity (UI): Launches App
        MainActivity (UI)->>ObjectRecognizer: Initializes & Starts Camera Feed
        loop Recognition Loop
            ObjectRecognizer->>ObjectRecognizer: Process Camera Frame & Infer
        end
        ObjectRecognizer->>MainActivity (UI): Object Detected ("cup")
        MainActivity (UI)->>ObjectRecognizer: Release Model (Free Memory)
        deactivate ObjectRecognizer
    end

    rect rgb(230, 255, 230)
        note over MainActivity, GeminiAPI: Phase 2: AI-Powered Interaction
        MainActivity (UI)->>ConversationLLM: Load Gemini API Client
        activate ConversationLLM
        MainActivity (UI)->>MLKit_Translate: Translate Label ("cup" -> "컵")
        MainActivity (UI)->>User: Display Overlay ("cup") & Chat UI
        User->>MainActivity (UI): Voice Input: "중국어로 뭐야?"
        MainActivity (UI)->>MainActivity (UI): Transcribe Speech (On-Device ASR)
        MainActivity (UI)->>ConversationLLM: Send Query ("중국어로 뭐야?")
        ConversationLLM->>GeminiAPI: Natural Language Processing
        GeminiAPI-->>ConversationLLM: "In Chinese, cup is 杯子 (bēizi)"
        ConversationLLM-->>MainActivity (UI): AI Response + Extracted Word
        MainActivity (UI)->>MainActivity (UI): Update Overlay ("cup" -> "杯子")
        MainActivity (UI)-->>User: Show AI response + Dynamic Translation
        User->>MainActivity (UI): Tap Speaker Button
        MainActivity (UI)->>ConversationLLM: Request TTS
        ConversationLLM->>GeminiAPI: Generate Speech (TTS)
        GeminiAPI-->>ConversationLLM: Audio Data
        ConversationLLM-->>MainActivity (UI): Play Audio Response
    end

    User->>MainActivity (UI): Taps Refresh
    MainActivity (UI)->>ObjectRecognizer: Re-initialize for new object
    activate ObjectRecognizer
```

## 4. Implementation Details

### 4.1. ObjectRecognizer
-   **Model**: `CLIPImageEncoder.tflite`, a TensorFlow Lite model based on the CLIP architecture.
-   **Method**: Implements zero-shot object recognition. It encodes the camera image into an embedding vector and calculates the cosine similarity against a set of pre-computed text embeddings (for words in `words.txt`). The label corresponding to the highest similarity score is selected.
-   **Input**: `Bitmap` image converted from the camera's `YUV_420_888` format.
-   **Output**: `Pair<String, Float>` containing the highest-confidence label and its similarity score.
-   **Performance**: Optimized for mobile devices with efficient memory management.

### 4.2. ConversationLLM (Gemini API Integration)
-   **Text Model**: `gemini-1.5-flash-latest` for generating conversational responses.
-   **TTS Model**: `gemini-2.5-flash-preview-tts` for high-quality speech synthesis.
-   **Prompt Engineering**: A detailed prompt provides context to the AI, including its persona ("VocabVision, a concise language learning assistant"), the recognized object, and instructions for brief, clear answers.
-   **Features**:
    -   **Intelligent Conversation**: Powered by Google's advanced language model.
    -   **Translation Detection**: Automatically detects translation requests in multiple languages.
    -   **Dynamic Word Extraction**: Extracts translated words from responses for overlay updates.
    -   **Text-to-Speech**: 30 voice options supporting 24 languages.

### 4.3. Advanced Speech Recognition
-   **Primary**: On-device Conformer model (`conformer_librispeech.tflite`).
-   **Post-processing**: The raw output from the Conformer model is decoded using a Greedy CTC algorithm to form the final transcribed text.
-   **Fallback**: Android's native `RecognizerIntent` for universal compatibility.
-   **Features**:
    -   High-accuracy offline transcription.
    -   Privacy-preserving on-device processing.
    -   Robust handling of natural conversation patterns.

### 4.4. Text-to-Speech System
-   **Engine**: Google Gemini API (`gemini-2.5-flash-preview-tts`).
-   **Capabilities**:
    -   30 different voice options (e.g., Kore, Puck, Charon).
    -   24 language support with automatic language detection.
    -   High-quality speech synthesis at 24kHz.
    -   Real-time audio streaming via Android's `AudioTrack`.

## 5. Key Features

### 5.1. Dynamic Translation Display
-   **Smart Detection**: Recognizes translation requests in multiple languages using a comprehensive list of keywords (e.g., "in chinese", "일본어로", "translate", "뭐야").
-   **Real-time Updates**: The camera overlay dynamically changes from the English label to the translated word upon request.
-   **Robust Extraction**: A sequence of advanced regex patterns reliably extracts the translated word from the AI's natural language response.
-   **Visual Learning**: Immediate visual feedback reinforces vocabulary acquisition.

### 5.2. Natural Voice Interaction
-   **Bidirectional Voice**: Both input (STT) and output (TTS) voice capabilities create a seamless conversational loop.
-   **Multilingual Support**: Handles voice queries in multiple languages.
-   **Contextual Understanding**: The AI understands the recognized object's context within the voice conversation.
-   **Seamless Integration**: Voice and text inputs can be used interchangeably.

### 5.3. Intelligent Memory Management
-   **Sequential Model Loading**: The `ObjectRecognizer` and `ConversationLLM` models are loaded and unloaded sequentially to prevent memory pressure on mobile devices.
-   **Dynamic Resource Allocation**: Models are loaded only when needed and released immediately after use, ensuring optimized performance and a smooth user experience.

## 6. Results

The VocabVision app successfully demonstrates the integration of multiple cutting-edge AI technologies into a cohesive and effective language learning tool.

### Achievements
-   **Real-time AI Conversation**: Achieved natural language interaction with an AI assistant, powered by the Google Gemini API.
-   **Dynamic Visual Learning**: The camera overlay successfully updates in real-time based on the conversational context, providing instant visual reinforcement.
-   **Multilingual Voice Support**: Implemented a complete voice interaction loop (STT -> AI -> TTS) supporting over 24 languages.
-   **Seamless User Experience**: Created an intuitive interface that fluidly combines camera-based object recognition, chat, and voice modalities.
-   **Robust Performance**: Efficient resource management enables complex AI features to run smoothly on a standard mobile device.

### Technical Accomplishments
-   **Hybrid AI Architecture**: Successfully combined on-device ML (TensorFlow Lite for object recognition and ASR) with cloud-based AI (Google Gemini API) for an optimal balance of performance, privacy, and capability.
-   **Advanced NLP**: Implemented robust translation request detection and word extraction from natural language responses using keyword matching and regular expressions.
-   **Real-time Audio Processing**: Integrated high-quality TTS from the Gemini API with Android's `AudioTrack` for low-latency audio playback.
-   **Cross-modal Interaction**: Built a system where visual (camera), text (chat), and audio (voice) inputs and outputs are seamlessly integrated.

## 7. Discussion

This project represents a significant advancement in mobile AI-powered language learning applications.

### Strengths
-   **Contextual AI Learning**: Combines object recognition with intelligent conversation for immersive, real-world learning.
-   **Natural Interaction**: A voice-first design with intelligent speech processing makes learning intuitive.
-   **Dynamic Visual Feedback**: The real-time translation display enhances memory retention and engagement.
-   **Cutting-edge Technology**: Leverages the latest Google AI technologies (Gemini API, advanced TTS, on-device Conformer).
-   **Privacy-Conscious**: On-device processing for object recognition and primary speech-to-text ensures user privacy.

### Future Enhancements
-   **Expanded Object Recognition**: Train the model on a larger vocabulary and more object categories.
-   **Personalized Learning**: Implement user progress tracking and adaptive learning paths.
-   **Offline AI Capabilities**: Explore using local language models for complete offline functionality.
-   **Advanced Pronunciation Training**: Incorporate features for detailed pronunciation feedback and coaching.

## 8. Setup Instructions

### 8.1. Prerequisites
-   Android Studio (latest version recommended)
-   Android device or emulator with API level 24+
-   A Google Gemini API key

### 8.2. Configuration
1.  Clone the repository from GitHub.
2.  Create a `local.properties` file in the root of the project.
3.  Add your Gemini API key to `local.properties`:
   ```
   GEMINI_API_KEY="your_actual_gemini_api_key_here"
   ```
4.  Build and run the application on your device or emulator.

### 8.3. Required Permissions
-   `CAMERA`: For real-time object recognition.
-   `RECORD_AUDIO`: For voice input functionality.
-   `INTERNET`: For communication with the Gemini API.

The VocabVision app represents the future of AI-powered language learning, combining the latest advances in computer vision, natural language processing, and speech technology to create an immersive and intelligent learning experience. 