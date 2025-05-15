# ARVocabApp (On-Device AI Edition)

This Android application helps with vocabulary learning using on-device AI, without relying on Augmented Reality (AR) features. It uses the device's camera to recognize objects in real-time, providing their names, translations, and a simple (currently placeholder) description.

## Core Features

*   **Real-time Object Recognition**: Identifies objects in the camera's view.
*   **Multilingual Translation**: Translates recognized object names (currently configured for English to Korean).
*   **LLM Integration (Basic Stub)**: A foundational stub for Large Language Model (LLM) integration is included to provide more information about recognized objects (currently returns a placeholder response).
*   **Fully On-Device**: All AI models (object recognition, translation) run directly on the device without requiring an internet connection.

## Key Technical Stack

*   **Camera2 API**: Directly accesses the Android camera hardware for image frame input.
*   **TensorFlow Lite (Runtime)**: Executes deep learning models on-device.
    *   **OpenAI CLIP (Image Encoder part)**: Used for extracting image features. The model file should be named `CLIPImageEncoder.tflite`.
*   **ML Kit Translation**: Google's on-device translation library for offline text translation.
*   **Kotlin**: The primary development language.
*   **Coroutines**: For managing asynchronous operations efficiently.

## Setup and Required Assets

Before building and running the application, ensure the following files are placed in the `app/src/main/assets/` directory:

1.  `CLIPImageEncoder.tflite`: The image encoder part of the OpenAI CLIP model, in TensorFlow Lite format.
2.  `words.txt`: A text file containing the list of object labels to be recognized. Each label should be on a new line. (Refer to `app/src/main/assets/words.txt` for an example).
3.  `text_embed.npy`: A NumPy array file (`.npy`) containing pre-computed text embeddings for each label in `words.txt`. These embeddings must be generated using the CLIP text encoder, correspond to the order of labels in `words.txt`, and should be L2 normalized. The `ObjectRecognizer.kt` class loads this file to compare against image embeddings.

> **Important**: If these files are missing or not correctly placed, the application may not function correctly, or object recognition will fail.

## How it Works

The application follows these main steps to recognize objects and display information:

1.  **Camera Initialization (`MainActivity.kt`)**:
    *   The app requests camera permission upon launch.
    *   Once permission is granted and the `TextureView` for preview is available, `openCamera()` is called.
    *   `openCamera()` (see `MainActivity.kt` lines 204-238) configures the camera:
        *   It selects the back-facing camera.
        *   It determines an appropriate preview size (e.g., 640x480) for the `YUV_420_888` image format.
        *   An `ImageReader` is set up with this size and format to receive camera frames for analysis (see `MainActivity.kt` lines 224-226).
        *   The `ImageReader`'s `OnImageAvailableListener` (see `MainActivity.kt` lines 97-140) is set to process new frames.
        *   The `CameraDevice` is opened.

2.  **Camera Preview Session (`MainActivity.kt`)**:
    *   When the camera device is successfully opened, `createCameraPreviewSession()` is called (see `MainActivity.kt` lines 256-289).
    *   This function sets up a `Surface` from the `TextureView` for displaying the live camera feed.
    *   A `CaptureRequest` is created, targeting both the preview `Surface` and the `ImageReader`'s `Surface`.
    *   A `CameraCaptureSession` starts a repeating request, continuously streaming camera frames to both the live preview and the `ImageReader` for processing.

3.  **Frame Acquisition and Pre-processing (`MainActivity.kt`, `ObjectRecognizer.kt`, `YuvUtils.kt`)**:
    *   The `ImageReader.OnImageAvailableListener` is triggered whenever a new camera frame is ready.
    *   It acquires the latest `Image` (in `YUV_420_888` format).
    *   Processing is throttled (default 1 second interval) to prevent system overload.
    *   On a background thread (Kotlin Coroutine with `Dispatchers.Default`):
        *   The YUV `Image` is converted to a `Bitmap` using `objectRecognizer.yuvToBitmap(image)` (see `MainActivity.kt` line 109), which internally calls `YuvUtils.yuvToBitmap(image)` (see `app/src/main/java/com/example/arvocab/YuvUtils.kt` lines 10-17). The `Image` object is closed immediately after conversion.

4.  **Object Recognition (`ObjectRecognizer.kt`)**:
    *   The `Bitmap` is passed to `objectRecognizer.recognize(bitmap)` (see `app/src/main/java/com/example/arvocab/ObjectRecognizer.kt` lines 145-189).
    *   **Image Preprocessing**: Inside `recognize`, the `preprocess(bitmap)` method (see `app/src/main/java/com/example/arvocab/ObjectRecognizer.kt` lines 120-142) scales the bitmap to 224x224 pixels and normalizes its pixel values according to the CLIP model's requirements. The result is a `ByteBuffer`.
    *   **Inference**:
        *   The preprocessed `ByteBuffer` is fed into the TensorFlow Lite `imageInterpreter` (loaded from `CLIPImageEncoder.tflite`).
        *   The model outputs an image embedding (a float array representing image features).
        *   This image embedding is then L2 normalized.
    *   **Zero-Shot Classification**:
        *   The normalized image embedding is compared (using dot product similarity) against a list of pre-loaded, L2-normalized text embeddings. These text embeddings (`textEmbeddings` loaded from `text_embed.npy`, see `app/src/main/java/com/example/arvocab/ObjectRecognizer.kt` lines 42-98) correspond to the vocabulary words in `words.txt`.
        *   The word (label) whose text embedding has the highest similarity score with the image embedding is chosen as the `recognizedLabel`.
    *   The `Bitmap` used for recognition is recycled to free memory.

5.  **Displaying Results (`MainActivity.kt`)**:
    *   Back in `MainActivity.kt` (now on `Dispatchers.Main` for UI updates, see lines 117-131):
        *   If a new, valid label is recognized with a confidence score above a certain threshold (e.g., 0.1):
            *   The `objectLabel` TextView is updated with the current time, the `recognizedLabel`, and its score.
            *   The `infoCard` containing the information is made visible.
            *   (Currently commented out in `MainActivity.kt` lines 124-130) The recognized label can then be passed to:
                *   `translator.translate()` (see `app/src/main/java/com/example/arvocab/TranslationHelper.kt` lines 18-22) for translation.
                *   `llm.getResponse()` (see `app/src/main/java/com/example/arvocab/ConversationLLM.kt` lines 11-14) for a descriptive (placeholder) text.
                *   The results would update `translationView` and `chatResponse` TextViews.

6.  **Resource Management (`MainActivity.kt`)**:
    *   The camera and related resources are released in `onPause()` via `closeCamera()` (see `MainActivity.kt` lines 240-254).
    *   A dedicated background thread for camera operations is managed through `startBackgroundThread()` and `stopBackgroundThread()` in `onResume()` and `onPause()` respectively (see `MainActivity.kt` lines 181-195).

## Build Environment

*   **Android Studio**: Iguana | 2025.1.1 Canary (or compatible)
*   **Gradle Version**: 8.12 (as per `gradle-wrapper.properties`)
*   **Min SDK**: 24
*   **Target SDK**: 34

## Key Library Versions (refer to `app/build.gradle`)

*   `org.tensorflow:tensorflow-lite:2.14.0`
*   `com.google.mlkit:translate:17.0.2`
*   `androidx.camera:camera-camera2:1.4.0` (Note: Camera2 API is used directly, CameraX dependencies might be present but not central to the described object recognition flow)
*   `org.jetbrains.kotlin:kotlin-stdlib:1.9.22`
*   `androidx.lifecycle:lifecycle-runtime-ktx:2.7.0`
