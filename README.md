# ARVocabApp

An Android app that teaches vocabulary using Augmented Reality and on‑device AI.

* **ARCore** `1.48.0` for world tracking
* **Sceneform** `1.17.1` to render content in the AR scene
* **CameraX** `1.4.0` for frame access
* **TensorFlow Lite** (`LiteRT`) to run the OpenAI CLIP model you supply as `app/src/main/assets/clip.tflite`
* **ML Kit Translation** for offline multilingual translation
* **Qualcomm LLM (stub)** ready for integration

> **Important:** Copy your downloaded `clip.tflite` into `app/src/main/assets/` before you build.

Built & tested with **Android Studio Iguana | 2025.1.1 Canary** and **Gradle 8.3**.
